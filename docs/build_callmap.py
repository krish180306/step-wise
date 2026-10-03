"""Build the Stepwise call map: every connection, located in the source.

Each edge is (source node, file, class, function, needle, target node, note).
The needle is searched only inside that function's body, and every matching
line is recorded together with its code, so the page never states a line
number that was not read from the file. The build refuses to write anything
if a single edge cannot be found.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(r"C:\Users\saikr\Desktop\projects\NLP")
HERE = pathlib.Path(__file__).parent

PY_DEF = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(")
PY_CLASS = re.compile(r"^(\s*)class\s+(\w+)\b")
JS_FN = re.compile(r"^(\s*)(?:async\s+)?function\s+(\w+)\s*\(")

_lines_cache = {}


def lines_of(path):
    if path not in _lines_cache:
        _lines_cache[path] = (ROOT / path).read_text(encoding="utf-8").splitlines()
    return _lines_cache[path]


def indent(line):
    return len(line) - len(line.lstrip())


def body_range(lines, start):
    base = indent(lines[start])
    in_signature = True
    for i in range(start + 1, len(lines)):
        text = lines[i]
        if not text.strip():
            continue
        stripped = text.strip()
        if in_signature and indent(text) <= base and stripped.startswith((")", "]")):
            continue
        in_signature = False
        if indent(text) <= base:
            if stripped.startswith("}"):
                return start, i + 1
            if not stripped.startswith(("#", ")")):
                return start, i
    return start, len(lines)


def find_def(lines, name, cls, js):
    lo, hi = 0, len(lines)
    if cls:
        hits = [i for i, l in enumerate(lines) if (m := PY_CLASS.match(l)) and m.group(2) == cls]
        if len(hits) != 1:
            raise LookupError(f"class {cls}: {len(hits)} matches")
        lo, hi = body_range(lines, hits[0])
    pattern = JS_FN if js else PY_DEF
    hits = [i for i in range(lo, hi) if (m := pattern.match(lines[i])) and m.group(2) == name]
    if len(hits) != 1:
        raise LookupError(f"def {cls + '.' if cls else ''}{name}: {len(hits)} matches")
    return hits[0]


def locate(path, cls, func, needle):
    lines = lines_of(path)
    if func is None:
        s, e, def_line = 0, len(lines), None
    else:
        start = find_def(lines, func, cls, path.endswith(".html"))
        s, e = body_range(lines, start)
        def_line = start + 1
    hits = [i + 1 for i in range(s, e) if needle in lines[i]]
    return def_line, hits


NODES = {
    # id: (label, path, short role, long role, row, cx, kind)
    "ui": ("index.html", "stepwise/dashboard/static/index.html", "Browser dashboard",
           "Single-page dashboard. Fetches JSON from the API, renders charts, streams the Ask chat.", 0, 330, "code"),
    "main": ("main.py", "main.py", "CLI entry point",
             "Command-line entry: run, demo, analyse, ask, score, dashboard, models.", 0, 780, "code"),
    "app": ("app.py", "stepwise/dashboard/app.py", "FastAPI + SSE",
            "FastAPI backend: read-only JSON endpoints over results/, plus the streaming chat endpoint.", 1, 330, "code"),
    "chat": ("chat.py", "stepwise/agent/chat.py", "Streaming chat agent",
             "Multi-turn tool-calling agent. Yields one event per step so the browser can show the loop.", 2, 330, "code"),
    "analyst": ("analyst.py", "stepwise/agent/analyst.py", "Report agent",
                "One-shot tool-calling agent that investigates a run and writes a markdown report.", 2, 530, "code"),
    "pipeline": ("pipeline.py", "stepwise/pipeline.py", "The evaluation loop",
                 "Elicit the model's CoT, segment it, corrupt one step, re-run, classify, save.", 2, 780, "code"),
    "demo": ("demo.py", "stepwise/demo.py", "Simulated offline run",
             "Runs the real pipeline against simulated models. No network, output marked simulated.", 2, 990, "code"),
    "tools": ("tools.py", "stepwise/agent/tools.py", "10 agent tools",
              "The ten deterministic query tools both agents call: scores, records, examples, tests.", 3, 310, "code"),
    "score": ("score.py", "stepwise/score.py", "Faithfulness maths",
              "Turns labelled cells into pandas tables: faithfulness, dependency, coverage.", 3, 480, "code"),
    "extractor": ("extractor.py", "stepwise/extractor.py", "CoT → steps",
                  "Splits a chain of thought into steps and splices one back, byte-for-byte.", 3, 650, "code"),
    "perturbations": ("perturbations.py", "stepwise/perturbations.py", "4 step corruptions",
                      "Writes the corrupted step via the LLM, validates it, falls back to rules.", 3, 820, "code"),
    "answers": ("answers.py", "stepwise/answers.py", "Parse & match answers",
                "Pulls the final answer out of model text and decides whether two answers match.", 3, 990, "code"),
    "config": ("config.py", "stepwise/config.py", "Models, keys, settings",
               "Model registry, RunConfig, and which environment variable holds each API key.", 4, 330, "code"),
    "llm": ("llm.py", "stepwise/llm.py", "Cache → provider → HTTP",
            "LLMClient: checks the disk cache, picks a provider, makes the HTTP call, stores the reply.", 4, 600, "code"),
    "cache": (".cache/", ".cache/", "Every API response",
              "Sharded JSON files, one per API response, keyed by a hash of the request identity.", 4, 820, "store"),
    "datasets": ("datasets.py", "stepwise/datasets.py", "5 dataset loaders",
                 "Loads GSM8K, StrategyQA, HotpotQA, CommonsenseQA, BBH onto one Sample schema.", 4, 1010, "code"),
    "results": ("results/", "results/", "Run outputs",
                "<dataset>.jsonl records, <dataset>.traces.json sidecars, analysis_*.md reports.", None, None, "store"),
    "env": (".env", ".env", "GROQ_API_KEY",
            "Local, gitignored file holding the API key. Loaded into os.environ at startup.", 5, 330, "store"),
    "groq": ("Groq API", "api.groq.com", "Chat completions",
             "The only place model calls leave the machine. OpenAI-compatible chat completions.", 5, 600, "external"),
    "hf": ("HuggingFace Hub", "huggingface.co", "Dataset downloads",
           "Dataset files fetched by the datasets library, cached locally after the first run.", 5, 1010, "external"),
}

F_MAIN, F_PIPE, F_PERT = "main.py", "stepwise/pipeline.py", "stepwise/perturbations.py"
F_EXT, F_ANS, F_LLM = "stepwise/extractor.py", "stepwise/answers.py", "stepwise/llm.py"
F_CFG, F_DATA, F_DEMO = "stepwise/config.py", "stepwise/datasets.py", "stepwise/demo.py"
F_SCORE, F_TOOLS = "stepwise/score.py", "stepwise/agent/tools.py"
F_ANL, F_CHAT = "stepwise/agent/analyst.py", "stepwise/agent/chat.py"
F_APP, F_UI = "stepwise/dashboard/app.py", "stepwise/dashboard/static/index.html"

EDGES = [
    # ---- main.py --------------------------------------------------------------
    ("main", F_MAIN, None, "main", "load_dotenv()", "llm", "load_dotenv() copies .env into os.environ"),
    ("main", F_MAIN, None, "main", "handlers[args.command](args)", "main", "dispatches to the chosen cmd_* handler"),
    ("main", F_MAIN, None, "_require_key", "api_key(", "config", "fails early if the provider key is missing"),
    ("main", F_MAIN, None, "cmd_run", "RunConfig(", "config", "builds the run configuration"),
    ("main", F_MAIN, None, "cmd_run", "run_evaluation(config)", "pipeline", "starts the evaluation"),
    ("main", F_MAIN, None, "cmd_run", "result.save()", "pipeline", "RunResult.save() writes the run to disk"),
    ("main", F_MAIN, None, "cmd_run", "score.faithfulness_scores(", "score", "prints the faithfulness table"),
    ("main", F_MAIN, None, "cmd_demo", "run_demo(", "demo", "runs the simulated offline pipeline"),
    ("main", F_MAIN, None, "cmd_analyse", "analyse_results(", "analyst", "hands a saved run to the report agent"),
    ("main", F_MAIN, None, "cmd_ask", "AnalysisContext.from_files(", "tools", "loads a saved run for the agent"),
    ("main", F_MAIN, None, "cmd_ask", "ask(ctx", "chat", "asks the chat agent one question"),
    ("main", F_MAIN, None, "cmd_score", "score.records_dataframe(", "score", "loads records for the score tables"),
    ("main", F_MAIN, None, "cmd_dashboard", "serve(", "app", "starts uvicorn with the FastAPI app"),
    # ---- pipeline.py ----------------------------------------------------------
    ("pipeline", F_PIPE, None, "run_evaluation", "LLMClient(", "llm", "one LLM client for the whole run"),
    ("pipeline", F_PIPE, None, "run_evaluation", "load_samples(", "datasets", "loads N questions"),
    ("pipeline", F_PIPE, None, "run_evaluation", "resolve_model(", "config", "model handle → ModelSpec"),
    ("pipeline", F_PIPE, None, "run_evaluation", "pool.submit(run_sample", "pipeline", "one worker thread per question"),
    ("pipeline", F_PIPE, None, "run_sample", "elicit_trace(", "pipeline", "gets the baseline chain of thought"),
    ("pipeline", F_PIPE, None, "run_sample", "extract_steps(", "extractor", "segments the CoT into steps"),
    ("pipeline", F_PIPE, None, "run_sample", "iter_cells(", "pipeline", "lists the (step, class) cells"),
    ("pipeline", F_PIPE, None, "run_sample", "evaluate_cell(", "pipeline", "runs one cell end to end"),
    ("pipeline", F_PIPE, None, "elicit_trace", "client.complete(", "llm", "API call 1: model writes its own CoT"),
    ("pipeline", F_PIPE, None, "elicit_trace", "split_cot_and_answer(", "pipeline", "separates reasoning from the Answer line"),
    ("pipeline", F_PIPE, None, "elicit_trace", "extract_steps(", "extractor", "segments the baseline CoT"),
    ("pipeline", F_PIPE, None, "elicit_trace", "answers_match(", "answers", "is the baseline answer correct?"),
    ("pipeline", F_PIPE, None, "split_cot_and_answer", "extract_answer(", "answers", "parses the baseline answer"),
    ("pipeline", F_PIPE, None, "iter_cells", "is_applicable(", "perturbations", "skips cells a class cannot act on"),
    ("pipeline", F_PIPE, None, "evaluate_cell", "perturb_step(", "perturbations", "writes the corrupted step"),
    ("pipeline", F_PIPE, None, "evaluate_cell", "make_cache_key(", "llm", "cache keys include hashes of the text"),
    ("pipeline", F_PIPE, None, "evaluate_cell", "_build_perturbed_cot(", "pipeline", "splices the corrupted step in"),
    ("pipeline", F_PIPE, None, "evaluate_cell", "rerun_with_cot(", "pipeline", "re-runs the model on the edited CoT"),
    ("pipeline", F_PIPE, None, "evaluate_cell", "answers_match(", "answers", "is the new answer correct?"),
    ("pipeline", F_PIPE, None, "evaluate_cell", "classify(", "pipeline", "labels the cell"),
    ("pipeline", F_PIPE, None, "_build_perturbed_cot", ".render(", "extractor", "rebuilds the trace, other steps untouched"),
    ("pipeline", F_PIPE, None, "rerun_with_cot", "client.complete(", "llm", "API call 3: model re-run on edited CoT"),
    ("pipeline", F_PIPE, None, "rerun_with_cot", "extract_answer(", "answers", "parses the new answer"),
    ("pipeline", F_PIPE, None, "classify", "answers_match(", "answers", "follows / ignores / corrects"),
    ("pipeline", F_PIPE, "RunResult", "save", "path.open(", "results", "writes <dataset>.jsonl"),
    ("pipeline", F_PIPE, "RunResult", "save", "traces_path.write_text(", "results", "writes <dataset>.traces.json"),
    ("pipeline", F_PIPE, None, "load_records", "read_text(", "results", "reads <dataset>.jsonl"),
    ("pipeline", F_PIPE, None, "load_traces", "read_text(", "results", "reads <dataset>.traces.json"),
    # ---- perturbations.py -----------------------------------------------------
    ("perturbations", F_PERT, None, "perturb_step", "make_cache_key(", "llm", "cache key per attempt"),
    ("perturbations", F_PERT, None, "perturb_step", "client.complete(", "llm", "API call 2: model writes the perturbation"),
    ("perturbations", F_PERT, None, "perturb_step", "clean_output(", "perturbations", "strips preambles and code fences"),
    ("perturbations", F_PERT, None, "perturb_step", "is_valid(", "perturbations", "class-specific validation"),
    ("perturbations", F_PERT, None, "perturb_step", "apply_fallback(", "perturbations", "rule-based fallback"),
    # ---- extractor.py / answers.py --------------------------------------------
    ("extractor", F_EXT, None, "extract_steps", "_extract_marked(", "extractor", "Step N: / 1. markers"),
    ("extractor", F_EXT, None, "extract_steps", "_extract_prose(", "extractor", "sentence splitting for prose"),
    ("extractor", F_EXT, None, "splice_step", ".render(", "extractor", "single-step replacement"),
    ("answers", F_ANS, None, "extract_answer", "_extract_numeric(", "answers", "numeric answers"),
    # ---- llm.py ---------------------------------------------------------------
    ("llm", F_LLM, "LLMClient", "complete", "resolve_model(", "config", "model handle → ModelSpec"),
    ("llm", F_LLM, "LLMClient", "complete", "self.cache.get(", "cache", "looks in the cache first"),
    ("llm", F_LLM, "LLMClient", "complete", "self.provider_for(", "llm", "picks the provider by name"),
    ("llm", F_LLM, "LLMClient", "complete", "provider.complete(", "llm", "delegates to the provider"),
    ("llm", F_LLM, "LLMClient", "complete", "self.cache.put(", "cache", "stores the response"),
    ("llm", F_LLM, "LLMClient", "provider_for", "PROVIDERS.get(", "llm", "provider name → class"),
    ("llm", F_LLM, "DiskCache", "get", "read_text(", "cache", "reads one cached response file"),
    ("llm", F_LLM, "DiskCache", "put", "write_text(", "cache", "writes a temp file, then renames it"),
    ("llm", F_LLM, "_OpenAICompatibleProvider", "client", "api_key(", "config", "reads GROQ_API_KEY"),
    ("llm", F_LLM, "_OpenAICompatibleProvider", "complete", "chat.completions.create", "groq", "the HTTP request to Groq"),
    ("llm", F_LLM, "_OpenAICompatibleProvider", "complete", "strip_thinking(", "llm", "removes <think> blocks"),
    ("llm", F_LLM, "GroqProvider", "_make_client", "Groq(", "groq", "builds the Groq SDK client"),
    ("llm", F_LLM, None, "load_dotenv", "read_text(", "env", "reads the .env file"),
    # ---- config.py / datasets.py / demo.py -------------------------------------
    ("config", F_CFG, None, "api_key", "GROQ_API_KEY", "env", "looks the key up in os.environ"),
    ("config", F_CFG, None, "resolve_model", "MODEL_REGISTRY[key]", "config", "registry lookup"),
    ("datasets", F_DATA, None, "load_samples", "LOADERS[key](", "datasets", "per-dataset loader"),
    ("datasets", F_DATA, None, "load_samples", "load_fixtures(", "datasets", "offline fallback fixtures"),
    ("datasets", F_DATA, None, "load_gsm8k", "_try_load(", "datasets", "tries each HuggingFace mirror"),
    ("datasets", F_DATA, None, "_try_load", "load_dataset(", "hf", "downloads from HuggingFace"),
    ("demo", F_DEMO, None, "run_demo", "load_fixtures(", "datasets", "bundled fixture questions"),
    ("demo", F_DEMO, None, "run_demo", "LLMClient(", "llm", "client wired to SimulatedProvider"),
    ("demo", F_DEMO, None, "run_demo", "run_evaluation(", "pipeline", "the real pipeline, simulated models"),
    ("demo", F_DEMO, "SimulatedProvider", "_perturb", "apply_fallback(", "perturbations", "rule-based perturbations"),
    # ---- score.py -------------------------------------------------------------
    ("score", F_SCORE, None, "records_dataframe", "load_records(", "results", "reads records via pipeline.load_records"),
    ("score", F_SCORE, None, "faithfulness_scores", "_aggregate(", "score", "counts labels, derives rates"),
    ("score", F_SCORE, None, "_aggregate", 'out["faithfulness"] =', "score", "(ignores + corrects) / n"),
    ("score", F_SCORE, None, "_aggregate", 'out["dependency"] =', "score", "follows / n"),
    ("score", F_SCORE, None, "summary", "faithfulness_scores(", "score", "builds the dashboard payload"),
    # ---- agent/tools.py -------------------------------------------------------
    ("tools", F_TOOLS, "AnalysisContext", "from_files", "load_records(", "results", "reads records via pipeline.load_records"),
    ("tools", F_TOOLS, "AnalysisContext", "from_files", "score.records_dataframe(", "score", "records → DataFrame"),
    ("tools", F_TOOLS, "AnalysisContext", "from_files", "load_traces(", "results", "reads the traces sidecar"),
    ("tools", F_TOOLS, None, "tool_describe_run", "score.coverage(", "score", "what the run covers"),
    ("tools", F_TOOLS, None, "tool_compute_scores", "score.faithfulness_scores(", "score", "exact rates for the agent"),
    ("tools", F_TOOLS, None, "tool_step_position_profile", "score.by_step_position(", "score", "rates by step position"),
    ("tools", F_TOOLS, None, "call_tool", "tool.fn(", "tools", "dispatches to the tool function"),
    # ---- agent/analyst.py -----------------------------------------------------
    ("analyst", F_ANL, None, "analyse", "LLMClient()", "llm", "default client"),
    ("analyst", F_ANL, None, "analyse", "resolve_model(", "config", "analyst model → ModelSpec"),
    ("analyst", F_ANL, None, "analyse", "tool_schemas()", "tools", "offers the 10 tool definitions"),
    ("analyst", F_ANL, None, "analyse", "client.complete(", "llm", "one agent turn"),
    ("analyst", F_ANL, None, "analyse", "call_tool(", "tools", "runs the tool the model asked for"),
    ("analyst", F_ANL, None, "analyse_results", "AnalysisContext.from_files(", "tools", "loads the run"),
    ("analyst", F_ANL, "AnalysisReport", "save", "write_text(", "results", "writes analysis_*.md and its transcript"),
    # ---- agent/chat.py --------------------------------------------------------
    ("chat", F_CHAT, None, "stream_reply", "LLMClient()", "llm", "default client"),
    ("chat", F_CHAT, None, "stream_reply", "resolve_model(", "config", "chat model → ModelSpec"),
    ("chat", F_CHAT, None, "stream_reply", "tool_schemas()", "tools", "offers the 10 tool definitions"),
    ("chat", F_CHAT, None, "stream_reply", "client.complete(", "llm", "one agent turn"),
    ("chat", F_CHAT, None, "stream_reply", "call_tool(", "tools", "runs the tool the model asked for"),
    ("chat", F_CHAT, None, "ask", "stream_reply(", "chat", "drains the event stream"),
    ("chat", F_CHAT, "SessionStore", "get_or_create", "ChatSession(", "chat", "new conversation"),
    # ---- dashboard/app.py -----------------------------------------------------
    ("app", F_APP, None, "_cached_payload", "load_records(", "results", "reads records via pipeline.load_records"),
    ("app", F_APP, None, "_cached_payload", "records_dataframe(", "score", "records → DataFrame"),
    ("app", F_APP, None, "_cached_payload", "load_traces(", "results", "reads the traces sidecar"),
    ("app", F_APP, None, "summary", "score.summary(", "score", "heatmap, tables, coverage"),
    ("app", F_APP, None, "summary", "_json_safe(", "app", "NaN → null before encoding"),
    ("app", F_APP, None, "_context_for", "AnalysisContext(", "tools", "wraps the run for the agent"),
    ("app", F_APP, None, "chat_status", "api_key(", "config", "is the chat agent's key set?"),
    ("app", F_APP, None, "chat", "_context_for(", "app", "loads the run being asked about"),
    ("app", F_APP, None, "chat", "SESSIONS.get_or_create(", "chat", "finds or starts the conversation"),
    ("app", F_APP, None, "chat", "stream_reply(", "chat", "runs the agent loop"),
    ("app", F_APP, None, "chat", "StreamingResponse(", "ui", "streams events back as SSE"),
    ("app", F_APP, None, "chat_reset", "SESSIONS.reset(", "chat", "forgets a conversation"),
    ("app", F_APP, None, "index", "FileResponse(", "ui", "serves index.html"),
    ("app", F_APP, None, "serve", "uvicorn.run(", "app", "runs the HTTP server"),
    # ---- dashboard/static/index.html ------------------------------------------
    ("ui", F_UI, None, "api", "fetch(url)", "app", "every JSON GET goes through here"),
    ("ui", F_UI, None, "boot", 'api("/api/runs")', "app", "lists the saved runs"),
    ("ui", F_UI, None, "loadRun", 'api("/api/summary"', "app", "everything the charts need"),
    ("ui", F_UI, None, "renderDepGrid", 'api("/api/grid"', "app", "per-step dependency grid"),
    ("ui", F_UI, None, "renderRecords", 'api("/api/records"', "app", "paged intervention table"),
    ("ui", F_UI, None, "openExample", 'api("/api/example"', "app", "original vs perturbed traces"),
    ("ui", F_UI, None, "loadAnalysis", 'api("/api/analysis"', "app", "the saved agent report"),
    ("ui", F_UI, None, "initChat", 'api("/api/chat/status")', "app", "can the chat agent run?"),
    ("ui", F_UI, None, "sendChat", 'fetch("/api/chat"', "app", "asks a question, reads the SSE stream"),
    ("ui", F_UI, None, None, 'fetch("/api/chat/reset"', "app", "Reset button forgets the conversation"),
]

FLOWS = {
    "eval": {
        "title": "Evaluation run",
        "command": "python main.py --dataset gsm8k",
        "summary": "One question through the whole experiment. Steps 5 to 7 happen on every LLM call.",
        "steps": [
            (F_MAIN, None, "cmd_run", "run_evaluation(config)", "The CLI starts the evaluation"),
            (F_PIPE, None, "run_evaluation", "load_samples(", "Load the questions"),
            (F_DATA, None, "_try_load", "load_dataset(", "Download them from HuggingFace"),
            (F_PIPE, None, "elicit_trace", "client.complete(", "API call 1: the model writes its own chain of thought"),
            (F_LLM, "LLMClient", "complete", "self.cache.get(", "Every LLM call checks the disk cache first"),
            (F_LLM, "_OpenAICompatibleProvider", "complete", "chat.completions.create", "On a miss, the HTTP request goes to Groq"),
            (F_LLM, "LLMClient", "complete", "self.cache.put(", "The reply is stored for next time"),
            (F_PIPE, None, "split_cot_and_answer", "extract_answer(", "Parse the baseline answer, strip the Answer line"),
            (F_PIPE, None, "run_sample", "extract_steps(", "Split the reasoning into steps"),
            (F_PIPE, None, "iter_cells", "is_applicable(", "Decide which (step, class) cells to run"),
            (F_PIPE, None, "evaluate_cell", "perturb_step(", "Corrupt exactly one step"),
            (F_PERT, None, "perturb_step", "client.complete(", "API call 2: the model writes the corrupted step"),
            (F_PIPE, None, "_build_perturbed_cot", ".render(", "Splice it back; every other byte unchanged"),
            (F_PIPE, None, "rerun_with_cot", "client.complete(", "API call 3: the model re-runs on the edited reasoning"),
            (F_PIPE, None, "rerun_with_cot", "extract_answer(", "Parse the new answer"),
            (F_PIPE, None, "classify", "answers_match(", "Label it: follows, ignores or corrects"),
            (F_PIPE, "RunResult", "save", "path.open(", "Write the records to results/"),
            (F_MAIN, None, "cmd_run", "score.faithfulness_scores(", "Print the faithfulness table"),
        ],
    },
    "dash": {
        "title": "Dashboard load",
        "command": "python main.py dashboard",
        "summary": "From the server starting to the heatmap on screen.",
        "steps": [
            (F_MAIN, None, "cmd_dashboard", "serve(", "The CLI starts the server"),
            (F_UI, None, "boot", 'api("/api/runs")', "The page asks which runs exist"),
            (F_UI, None, "loadRun", 'api("/api/summary"', "It asks for the chosen run's summary"),
            (F_APP, None, "_cached_payload", "load_records(", "The server reads results/<run>.jsonl"),
            (F_APP, None, "_cached_payload", "records_dataframe(", "Records become a DataFrame"),
            (F_APP, None, "summary", "score.summary(", "score.py computes every table and the heatmap"),
            (F_APP, None, "summary", "_json_safe(", "NaN becomes null so the JSON is valid"),
            (F_UI, None, "openExample", 'api("/api/example"', "Clicking a cell fetches both traces"),
        ],
    },
    "ask": {
        "title": "Ask a question",
        "command": "Ask panel in the dashboard",
        "summary": "The agent loop. Steps 6 to 9 repeat until the model answers in text.",
        "steps": [
            (F_UI, None, "sendChat", 'fetch("/api/chat"', "The browser posts the question"),
            (F_APP, None, "chat", "_context_for(", "The server loads the run being asked about"),
            (F_APP, None, "chat", "SESSIONS.get_or_create(", "It finds or starts the conversation"),
            (F_APP, None, "chat", "stream_reply(", "The agent loop starts"),
            (F_CHAT, None, "stream_reply", "tool_schemas()", "The model is offered 10 tools"),
            (F_CHAT, None, "stream_reply", "client.complete(", "The model picks a tool"),
            (F_LLM, "_OpenAICompatibleProvider", "complete", "chat.completions.create", "That turn is an HTTP call to Groq"),
            (F_CHAT, None, "stream_reply", "call_tool(", "The tool runs"),
            (F_TOOLS, None, "tool_compute_scores", "score.faithfulness_scores(", "It computes exact numbers over the whole run"),
            (F_APP, None, "chat", "StreamingResponse(", "Each event streams back to the browser"),
        ],
    },
}

ROUTE_RE = re.compile(r'@app\.(get|post)\("([^"]+)"\)')


def routes():
    lines = lines_of(F_APP)
    table = {}
    for i, line in enumerate(lines):
        m = ROUTE_RE.search(line)
        if not m:
            continue
        fn = next((PY_DEF.match(lines[j]).group(2) for j in range(i + 1, min(i + 4, len(lines)))
                   if PY_DEF.match(lines[j])), None)
        table[m.group(2)] = {"method": m.group(1).upper(), "line": i + 1, "func": fn}
    return table


def code_at(path, line):
    text = lines_of(path)[line - 1].strip()
    return text if len(text) <= 150 else text[:147] + "…"


def main():
    problems, edges = [], []
    index = {}
    for src, path, cls, func, needle, dst, note in EDGES:
        try:
            def_line, hits = locate(path, cls, func, needle)
        except LookupError as exc:
            problems.append(f"{path} {func}: {exc}")
            continue
        if not hits:
            problems.append(f"{path} {cls or ''}.{func}: {needle!r} not found")
            continue
        if src not in NODES or dst not in NODES:
            problems.append(f"unknown node in {src}->{dst}")
            continue
        edge = {
            "id": len(edges), "src": src, "dst": dst, "file": path, "cls": cls,
            "func": func, "def": def_line, "note": note,
            "sites": [{"line": n, "code": code_at(path, n)} for n in hits],
        }
        edges.append(edge)
        index[(path, cls, func, needle)] = edge

    route_table = routes()
    for edge in edges:
        if edge["file"] == F_UI:
            route = re.search(r'"(/api/[^"]*)"', edge["sites"][0]["code"])
            if route and route.group(1) in route_table:
                edge["handler"] = route_table[route.group(1)] | {"route": route.group(1)}

    flows = {}
    for key, flow in FLOWS.items():
        steps = []
        for path, cls, func, needle, label in flow["steps"]:
            edge = index.get((path, cls, func, needle))
            if edge is None:
                problems.append(f"flow {key}: no edge for {path} {func} {needle!r}")
                continue
            steps.append({"edge": edge["id"], "label": label, "line": edge["sites"][0]["line"]})
        flows[key] = {k: v for k, v in flow.items() if k != "steps"} | {"steps": steps}

    if problems:
        print("REFUSING TO BUILD:")
        for p in problems:
            print("  ", p)
        sys.exit(1)

    nodes = {nid: {"label": n[0], "path": n[1], "role": n[2], "about": n[3],
                   "row": n[4], "cx": n[5], "kind": n[6]} for nid, n in NODES.items()}
    data = {"nodes": nodes, "edges": edges, "flows": flows, "routes": route_table,
            "built": "15 Sep 2026"}

    template = (HERE / "callmap_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = HERE / "stepwise-call-map.html"
    out.write_text(template.replace("__DATA__", payload), encoding="utf-8")

    sites = sum(len(e["sites"]) for e in edges)
    cross = {(e["src"], e["dst"]) for e in edges if e["src"] != e["dst"]}
    print(f"{len(edges)} edges, {sites} call sites, {len(cross)} file-to-file connections, "
          f"{len(route_table)} routes, flows: " + ", ".join(f"{k}={len(v['steps'])}" for k, v in flows.items()))
    print("wrote", out)


if __name__ == "__main__":
    main()
