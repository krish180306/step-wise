# Deploying Stepwise

Two audiences, two deploys. The dashboard is read-only over result files and is
safe to make public. The **live chat/agent must never run on your Groq key**
when strangers can reach it — a public endpoint with your key is an open tab on
your rate limit (and your bill, if you leave the free tier).

## 1. Local / internal (simplest)

```bash
python main.py dashboard          # http://127.0.0.1:8000
```

Or in Docker, which bundles a tiny simulated demo run so the page is populated
on first boot:

```bash
docker build -t stepwise .
docker run -p 8000:8000 -v "$PWD/results:/app/results" stepwise
# with the live chat agent, on YOUR key, for trusted users only:
docker run -p 8000:8000 -v "$PWD/results:/app/results" -e GROQ_API_KEY stepwise
```

## 2. Public demo on Hugging Face Spaces

A Space gives you a clickable URL. Use the **Docker** SDK Space and this
`README.md` header in the Space repo (not this file):

```yaml
---
title: Stepwise
emoji: 🪜
sdk: docker
app_port: 8000
---
```

Steps:

1. Create a new **Docker** Space.
2. Push this repo to it (the `Dockerfile` here works as-is).
3. Commit a results file so the page has real data — or let the image's baked-in
   simulated demo show the shape.

### Make the live chat safe for a public Space

The chat and agent endpoints call Groq. Before exposing them publicly, pick one:

- **Disable them** (recommended for a pure demo). The dashboard, findings,
  domain comparison, explorer and saved reports all work with **no key** — they
  read result files. Only the Ask panel and `analyse` need Groq. Ship the Space
  with no `GROQ_API_KEY` secret; the Ask panel already degrades to an
  explanatory "unavailable" state (`/api/chat/status` reports it).
- **Bring-your-own-key (built).** Ship the Space with no `GROQ_API_KEY`. The
  Ask panel detects there is no server key (`/api/chat/status` →
  `accepts_byo_key`) and shows a key box; the visitor's key is sent only with
  their request, used for that request's client, and never stored or logged.
  Everything else on the page needs no key.

Do **not** set your own `GROQ_API_KEY` as a public Space secret and leave chat
on — that spends your quota on every visitor.

## 3. Rotate the key

If a key has ever been pasted into a chat, a commit, or a screenshot, rotate it
at <https://console.groq.com/keys> before deploying. `.env` is gitignored; keep
it that way.
