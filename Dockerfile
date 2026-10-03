# Stepwise dashboard, containerised. Serves the FastAPI app (API + the static
# page) on port 8000. The container does NOT bundle torch or a Groq key:
#   - the dashboard reads result files; it makes no model calls itself
#   - mount results/ and pass GROQ_API_KEY only if you want the live chat/agent
#
#   docker build -t stepwise .
#   docker run -p 8000:8000 -v "$PWD/results:/app/results" stepwise
#   # with the live chat agent:
#   docker run -p 8000:8000 -v "$PWD/results:/app/results" -e GROQ_API_KEY stepwise
FROM python:3.12-slim

WORKDIR /app

# Core + validation deps only. torch/transformers (the mechanistic track) and
# mcp are left out to keep the image small; they are optional CLI features, not
# part of serving the dashboard.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt statsmodels

COPY stepwise/ ./stepwise/
COPY main.py ./

# A read-only demo run so the container shows something on first boot even
# with no mounted results. Regenerated at build time, marked simulated.
RUN python main.py demo --n_samples 2 >/dev/null 2>&1 || true

EXPOSE 8000
ENV STEPWISE_HOST=0.0.0.0
CMD ["python", "main.py", "dashboard", "--host", "0.0.0.0", "--port", "8000"]
