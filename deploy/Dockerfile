# Litmus dashboard image: serves `litmus dashboard` and can also run
# `litmus run` / `litmus synth` (used by the nightly cron on the server).
FROM python:3.12-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir "/app[dashboard,tracing,agent]"

EXPOSE 8000

# Overridden per use: the server compose runs the dashboard; the cron
# overrides the command with `litmus run ...`.
CMD ["litmus", "dashboard", "--reports-dir", "/reports", "--host", "0.0.0.0", "--port", "8000"]
