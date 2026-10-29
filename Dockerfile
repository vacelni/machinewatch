# Downloads the NASA data and trains the model during the build, so the image never ships third-party data
# in this repository and always serves a model it trained and evaluated itself.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY machinewatch ./machinewatch
RUN python -m machinewatch.train

RUN useradd --create-home watch && mkdir -p /app/data && chown -R watch /app
USER watch
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "machinewatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
