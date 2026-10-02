FROM python:3.11-slim

WORKDIR /app

# Install build deps then runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the end-to-end demo and emit benchmark.json
CMD ["python", "-m", "modalforge.examples.run_demo"]
