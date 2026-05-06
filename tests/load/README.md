# Load Testing with k6

## Prerequisites

Install k6: https://k6.io/docs/getting-started/installation/

```bash
# macOS
brew install k6

# Windows (Chocolatey)
choco install k6

# Docker
docker pull grafana/k6
```

## Running the load test

### 1. Start the app
```bash
docker-compose up -d
# or
uvicorn src.main:app --host 0.0.0.0 --port 8200
```

### 2. Get a test token
```bash
curl -X POST http://localhost:8200/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password"}'
```

### 3. Run the load test
```bash
# Local
k6 run tests/load/chat_stream.js \
  -e BASE_URL=http://localhost:8200 \
  -e TEST_TOKEN=<your-token>

# Docker
docker run --rm -i grafana/k6 run - < tests/load/chat_stream.js \
  -e BASE_URL=http://host.docker.internal:8200 \
  -e TEST_TOKEN=<your-token>
```

## Target thresholds

| Metric | Target |
|--------|--------|
| Concurrent users | 50 |
| p95 response time | < 5000ms |
| Failure rate | < 10% |
| Success rate | > 90% |

## Interpreting results

- `http_req_duration p(95)` — 95th percentile response time, must be under 5s
- `http_req_failed` — fraction of failed requests, must be under 10%
- `stream_duration_ms` — custom metric tracking full SSE stream duration

Results are saved to `tests/load/results/summary.json` after each run.
