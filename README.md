# ML Dashboard

A real-time analytics platform for monitoring behavioral telemetry and machine learning model outputs. Built to support iterative design science research (DSR) workflows where evaluation data must be continuously collected, visualized, and fed back into the design cycle.

## Motivation

In design science research (Peffers et al., 2007), artifact development follows iterative cycles of design, demonstration, and evaluation. Each evaluation round produces behavioral and survey data that must be analyzed to inform the next design iteration. This platform addresses a concrete infrastructure gap in that process:

**Problem.** DSR evaluation data — particularly behavioral telemetry such as event frequencies, session durations, and interaction patterns — arrives as a continuous stream during field experiments. Conventional approaches require researchers to manually extract data from the application backend, transform it offline, run analyses in separate tools (R, Python notebooks), and compile reports before the team can discuss design revisions. This workflow introduces latency between data collection and design insight, slowing the build-measure-learn cycle that DSR methodology depends on.

**Solution.** This platform provides a unified pipeline from data ingestion to visualization. The application backend pushes behavioral telemetry to the dashboard via REST API or Python SDK in real time. Researchers can monitor ongoing experiment sessions through WebSocket-driven live updates, run built-in ML analyses (classification, clustering, model comparison) directly on collected data, and share results with collaborators through a web interface — without requiring them to execute code. By collapsing the gap between evaluation and design, the platform enables faster iteration on artifact parameters informed by empirical evidence rather than post-hoc reporting.

**Research context.** This platform was developed as part of a design science study on phubbing mitigation (Ko, Pan, Wu, Chen, Cheng & Chou, 2026), where the IT artifact generates behavioral telemetry (screen wake frequency, buffer-zone activations, session duration) and the research team needs to observe interaction patterns across experimental conditions to iteratively refine design friction parameters and social scaffolding mechanisms.

### References

- Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. *Journal of Management Information Systems*, 24(3), 45-77.
- Ko, I.-W., Pan, J.-S., Wu, P.-N., Chen, H.-M., Cheng, P.-C., & Chou, C.-Y. (2026). Choice architecture for co-presence: A design science research on phubbing. *Proceedings of the 32nd Americas Conference on Information Systems (AMCIS)*, Reno, NV.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Features](#features)
3. [Pushing Data via API](#pushing-data-via-api)
4. [Pushing Data via Python SDK](#pushing-data-via-python-sdk)
5. [Pushing Data via JavaScript SDK](#pushing-data-via-javascript-sdk)
6. [API Key Authentication](#api-key-authentication)
7. [API Reference](#api-reference)
8. [Docker Deployment](#docker-deployment)
9. [Running Tests](#running-tests)
10. [Project Structure](#project-structure)

---

## Quick Start

### Prerequisites

- Python 3.10+ (3.12+ recommended)
- pip

### 1. Clone

```bash
git clone https://github.com/jxuan083/ml-dashboard.git
cd ml-dashboard
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 3. Start the server

```bash
cd backend
uvicorn main:app --port 8000
```

### 4. Open the dashboard

Go to http://localhost:8000. You can upload CSV files directly in the UI or push data programmatically via the API/SDK (see below).

---

## Features

| Feature | Description |
|---------|-------------|
| CSV upload | Drag-and-drop CSV files in the web UI |
| REST API | Push data from any language via HTTP |
| Streaming | Continuous real-time data ingestion with auto-flush |
| Clustering | KMeans, DBSCAN |
| Prediction | Random Forest, Logistic Regression, XGBoost, LightGBM |
| Model comparison | Run multiple models on the same dataset, auto-ranked |
| AutoML | Automated model selection + hyperparameter tuning (Optuna) |
| Recommendation | Collaborative filtering |
| Code sandbox | Run Python code directly in the browser |
| Project management | Multi-project isolation with per-project API keys |
| Real-time updates | WebSocket push — data appears in the dashboard instantly |
| Experiment tracking | Log training metrics from Colab/Jupyter, view live curves |

---

## Pushing Data via API

No SDK required. Use `curl` or any HTTP client.

### Create a new dataset

```bash
curl -X POST http://localhost:8000/api/datasets/push \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sales_records",
    "project": "default",
    "data": [
      {"product": "A", "price": 100, "sold": 50},
      {"product": "B", "price": 200, "sold": 30}
    ]
  }'
```

Response:

```json
{
  "id": "a1b2c3d4",
  "message": "Dataset received",
  "rows": 2,
  "cols": 3
}
```

### Append rows to an existing dataset

```bash
curl -X POST http://localhost:8000/api/datasets/a1b2c3d4/append \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"product": "C", "price": 150, "sold": 45}
    ]
  }'
```

Data appears in the dashboard immediately after pushing.

---

## Pushing Data via Python SDK

### Install

```bash
pip install -e sdk/python
```

### Basic usage

```python
from ml_platform import MLPlatform

client = MLPlatform("http://localhost:8000")

# Create a dataset with initial data
ds = client.create_dataset("user_behavior", data=[
    {"age": 25, "action": "click", "value": 100},
    {"age": 30, "action": "buy",   "value": 500},
])

# Append more rows
ds.push([{"age": 22, "action": "click", "value": 80}])

# Preview, info, list
ds.preview(rows=3)
ds.info()
client.get_datasets()
```

### Streaming mode

Buffers rows and flushes in batches. Suitable for real-time telemetry.

```python
with client.stream("sensor_data", flush_every=5.0, batch_size=100) as stream:
    for reading in sensor_readings:
        stream.add({"temperature": reading.temp, "humidity": reading.hum})
# Remaining buffer is flushed on exit
```

### Experiment tracking

Log ML training metrics from Colab or Jupyter. The dashboard displays results in real time.

```python
exp = client.experiment("ResNet-50", params={"lr": 0.001, "optimizer": "AdamW"})

for epoch in range(50):
    loss, acc = train_one_epoch(model, ...)
    exp.log({"epoch": epoch, "loss": loss, "val_acc": acc})

exp.log_model({"best_acc": 0.94, "params": "25.6M"})
```

---

## Pushing Data via JavaScript SDK

Works in both Node.js and browsers.

```javascript
const { MLPlatform } = require('./sdk/js/index.js');
const client = new MLPlatform('http://localhost:8000');

const ds = await client.createDataset('click_log', {
  data: [
    { page: '/home', clicks: 42 },
    { page: '/about', clicks: 15 },
  ]
});

await ds.push([{ page: '/contact', clicks: 8 }]);
console.log(await ds.preview(3));
```

### Streaming

```javascript
const stream = client.stream('live_events', { flushEvery: 5000, batchSize: 100 });
stream.add({ event: 'click', ts: Date.now() });
await stream.close();
```

---

## API Key Authentication

Protect sensitive projects with per-project API keys.

```bash
# Create a protected project
curl -X POST http://localhost:8000/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "secret_project", "id": "secret", "api_key": "sk-your-secret-key"}'

# Push data with the key
curl -X POST http://localhost:8000/api/datasets/push \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-your-secret-key" \
  -d '{"name": "classified", "project": "secret", "data": [{"x": 1, "y": 2}]}'
```

```python
# SDK with API key
client = MLPlatform("http://localhost:8000", api_key="sk-your-secret-key")
```

Requests without a valid key return `401 Unauthorized`.

---

## API Reference

### Datasets

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/datasets/push` | Create a new dataset |
| `POST` | `/api/datasets/{id}/append` | Append rows to an existing dataset |
| `GET` | `/api/datasets` | List all datasets |
| `GET` | `/api/datasets/{id}` | Get dataset metadata |
| `GET` | `/api/datasets/{id}/preview?rows=5` | Preview rows (default 5) |

### Analysis

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analyze/cluster` | Clustering (KMeans / DBSCAN) |
| `POST` | `/api/analyze/predict` | Prediction (RF / Logistic / XGBoost / LightGBM) |
| `POST` | `/api/analyze/predict/compare` | Multi-model comparison |
| `POST` | `/api/analyze/recommend` | Recommendation |

### AutoML

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/automl/run` | Start an AutoML job |
| `GET` | `/api/automl/status/{job_id}` | Check job progress |

### Code Execution

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/code/run` | Execute Python in a sandboxed environment |

### Projects

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/projects` | List all projects |
| `POST` | `/api/projects` | Create a new project |
| `DELETE` | `/api/projects/{id}` | Delete a project and its datasets/keys |

### Other

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `WS` | `/ws` | WebSocket for real-time updates |

---

## Docker Deployment

```bash
docker build -t ml-dashboard .
docker run -p 8080:8080 ml-dashboard
```

Open http://localhost:8080.

---

## Running Tests

```bash
source .venv/bin/activate
cd backend
pytest tests/ -v
```

---

## Project Structure

```
ml-dashboard/
├── index.html              # Frontend (React SPA)
├── Dockerfile              # Container deployment
├── backend/
│   ├── main.py             # API server (FastAPI)
│   ├── db.py               # SQLite persistence layer
│   ├── requirements.txt    # Python dependencies
│   ├── ml/
│   │   ├── automl.py       # AutoML pipeline
│   │   ├── cluster.py      # Clustering
│   │   ├── predict.py      # Prediction models
│   │   ├── recommend.py    # Recommendation
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   └── feature_selection.py
│   └── tests/
│       ├── test_api.py     # API tests
│       └── test_automl_e2e.py
├── sdk/
│   ├── python/             # Python SDK
│   │   ├── ml_platform/__init__.py
│   │   └── setup.py
│   └── js/                 # JavaScript SDK
│       └── index.js
└── .github/workflows/
    └── ci.yml              # CI/CD (test + deploy to Cloud Run)
```
