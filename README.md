# ML Dashboard

[English](#motivation) | [繁體中文](#動機)

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

---

# ML Dashboard（繁體中文）

即時分析平台，用於監控行為遙測資料與機器學習模型輸出。專為設計科學研究（DSR）的迭代流程而建，支援持續蒐集、視覺化與回饋評估資料。

## 動機

在設計科學研究中（Peffers et al., 2007），人工物的開發遵循設計、展示與評估的迭代循環。每一輪評估都會產生行為與問卷資料，需要分析後才能指引下一輪設計迭代。本平台解決了這個流程中的基礎設施缺口：

**問題。** DSR 評估資料——特別是事件頻率、session 時長、互動模式等行為遙測——在田野實驗中以連續串流的形式產生。傳統做法需要研究者手動從後端匯出資料、離線轉換、在 R 或 Python notebook 中分析、再彙整報告，團隊才能討論設計修改方向。這套流程在資料蒐集與設計洞察之間引入了延遲，拖慢了 DSR 方法論賴以運作的 build-measure-learn 循環。

**解決方案。** 本平台提供從資料匯入到視覺化的一站式 pipeline。應用程式後端透過 REST API 或 Python SDK 即時推送行為遙測至 dashboard。研究者可以透過 WebSocket 驅動的即時更新監控進行中的實驗 session、直接對蒐集到的資料執行內建 ML 分析（分類、分群、模型比較），並透過 web 介面與協作者共享結果——無需撰寫程式碼。透過消除評估與設計之間的落差，本平台能以實證資料而非事後報告來驅動更快速的人工物參數迭代。

**研究背景。** 本平台為 phubbing 緩解設計科學研究的一部分（Ko, Pan, Wu, Chen, Cheng & Chou, 2026），該研究中的 IT 人工物會產生行為遙測（螢幕喚醒頻率、緩衝區啟動次數、session 時長），研究團隊需要觀察不同實驗條件下的互動模式，以迭代式地調整設計摩擦參數與社會鷹架機制。

### 參考文獻

- Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. *Journal of Management Information Systems*, 24(3), 45-77.
- Ko, I.-W., Pan, J.-S., Wu, P.-N., Chen, H.-M., Cheng, P.-C., & Chou, C.-Y. (2026). Choice architecture for co-presence: A design science research on phubbing. *Proceedings of the 32nd Americas Conference on Information Systems (AMCIS)*, Reno, NV.

---

## 目錄

1. [快速開始](#快速開始)
2. [功能一覽](#功能一覽)
3. [透過 API 推送資料](#透過-api-推送資料)
4. [透過 Python SDK 推送資料](#透過-python-sdk-推送資料)
5. [透過 JavaScript SDK 推送資料](#透過-javascript-sdk-推送資料)
6. [API Key 驗證](#api-key-驗證)
7. [API 參考](#api-參考)
8. [Docker 部署](#docker-部署)
9. [執行測試](#執行測試)
10. [專案結構](#專案結構)

---

## 快速開始

### 前置需求

- Python 3.10+（建議 3.12+）
- pip

### 1. Clone

```bash
git clone https://github.com/jxuan083/ml-dashboard.git
cd ml-dashboard
```

### 2. 安裝相依套件

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 3. 啟動伺服器

```bash
cd backend
uvicorn main:app --port 8000
```

### 4. 開啟 Dashboard

前往 http://localhost:8000。可以在 UI 中直接上傳 CSV 檔案，或透過 API/SDK 以程式方式推送資料（見下方說明）。

---

## 功能一覽

| 功能 | 說明 |
|------|------|
| CSV 上傳 | 在 web UI 中拖放 CSV 檔案 |
| REST API | 從任何語言透過 HTTP 推送資料 |
| 串流模式 | 持續即時資料匯入，自動批次 flush |
| 分群分析 | KMeans、DBSCAN |
| 預測分析 | Random Forest、Logistic Regression、XGBoost、LightGBM |
| 模型比較 | 在同一資料集上執行多個模型，自動排名 |
| AutoML | 自動模型選擇 + 超參數調校（Optuna） |
| 推薦引擎 | 協同過濾 |
| 程式碼沙盒 | 在瀏覽器中直接執行 Python 程式碼 |
| 專案管理 | 多專案隔離，各專案獨立 API key |
| 即時更新 | WebSocket 推送——資料即時呈現於 dashboard |
| 實驗追蹤 | 從 Colab/Jupyter 記錄訓練指標，即時檢視曲線 |

---

## 透過 API 推送資料

不需要 SDK，使用 `curl` 或任何 HTTP client 即可。

### 建立新資料集

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

回應：

```json
{
  "id": "a1b2c3d4",
  "message": "Dataset received",
  "rows": 2,
  "cols": 3
}
```

### 追加資料列到既有資料集

```bash
curl -X POST http://localhost:8000/api/datasets/a1b2c3d4/append \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"product": "C", "price": 150, "sold": 45}
    ]
  }'
```

推送後資料立即顯示於 dashboard。

---

## 透過 Python SDK 推送資料

### 安裝

```bash
pip install -e sdk/python
```

### 基本用法

```python
from ml_platform import MLPlatform

client = MLPlatform("http://localhost:8000")

# 建立資料集並匯入初始資料
ds = client.create_dataset("user_behavior", data=[
    {"age": 25, "action": "click", "value": 100},
    {"age": 30, "action": "buy",   "value": 500},
])

# 追加更多資料列
ds.push([{"age": 22, "action": "click", "value": 80}])

# 預覽、資訊、列表
ds.preview(rows=3)
ds.info()
client.get_datasets()
```

### 串流模式

緩衝資料列並批次 flush，適合即時遙測場景。

```python
with client.stream("sensor_data", flush_every=5.0, batch_size=100) as stream:
    for reading in sensor_readings:
        stream.add({"temperature": reading.temp, "humidity": reading.hum})
# 離開時自動 flush 剩餘緩衝區
```

### 實驗追蹤

從 Colab 或 Jupyter 記錄 ML 訓練指標，dashboard 即時顯示結果。

```python
exp = client.experiment("ResNet-50", params={"lr": 0.001, "optimizer": "AdamW"})

for epoch in range(50):
    loss, acc = train_one_epoch(model, ...)
    exp.log({"epoch": epoch, "loss": loss, "val_acc": acc})

exp.log_model({"best_acc": 0.94, "params": "25.6M"})
```

---

## 透過 JavaScript SDK 推送資料

支援 Node.js 與瀏覽器。

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

### 串流

```javascript
const stream = client.stream('live_events', { flushEvery: 5000, batchSize: 100 });
stream.add({ event: 'click', ts: Date.now() });
await stream.close();
```

---

## API Key 驗證

透過各專案獨立的 API key 保護敏感專案。

```bash
# 建立受保護的專案
curl -X POST http://localhost:8000/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "secret_project", "id": "secret", "api_key": "sk-your-secret-key"}'

# 帶 key 推送資料
curl -X POST http://localhost:8000/api/datasets/push \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-your-secret-key" \
  -d '{"name": "classified", "project": "secret", "data": [{"x": 1, "y": 2}]}'
```

```python
# SDK 附帶 API key
client = MLPlatform("http://localhost:8000", api_key="sk-your-secret-key")
```

未提供有效 key 的請求會回傳 `401 Unauthorized`。

---

## API 參考

### 資料集

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/api/datasets/push` | 建立新資料集 |
| `POST` | `/api/datasets/{id}/append` | 追加資料列到既有資料集 |
| `GET` | `/api/datasets` | 列出所有資料集 |
| `GET` | `/api/datasets/{id}` | 取得資料集 metadata |
| `GET` | `/api/datasets/{id}/preview?rows=5` | 預覽資料列（預設 5 筆） |

### 分析

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/api/analyze/cluster` | 分群（KMeans / DBSCAN） |
| `POST` | `/api/analyze/predict` | 預測（RF / Logistic / XGBoost / LightGBM） |
| `POST` | `/api/analyze/predict/compare` | 多模型比較 |
| `POST` | `/api/analyze/recommend` | 推薦 |

### AutoML

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/api/automl/run` | 啟動 AutoML 任務 |
| `GET` | `/api/automl/status/{job_id}` | 查詢任務進度 |

### 程式碼執行

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/api/code/run` | 在沙盒環境中執行 Python |

### 專案

| 方法 | 路徑 | 說明 |
|------|------|------|
| `GET` | `/api/projects` | 列出所有專案 |
| `POST` | `/api/projects` | 建立新專案 |
| `DELETE` | `/api/projects/{id}` | 刪除專案及其資料集與 key |

### 其他

| 方法 | 路徑 | 說明 |
|------|------|------|
| `GET` | `/health` | 健康檢查 |
| `WS` | `/ws` | WebSocket 即時更新 |

---

## Docker 部署

```bash
docker build -t ml-dashboard .
docker run -p 8080:8080 ml-dashboard
```

開啟 http://localhost:8080。

---

## 執行測試

```bash
source .venv/bin/activate
cd backend
pytest tests/ -v
```

---

## 專案結構

```
ml-dashboard/
├── index.html              # 前端（React SPA）
├── Dockerfile              # 容器部署
├── backend/
│   ├── main.py             # API 伺服器（FastAPI）
│   ├── db.py               # SQLite 持久化層
│   ├── requirements.txt    # Python 相依套件
│   ├── ml/
│   │   ├── automl.py       # AutoML pipeline
│   │   ├── cluster.py      # 分群
│   │   ├── predict.py      # 預測模型
│   │   ├── recommend.py    # 推薦
│   │   ├── preprocessing.py    # 資料前處理
│   │   ├── feature_engineering.py  # 特徵工程
│   │   └── feature_selection.py    # 特徵選擇
│   └── tests/
│       ├── test_api.py     # API 測試
│       └── test_automl_e2e.py  # AutoML 端到端測試
├── sdk/
│   ├── python/             # Python SDK
│   │   ├── ml_platform/__init__.py
│   │   └── setup.py
│   └── js/                 # JavaScript SDK
│       └── index.js
└── .github/workflows/
    └── ci.yml              # CI/CD（測試 + 部署至 Cloud Run）
```
