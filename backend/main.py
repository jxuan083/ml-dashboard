import uuid, time, io
from typing import Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from ml.cluster import run_cluster
from ml.predict import run_predict
from ml.recommend import run_recommend

app = FastAPI(title="ML Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory store ──
datasets: dict[str, dict] = {}
jobs: dict[str, dict] = {}


# ── Schemas ──
class PushDataset(BaseModel):
    project: str = "default"
    name: str
    data: list[dict[str, Any]]

class AnalyzeRequest(BaseModel):
    dataset_id: str
    model_id: str
    params: dict[str, Any] = {}


# ── Dataset endpoints ──

@app.get("/api/datasets")
def list_datasets(project: str | None = None):
    items = list(datasets.values())
    if project:
        items = [d for d in items if d["project"] == project]
    return {"datasets": items}


@app.post("/api/datasets/push")
def push_dataset(body: PushDataset):
    ds_id = str(uuid.uuid4())[:8]
    df = pd.DataFrame(body.data)
    ds = {
        "id": ds_id,
        "name": body.name,
        "project": body.project,
        "rows": len(body.data),
        "cols": len(df.columns),
        "columns": df.columns.tolist(),
        "source": "api",
        "icon": "📡",
        "iconBg": "#DDD6FE",
        "data": body.data,
        "created_at": time.time(),
    }
    datasets[ds_id] = ds
    return {"id": ds_id, "message": "Dataset received", "rows": ds["rows"], "cols": ds["cols"]}


@app.post("/api/datasets/upload")
async def upload_csv(file: UploadFile = File(...), project: str = "default"):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    ds_id = str(uuid.uuid4())[:8]
    data = df.fillna("").to_dict(orient="records")
    ds = {
        "id": ds_id,
        "name": file.filename.replace(".csv", ""),
        "project": project,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": df.columns.tolist(),
        "source": "csv",
        "icon": "📄",
        "iconBg": "#DBEAFE",
        "data": data,
        "created_at": time.time(),
    }
    datasets[ds_id] = ds
    return {"id": ds_id, "name": ds["name"], "rows": ds["rows"], "cols": ds["cols"], "columns": ds["columns"]}


@app.get("/api/datasets/{ds_id}")
def get_dataset(ds_id: str):
    if ds_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    ds = dict(datasets[ds_id])
    ds.pop("data", None)  # don't return full data by default
    return ds


@app.get("/api/datasets/{ds_id}/preview")
def preview_dataset(ds_id: str, rows: int = 5):
    if ds_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    ds = datasets[ds_id]
    return {
        "columns": ds["columns"],
        "rows": ds["data"][:rows],
        "total_rows": ds["rows"],
    }


# ── Analyze endpoints ──

def _run_analysis(ds_id: str, analysis_type: str, model_id: str, params: dict) -> dict:
    if ds_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    data = datasets[ds_id]["data"]
    if not data:
        raise HTTPException(400, "Dataset is empty")

    t0 = time.time()
    try:
        if analysis_type == "cluster":
            result = run_cluster(data, model_id, params)
        elif analysis_type == "predict":
            result = run_predict(data, model_id, params)
        elif analysis_type == "recommend":
            result = run_recommend(data, model_id, params)
        else:
            raise HTTPException(400, f"Unknown analysis type: {analysis_type}")
    except Exception as e:
        raise HTTPException(422, str(e))

    result["latency_ms"] = round((time.time() - t0) * 1000)
    return result


@app.post("/api/analyze/{analysis_type}")
def analyze(analysis_type: str, body: AnalyzeRequest):
    result = _run_analysis(body.dataset_id, analysis_type, body.model_id, body.params)
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "dataset_id": body.dataset_id,
        "type": analysis_type,
        "model_id": body.model_id,
        "result": result,
        "created_at": time.time(),
    }
    return {"job_id": job_id, "result": result}


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/health")
def health():
    return {"status": "ok", "datasets": len(datasets), "jobs": len(jobs)}
