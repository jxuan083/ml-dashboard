import uuid, time, io, sys, traceback, contextlib, asyncio, json, secrets, hashlib
from typing import Any
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np

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
projects: dict[str, dict] = {
    "ecom": {"id": "ecom", "name": "電商分析", "icon": "E"},
    "bank": {"id": "bank", "name": "金融風控", "icon": "F"},
    "health": {"id": "health", "name": "醫療數據", "icon": "H"},
}
# api_key hash → project_id
api_keys: dict[str, str] = {}


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(request: Request) -> str | None:
    """Return project_id if valid API key provided, else None."""
    key = request.headers.get("x-api-key")
    if not key:
        return None
    h = _hash_key(key)
    return api_keys.get(h)

# ── WebSocket connections ──
ws_clients: list[WebSocket] = []


async def broadcast(event: dict):
    msg = json.dumps(event)
    disconnected = []
    for ws in ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        ws_clients.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_clients.remove(ws)


# ── Project endpoints ──

class CreateProject(BaseModel):
    id: str | None = None
    name: str
    icon: str | None = None
    api_key: str | None = None


@app.get("/api/projects")
def list_projects():
    result = []
    for p in projects.values():
        count = sum(1 for d in datasets.values() if d.get("project") == p["id"])
        result.append({**p, "datasetCount": count})
    return {"projects": result}


@app.post("/api/projects")
async def create_project(body: CreateProject):
    proj_id = body.id or body.name.lower().replace(" ", "_")
    if proj_id in projects:
        raise HTTPException(409, "Project already exists")
    icon = body.icon or body.name[0].upper()
    proj = {"id": proj_id, "name": body.name, "icon": icon, "has_api": bool(body.api_key)}
    projects[proj_id] = proj
    if body.api_key:
        api_keys[_hash_key(body.api_key)] = proj_id
    await broadcast({"type": "project_new", "project": {**proj, "datasetCount": 0}})
    return proj


@app.delete("/api/projects/{proj_id}")
async def delete_project(proj_id: str):
    if proj_id not in projects:
        raise HTTPException(404, "Project not found")
    # Remove associated API keys
    to_remove = [h for h, pid in api_keys.items() if pid == proj_id]
    for h in to_remove:
        del api_keys[h]
    # Remove associated datasets
    ds_to_remove = [did for did, d in datasets.items() if d.get("project") == proj_id]
    for did in ds_to_remove:
        del datasets[did]
    del projects[proj_id]
    await broadcast({"type": "project_deleted", "id": proj_id})
    return {"ok": True}


# ── Schemas ──
class PushDataset(BaseModel):
    project: str = "default"
    name: str
    data: list[dict[str, Any]]

class AnalyzeRequest(BaseModel):
    dataset_id: str
    model_id: str
    params: dict[str, Any] = {}

class CompareRequest(BaseModel):
    dataset_id: str
    test_size: float = 0.2
    feature_cols: list[str] = []
    models: dict[str, dict[str, Any]] = {}  # {model_id: params}

class CodeRunRequest(BaseModel):
    code: str
    dataset_id: str | None = None


# ── Dataset endpoints ──

@app.get("/api/datasets")
def list_datasets(project: str | None = None):
    items = list(datasets.values())
    if project:
        items = [d for d in items if d["project"] == project]
    return {"datasets": items}


class AppendData(BaseModel):
    data: list[dict[str, Any]]


@app.post("/api/datasets/push")
async def push_dataset(body: PushDataset, request: Request):
    # If project has API key enabled, verify it
    proj = projects.get(body.project)
    if proj and proj.get("has_api"):
        authed_proj = verify_api_key(request)
        if authed_proj != body.project:
            raise HTTPException(401, "Invalid or missing API key for this project")
    # Auto-create project if it doesn't exist
    if body.project and body.project not in projects:
        proj = {"id": body.project, "name": body.project, "icon": body.project[0].upper()}
        projects[body.project] = proj
        await broadcast({"type": "project_new", "project": {**proj, "datasetCount": 0}})
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
        "row_count": len(body.data),
        "col_count": len(df.columns),
        "created_at": time.time(),
    }
    datasets[ds_id] = ds
    await broadcast({"type": "dataset_new", "dataset": {k: v for k, v in ds.items() if k != "data"}})
    return {"id": ds_id, "message": "Dataset received", "rows": ds["rows"], "cols": ds["cols"]}


@app.post("/api/datasets/{ds_id}/append")
async def append_dataset(ds_id: str, body: AppendData):
    if ds_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    ds = datasets[ds_id]
    ds["data"].extend(body.data)
    ds["rows"] = len(ds["data"])
    ds["row_count"] = ds["rows"]
    # Update columns if new fields appear
    if body.data:
        new_cols = set(body.data[0].keys()) - set(ds["columns"])
        if new_cols:
            ds["columns"].extend(sorted(new_cols))
            ds["cols"] = len(ds["columns"])
            ds["col_count"] = ds["cols"]
    await broadcast({"type": "dataset_updated", "id": ds_id, "rows": ds["rows"], "cols": ds["cols"]})
    return {"id": ds_id, "rows": ds["rows"], "cols": ds["cols"], "appended": len(body.data)}


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


@app.post("/api/analyze/predict/compare")
def compare_predict(body: CompareRequest):
    if body.dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    data = datasets[body.dataset_id]["data"]
    if not data:
        raise HTTPException(400, "Dataset is empty")

    feature_cols = body.feature_cols or None
    model_configs = body.models or {mid: {} for mid in ["xgboost", "random_forest", "logistic", "lightgbm"]}
    results = []

    for model_id, params in model_configs.items():
        t0 = time.time()
        try:
            merged = {**params, "test_size": body.test_size}
            if feature_cols:
                merged["feature_cols"] = feature_cols
            result = run_predict(data, model_id, merged)
            result["model_id"] = model_id
            result["latency_ms"] = round((time.time() - t0) * 1000)
            results.append(result)
        except Exception as e:
            results.append({"model_id": model_id, "model": model_id, "error": str(e),
                            "f1": 0, "accuracy": 0, "precision": 0, "recall": 0, "roc_auc": None})

    results.sort(key=lambda r: r.get("f1", 0), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return {"results": results}


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


# ── Code execution endpoint ──

_BLOCKED_NAMES = frozenset([
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
    "os", "subprocess", "shutil", "pathlib", "importlib",
])


def _make_sandbox(df: pd.DataFrame | None) -> dict:
    """Build a restricted namespace for user code execution."""
    import sklearn.cluster as sk_cluster
    import sklearn.preprocessing as sk_prep
    import sklearn.model_selection as sk_split
    import sklearn.metrics as sk_metrics
    import sklearn.ensemble as sk_ensemble
    import sklearn.linear_model as sk_linear
    from scipy import stats as sp_stats

    ns = {
        "__builtins__": {
            k: v for k, v in __builtins__.__dict__.items()
            if k not in _BLOCKED_NAMES and not k.startswith("_")
        } if isinstance(__builtins__, type(sys)) else {
            k: v for k, v in __builtins__.items()
            if k not in _BLOCKED_NAMES and not k.startswith("_")
        },
        "pd": pd,
        "np": np,
        "sklearn": __import__("sklearn"),
        "cluster": sk_cluster,
        "preprocessing": sk_prep,
        "model_selection": sk_split,
        "metrics": sk_metrics,
        "ensemble": sk_ensemble,
        "linear_model": sk_linear,
        "stats": sp_stats,
    }
    if df is not None:
        ns["df"] = df.copy()
    return ns


@app.post("/api/code/run")
def run_code(body: CodeRunRequest):
    if len(body.code) > 10_000:
        raise HTTPException(400, "Code too long (max 10 000 chars)")

    # Prepare dataframe if dataset specified
    df = None
    if body.dataset_id:
        if body.dataset_id not in datasets:
            raise HTTPException(404, "Dataset not found")
        data = datasets[body.dataset_id]["data"]
        if data:
            df = pd.DataFrame(data)

    sandbox = _make_sandbox(df)
    stdout_buf = io.StringIO()
    result_value = None

    t0 = time.time()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(body.code, sandbox)
        # Check if user stored anything in `result`
        result_value = sandbox.get("result", None)
    except Exception:
        tb = traceback.format_exc()
        return {
            "ok": False,
            "stdout": stdout_buf.getvalue(),
            "error": tb,
            "latency_ms": round((time.time() - t0) * 1000),
        }

    # Serialize result_value
    serialized = None
    if result_value is not None:
        if isinstance(result_value, pd.DataFrame):
            serialized = {
                "type": "dataframe",
                "columns": result_value.columns.tolist(),
                "rows": result_value.head(200).fillna("").to_dict(orient="records"),
                "shape": list(result_value.shape),
            }
        elif isinstance(result_value, (dict, list, int, float, str, bool)):
            serialized = {"type": "json", "data": result_value}
        elif isinstance(result_value, np.ndarray):
            serialized = {"type": "json", "data": result_value.tolist()}
        else:
            serialized = {"type": "text", "data": str(result_value)}

    return {
        "ok": True,
        "stdout": stdout_buf.getvalue(),
        "result": serialized,
        "latency_ms": round((time.time() - t0) * 1000),
    }


@app.get("/health")
def health():
    return {"status": "ok", "datasets": len(datasets), "jobs": len(jobs)}


@app.get("/")
def serve_frontend():
    return FileResponse("/app/index.html", media_type="text/html")
