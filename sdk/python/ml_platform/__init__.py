"""ML Platform Python SDK - lightweight client for pushing data to the dashboard."""

import time
import threading
import requests
from datetime import datetime

__version__ = "0.1.0"


class MLPlatform:
    """Client for ML Platform data ingestion.

    Usage:
        from ml_platform import MLPlatform

        client = MLPlatform("http://localhost:8000", api_key="sk-xxx")
        ds = client.create_dataset("用戶行為", project="ecom")
        ds.push([{"age": 25, "amount": 100}, {"age": 30, "amount": 200}])

        # Auto-flush mode: buffer rows and send in batches
        with client.stream("即時事件", project="ecom", flush_every=5.0) as stream:
            for event in events:
                stream.add(event)  # buffered, flushed every 5 seconds
    """

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-API-Key"] = api_key
        self.session.headers["Content-Type"] = "application/json"

    def create_dataset(self, name: str, project: str = "default", data: list[dict] | None = None) -> "Dataset":
        """Create a new dataset on the platform."""
        payload = {"name": name, "project": project, "data": data or []}
        resp = self.session.post(f"{self.base_url}/api/datasets/push", json=payload)
        resp.raise_for_status()
        info = resp.json()
        return Dataset(self, info["id"], name, project)

    def get_datasets(self, project: str | None = None) -> list[dict]:
        """List all datasets."""
        params = {"project": project} if project else {}
        resp = self.session.get(f"{self.base_url}/api/datasets", params=params)
        resp.raise_for_status()
        return resp.json()["datasets"]

    def dataset(self, ds_id: str) -> "Dataset":
        """Get a handle to an existing dataset by ID."""
        resp = self.session.get(f"{self.base_url}/api/datasets/{ds_id}")
        resp.raise_for_status()
        info = resp.json()
        return Dataset(self, ds_id, info["name"], info["project"])

    def stream(self, name: str, project: str = "default", flush_every: float = 5.0, batch_size: int = 100):
        """Create a streaming dataset with auto-flush."""
        ds = self.create_dataset(name, project)
        return StreamContext(ds, flush_every, batch_size)

    def experiment(self, name: str, project: str = "default", params: dict | None = None) -> "Experiment":
        """Create an ML experiment for logging training metrics.

        Usage (e.g. in Colab):
            exp = client.experiment("ResNet-50 lr=0.001")
            for epoch in range(10):
                train(...)
                exp.log({"epoch": epoch, "loss": 0.5, "accuracy": 0.85})
            exp.log_model({"model": "ResNet-50", "best_acc": 0.92, "params": 1.2e6})
        """
        return Experiment(self, name, project, params or {})


class Dataset:
    """Handle to a remote dataset."""

    def __init__(self, client: MLPlatform, ds_id: str, name: str, project: str):
        self.client = client
        self.id = ds_id
        self.name = name
        self.project = project

    def push(self, data: list[dict]) -> dict:
        """Push (append) rows to this dataset."""
        resp = self.client.session.post(
            f"{self.client.base_url}/api/datasets/{self.id}/append",
            json={"data": data},
        )
        resp.raise_for_status()
        return resp.json()

    def info(self) -> dict:
        """Get dataset metadata."""
        resp = self.client.session.get(f"{self.client.base_url}/api/datasets/{self.id}")
        resp.raise_for_status()
        return resp.json()

    def preview(self, rows: int = 5) -> dict:
        """Get a preview of the dataset."""
        resp = self.client.session.get(f"{self.client.base_url}/api/datasets/{self.id}/preview", params={"rows": rows})
        resp.raise_for_status()
        return resp.json()


class StreamContext:
    """Context manager for buffered streaming."""

    def __init__(self, dataset: Dataset, flush_every: float, batch_size: int):
        self.dataset = dataset
        self.flush_every = flush_every
        self.batch_size = batch_size
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._running = False

    def __enter__(self):
        self._running = True
        self._schedule_flush()
        return self

    def __exit__(self, *_):
        self._running = False
        if self._timer:
            self._timer.cancel()
        self.flush()

    def add(self, row: dict):
        """Add a row to the buffer."""
        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self.batch_size:
                self._flush_locked()

    def flush(self):
        """Manually flush the buffer."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        try:
            self.dataset.push(batch)
        except Exception as e:
            # Re-add failed rows to buffer
            self._buffer = batch + self._buffer
            raise

    def _schedule_flush(self):
        if not self._running:
            return
        self._timer = threading.Timer(self.flush_every, self._timed_flush)
        self._timer.daemon = True
        self._timer.start()

    def _timed_flush(self):
        self.flush()
        self._schedule_flush()


class Experiment:
    """Track ML experiment metrics and push them to the dashboard.

    Designed for use in Colab / Jupyter notebooks. Each log() call
    appends a row to a "metrics" dataset; the dashboard shows it
    as a live-updating table that can be visualized with built-in charts.
    """

    def __init__(self, client: MLPlatform, name: str, project: str, params: dict):
        self.client = client
        self.name = name
        self.project = project
        self.params = params
        self._step = 0
        ts = datetime.now().strftime("%m/%d %H:%M")
        tag = f"{name} ({ts})"
        self._metrics_ds = client.create_dataset(
            f"[metrics] {tag}", project=project,
            data=[{"_experiment": name, "_type": "config", **params}] if params else [],
        )
        self._models_ds = None

    @property
    def id(self) -> str:
        return self._metrics_ds.id

    def log(self, metrics: dict, step: int | None = None):
        """Log one row of metrics (e.g. one epoch).

        Args:
            metrics: dict of metric values, e.g. {"loss": 0.5, "val_acc": 0.85}
            step: optional step number; auto-increments if omitted
        """
        if step is None:
            step = self._step
        self._step = step + 1
        row = {"step": step, **metrics}
        self._metrics_ds.push([row])

    def log_model(self, info: dict):
        """Log a trained model's summary (final metrics, hyperparams, etc.)."""
        if self._models_ds is None:
            self._models_ds = self.client.create_dataset(
                f"[models] {self.name}", project=self.project,
            )
        self._models_ds.push([{
            "_experiment": self.name,
            "_logged_at": datetime.now().isoformat(),
            **info,
        }])

    def summary(self) -> dict:
        """Get the current metrics dataset info."""
        return self._metrics_ds.info()

    def preview(self, rows: int = 10) -> dict:
        """Preview the latest logged metrics."""
        return self._metrics_ds.preview(rows=rows)
