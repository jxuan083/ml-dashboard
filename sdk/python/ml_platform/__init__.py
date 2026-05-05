"""ML Platform Python SDK - lightweight client for pushing data to the dashboard."""

import time
import threading
import requests

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
