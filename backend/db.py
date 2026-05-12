"""SQLite persistence layer for ML Dashboard."""

import sqlite3
import json
import os
import threading

DB_PATH = os.environ.get("ML_DASHBOARD_DB", os.path.join(os.path.dirname(__file__), "data.db"))

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            icon TEXT,
            has_api INTEGER DEFAULT 0,
            data TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS api_keys (
            key_hash TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            project TEXT DEFAULT 'default',
            rows INTEGER DEFAULT 0,
            cols INTEGER DEFAULT 0,
            columns TEXT DEFAULT '[]',
            source TEXT DEFAULT 'api',
            icon TEXT DEFAULT '📡',
            icon_bg TEXT DEFAULT '#DDD6FE',
            data TEXT DEFAULT '[]',
            created_at REAL
        );
    """)
    conn.commit()


# ── Projects ──

def load_projects() -> dict[str, dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM projects").fetchall()
    result = {}
    for r in rows:
        extra = json.loads(r["data"]) if r["data"] else {}
        result[r["id"]] = {
            "id": r["id"], "name": r["name"], "icon": r["icon"],
            "has_api": bool(r["has_api"]), **extra,
        }
    return result


def save_project(proj: dict):
    conn = _get_conn()
    extra = {k: v for k, v in proj.items() if k not in ("id", "name", "icon", "has_api")}
    conn.execute(
        "INSERT OR REPLACE INTO projects (id, name, icon, has_api, data) VALUES (?, ?, ?, ?, ?)",
        (proj["id"], proj["name"], proj.get("icon", ""), int(proj.get("has_api", False)), json.dumps(extra)),
    )
    conn.commit()


def delete_project(proj_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM api_keys WHERE project_id = ?", (proj_id,))
    conn.execute("DELETE FROM datasets WHERE project = ?", (proj_id,))
    conn.execute("DELETE FROM projects WHERE id = ?", (proj_id,))
    conn.commit()


# ── API Keys ──

def load_api_keys() -> dict[str, str]:
    conn = _get_conn()
    rows = conn.execute("SELECT key_hash, project_id FROM api_keys").fetchall()
    return {r["key_hash"]: r["project_id"] for r in rows}


def save_api_key(key_hash: str, project_id: str):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO api_keys (key_hash, project_id) VALUES (?, ?)",
        (key_hash, project_id),
    )
    conn.commit()


def delete_api_keys_for_project(proj_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM api_keys WHERE project_id = ?", (proj_id,))
    conn.commit()


# ── Datasets ──

def load_datasets() -> dict[str, dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM datasets").fetchall()
    result = {}
    for r in rows:
        columns = json.loads(r["columns"]) if r["columns"] else []
        data = json.loads(r["data"]) if r["data"] else []
        result[r["id"]] = {
            "id": r["id"], "name": r["name"], "project": r["project"],
            "rows": r["rows"], "cols": r["cols"], "columns": columns,
            "source": r["source"], "icon": r["icon"], "iconBg": r["icon_bg"],
            "data": data, "row_count": r["rows"], "col_count": r["cols"],
            "created_at": r["created_at"],
        }
    return result


def save_dataset(ds: dict):
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO datasets
           (id, name, project, rows, cols, columns, source, icon, icon_bg, data, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ds["id"], ds["name"], ds.get("project", "default"),
            ds.get("rows", 0), ds.get("cols", 0),
            json.dumps(ds.get("columns", [])),
            ds.get("source", "api"), ds.get("icon", "📡"), ds.get("iconBg", "#DDD6FE"),
            json.dumps(ds.get("data", [])),
            ds.get("created_at", 0),
        ),
    )
    conn.commit()


def delete_dataset(ds_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM datasets WHERE id = ?", (ds_id,))
    conn.commit()


def delete_datasets_for_project(proj_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM datasets WHERE project = ?", (proj_id,))
    conn.commit()
