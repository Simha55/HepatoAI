from __future__ import annotations
import sqlite3
import json
import os
import threading
from typing import Any, Dict, List, Optional
from .config import settings

_db_lock = threading.Lock()

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS assets (
  asset_id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  status TEXT NOT NULL,
  idempotency_key TEXT,
  expected_images INTEGER NOT NULL,
  processed_images INTEGER NOT NULL DEFAULT 0,
  failed_images INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_assets_idem ON assets(idempotency_key);

CREATE TABLE IF NOT EXISTS images (
  image_id TEXT PRIMARY KEY,
  asset_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  status TEXT NOT NULL,
  backend TEXT,
  latency_ms REAL,
  created_at REAL NOT NULL,
  FOREIGN KEY(asset_id) REFERENCES assets(asset_id)
);

CREATE TABLE IF NOT EXISTS detections (
  image_id TEXT PRIMARY KEY,
  detections_json TEXT NOT NULL,
  FOREIGN KEY(image_id) REFERENCES images(image_id)
);
"""

def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.sqlite_path), exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with _db_lock:
        conn = connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

def get_asset_by_idempotency(idem: str) -> Optional[Dict[str, Any]]:
    if not idem:
        return None
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM assets WHERE idempotency_key=? LIMIT 1", (idem,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def insert_asset(asset_id: str, created_at: float, expected_images: int, idempotency_key: Optional[str]) -> None:
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO assets(asset_id,created_at,status,idempotency_key,expected_images,processed_images,failed_images) VALUES(?,?,?,?,?,?,?)",
            (asset_id, created_at, "pending", idempotency_key, expected_images, 0, 0),
        )
        conn.commit()
    finally:
        conn.close()

def update_asset_progress(asset_id: str, processed_delta: int, failed_delta: int) -> None:
    conn = connect()
    try:
        conn.execute(
            "UPDATE assets SET processed_images=processed_images+?, failed_images=failed_images+? WHERE asset_id=?",
            (processed_delta, failed_delta, asset_id),
        )
        row = conn.execute(
            "SELECT expected_images, processed_images, failed_images FROM assets WHERE asset_id=?",
            (asset_id,),
        ).fetchone()
        if row:
            expected = int(row["expected_images"])
            processed = int(row["processed_images"])
            failed = int(row["failed_images"])
            if processed + failed >= expected:
                status = "done" if failed == 0 else "done_with_failures"
                conn.execute("UPDATE assets SET status=? WHERE asset_id=?", (status, asset_id))
        conn.commit()
    finally:
        conn.close()

def get_asset(asset_id: str) -> Optional[Dict[str, Any]]:
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def insert_image(image_id: str, asset_id: str, filename: str, path: str, sha256: str, width: int, height: int, created_at: float) -> None:
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO images(image_id,asset_id,filename,path,sha256,width,height,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (image_id, asset_id, filename, path, sha256, width, height, "queued", created_at),
        )
        conn.commit()
    finally:
        conn.close()

def set_image_result(image_id: str, status: str, backend: str, latency_ms: float, detections: List[Dict[str, Any]]) -> None:
    conn = connect()
    try:
        conn.execute(
            "UPDATE images SET status=?, backend=?, latency_ms=? WHERE image_id=?",
            (status, backend, float(latency_ms), image_id),
        )
        conn.execute(
            "INSERT OR REPLACE INTO detections(image_id, detections_json) VALUES(?,?)",
            (image_id, json.dumps(detections)),
        )
        conn.commit()
    finally:
        conn.close()

def list_images_for_asset(asset_id: str) -> List[Dict[str, Any]]:
    conn = connect()
    try:
        rows = conn.execute("SELECT * FROM images WHERE asset_id=? ORDER BY created_at ASC", (asset_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            det_row = conn.execute("SELECT detections_json FROM detections WHERE image_id=?", (d["image_id"],)).fetchone()
            d["detections"] = json.loads(det_row["detections_json"]) if det_row else None
            out.append(d)
        return out
    finally:
        conn.close()
