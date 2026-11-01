"""Tiny SQLite layer: users, models, predictions, logs, reports."""
import json, sqlite3
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
DB_PATH = DATA / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT UNIQUE, pw_hash TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS models(
  id INTEGER PRIMARY KEY AUTOINCREMENT, owner INTEGER, name TEXT, filename TEXT, path TEXT,
  size_mb REAL, framework_declared TEXT, framework_detected TEXT, task_type TEXT, kind TEXT,
  n_features INTEGER, feature_names TEXT, classes TEXT, class_labels TEXT, parent_id INTEGER,
  trained_on_demo INTEGER DEFAULT 0, is_demo INTEGER DEFAULT 0, dataset_path TEXT, dataset_target TEXT,
  fix_note TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, model_id INTEGER, ts TEXT, inputs TEXT, output TEXT,
  latency_ms REAL, ok INTEGER, source TEXT);
CREATE TABLE IF NOT EXISTS logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, owner INTEGER, ts TEXT, message TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS reports(
  id INTEGER PRIMARY KEY AUTOINCREMENT, owner INTEGER, filename TEXT, path TEXT, ts TEXT,
  size_bytes INTEGER, rtype TEXT, model_name TEXT);
"""

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    return c

def init():
    with conn() as c:
        c.executescript(SCHEMA)

def q(sql, args=(), one=False):
    with conn() as c:
        cur = c.execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
    return (rows[0] if rows else None) if one else rows

def x(sql, args=()):
    with conn() as c:
        cur = c.execute(sql, args)
        return cur.lastrowid

def log(owner, message, status="success"):
    x("INSERT INTO logs(owner,ts,message,status) VALUES(?,?,?,?)", (owner, now(), message, status))

def jl(v):
    return json.dumps(v)

def jd(v, default=None):
    try:
        return json.loads(v) if v else default
    except Exception:
        return default
