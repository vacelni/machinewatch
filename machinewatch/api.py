"""MachineWatch API: ingests one sensor reading per engine cycle, predicts remaining useful life from the
engine's recent history, stores both, and streams every new prediction to the dashboard.

    uvicorn machinewatch.api:app

Environment:
    MACHINEWATCH_DB           SQLite file (default data/machinewatch.db)
    MACHINEWATCH_INGEST_KEY   required: writes need header X-API-Key with this value
    MACHINEWATCH_INSECURE_DEV "1" accepts writes without a key (local development only)
    MACHINEWATCH_ALLOW_RESET  "1" lets the simulator wipe the fleet to start a new replay
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import sqlite3
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data import SENSORS as ALL_SENSORS
from .data import SETTINGS
from .features import WINDOW, engine_features
from .models import health_status
from .registry import Registry

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
CHART_SENSORS = ("s11", "s4")  # HPC static pressure and LPT outlet temperature: the clearest wear signals


class Reading(BaseModel):
    # NaN or Infinity would be stored and then poison every later feature window of that engine.
    model_config = ConfigDict(allow_inf_nan=False)

    unit: int = Field(ge=1, le=100_000)
    cycle: int = Field(ge=1, le=100_000)
    settings: list[float] = Field(min_length=3, max_length=3)
    sensors: dict[str, float]

    def row(self) -> dict:
        return {"unit": self.unit, "cycle": self.cycle, **dict(zip(SETTINGS, self.settings)), **self.sensors}

    @field_validator("sensors")
    @classmethod
    def all_sensors(cls, value: dict[str, float]) -> dict[str, float]:
        missing = set(ALL_SENSORS) - value.keys()
        unknown = value.keys() - set(ALL_SENSORS)
        if missing or unknown:
            raise ValueError(f"expected exactly s1..s21; missing {sorted(missing)}, unknown {sorted(unknown)}")
        return value


class Store:
    """SQLite with one connection guarded by a lock: ingest is small and sequential per engine."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.executescript(
                """
                create table if not exists readings (
                  unit integer not null, cycle integer not null, data text not null,
                  primary key (unit, cycle));
                create table if not exists predictions (
                  unit integer not null, cycle integer not null, rul real not null, status text not null,
                  model_version text not null, primary key (unit, cycle));
                """
            )

    def add_reading(self, reading: Reading) -> bool:
        """False if this (unit, cycle) was already stored: a retried POST must not count twice."""
        with self.lock, self.db:
            cur = self.db.execute(
                "insert or ignore into readings values (?, ?, ?)", (reading.unit, reading.cycle, json.dumps(reading.row()))
            )
            return cur.rowcount == 1

    def recent(self, unit: int, upto_cycle: int, n: int) -> pd.DataFrame:
        with self.lock:
            rows = self.db.execute(
                "select data from readings where unit = ? and cycle <= ? order by cycle desc limit ?", (unit, upto_cycle, n)
            ).fetchall()
        return pd.DataFrame([json.loads(r[0]) for r in reversed(rows)])

    def add_prediction(self, unit: int, cycle: int, rul: float, status: str, version: str) -> None:
        with self.lock, self.db:
            self.db.execute("insert or replace into predictions values (?, ?, ?, ?, ?)", (unit, cycle, rul, status, version))

    def fleet(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                """select p.unit, p.cycle, p.rul, p.status from predictions p
                   join (select unit, max(cycle) as cycle from predictions group by unit) last
                     on last.unit = p.unit and last.cycle = p.cycle
                   order by p.rul"""
            ).fetchall()
        return [{"unit": u, "cycle": c, "rul": round(r, 1), "status": s} for u, c, r, s in rows]

    def engine(self, unit: int) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                """select p.cycle, p.rul, r.data from predictions p join readings r using (unit, cycle)
                   where p.unit = ? order by p.cycle""",
                (unit,),
            ).fetchall()
        return [{"cycle": c, "rul": round(rul, 1), **{s: json.loads(d)[s] for s in CHART_SENSORS}} for c, rul, d in rows]

    def reset(self) -> None:
        with self.lock, self.db:
            self.db.execute("delete from readings")
            self.db.execute("delete from predictions")


class Broadcaster:
    """Fan-out of prediction events to every open dashboard (server-sent events)."""

    def __init__(self):
        self.queues: set[asyncio.Queue] = set()

    def publish(self, event: dict) -> None:
        for q in list(self.queues):
            if q.qsize() < 1000:  # a stalled client must not grow memory without bound
                q.put_nowait(event)


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store(os.environ.get("MACHINEWATCH_DB", str(ROOT / "data" / "machinewatch.db")))


class Predictor:
    def __init__(self):
        registry = Registry()
        self.meta = registry.current()
        self.model = registry.load()

    def predict(self, recent: pd.DataFrame) -> float:
        features = engine_features(recent).iloc[[-1]]
        return float(self.model.predict(features)[0])


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


broadcaster = Broadcaster()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store()
    yield


app = FastAPI(title="MachineWatch", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # The default 422 echoes the rejected input back, and a NaN or Infinity in it cannot be written as JSON.
    errors = [{k: v for k, v in error.items() if k not in ("input", "ctx")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


def require_ingest_key(x_api_key: str | None = Header(default=None)) -> None:
    """Fail closed: without a configured key, writes are refused unless local development says otherwise."""
    expected = os.environ.get("MACHINEWATCH_INGEST_KEY")
    if not expected:
        if os.environ.get("MACHINEWATCH_INSECURE_DEV") == "1":
            return
        raise HTTPException(status_code=503, detail="Ingest is disabled: set MACHINEWATCH_INGEST_KEY.")
    if not (x_api_key and hmac.compare_digest(x_api_key.encode(), expected.encode())):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key.")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/readings", dependencies=[Depends(require_ingest_key)])
def ingest(reading: Reading, store: Store = Depends(get_store), predictor: Predictor = Depends(get_predictor)) -> dict:
    # Predict first, from the previous WINDOW - 1 cycles plus this one (all the features look at), and store
    # nothing if that fails, so a reading never sits in the database without its prediction.
    earlier = store.recent(reading.unit, reading.cycle - 1, WINDOW - 1)
    recent = pd.concat([earlier, pd.DataFrame([reading.row()])], ignore_index=True)
    rul = predictor.predict(recent)
    status = health_status(rul)
    stored = store.add_reading(reading)
    store.add_prediction(reading.unit, reading.cycle, rul, status, predictor.meta.version)
    event = {"unit": reading.unit, "cycle": reading.cycle, "rul": round(rul, 1), "status": status}
    if stored:
        broadcaster.publish(event)
    return {**event, "duplicate": not stored, "model_version": predictor.meta.version}


@app.get("/api/fleet")
def fleet(store: Store = Depends(get_store)) -> list[dict]:
    return store.fleet()


@app.get("/api/engines/{unit}")
def engine(unit: int, store: Store = Depends(get_store)) -> list[dict]:
    history = store.engine(unit)
    if not history:
        raise HTTPException(status_code=404, detail=f"No data for engine {unit}.")
    return history


@app.get("/api/model")
def model_info(predictor: Predictor = Depends(get_predictor)) -> dict:
    meta = predictor.meta
    return {"version": meta.version, "name": meta.name, "trained_at": meta.created_at, "test_metrics": meta.metrics}


@app.post("/api/reset", dependencies=[Depends(require_ingest_key)])
def reset(store: Store = Depends(get_store)) -> dict:
    if os.environ.get("MACHINEWATCH_ALLOW_RESET") != "1":
        raise HTTPException(status_code=403, detail="Reset is disabled on this deployment.")
    store.reset()
    broadcaster.publish({"reset": True})
    return {"status": "reset"}


@app.get("/api/stream")
async def stream(request: Request) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    broadcaster.queues.add(queue)

    async def events():
        try:
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            broadcaster.queues.discard(queue)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})
