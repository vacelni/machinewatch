"""A small file-based model registry: every trained model is kept with its metrics, and the API serves the
one marked current. A new model only becomes current if it is at least as good on the held-out test set,
so retraining can never silently make predictions worse.

    models/registry/<version>/model.joblib
    models/registry/<version>/meta.json
    models/registry/CURRENT          (the version the API serves)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent / "models" / "registry"


@dataclass
class ModelMeta:
    version: str
    name: str
    created_at: str
    metrics: dict
    sha256: str


class Registry:
    def __init__(self, root: Path = ROOT):
        self.root = root

    def _current_file(self) -> Path:
        return self.root / "CURRENT"

    def versions(self) -> list[ModelMeta]:
        metas = [ModelMeta(**json.loads(p.read_text())) for p in self.root.glob("*/meta.json")]
        return sorted(metas, key=lambda m: m.created_at)

    def current_version(self) -> str | None:
        f = self._current_file()
        return f.read_text().strip() if f.exists() else None

    def current(self) -> ModelMeta | None:
        version = self.current_version()
        return next((m for m in self.versions() if m.version == version), None)

    def register(self, model, name: str, metrics: dict) -> ModelMeta:
        created = datetime.now(timezone.utc)
        staging = self.root / f".staging-{created.timestamp()}"
        staging.mkdir(parents=True)
        joblib.dump(model, staging / "model.joblib")
        digest = hashlib.sha256((staging / "model.joblib").read_bytes()).hexdigest()
        version = f"{created:%Y%m%d-%H%M%S}-{digest[:8]}"
        meta = ModelMeta(version, name, created.isoformat(), metrics, digest)
        (staging / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
        staging.rename(self.root / version)
        return meta

    def promote_if_better(self, meta: ModelMeta, metric: str = "rmse") -> bool:
        """Make [meta] current unless the current model has a lower (better) [metric]."""
        current = self.current()
        if current is not None and current.metrics[metric] < meta.metrics[metric]:
            return False
        self._current_file().write_text(meta.version)
        return True

    def load(self, version: str | None = None):
        version = version or self.current_version()
        if version is None:
            raise FileNotFoundError("No model has been promoted yet: run `python -m machinewatch.train`.")
        path = self.root / version / "model.joblib"
        expected = json.loads((self.root / version / "meta.json").read_text())["sha256"]
        # joblib unpickles: refuse a file that is not byte-for-byte the one this registry wrote.
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"model {version} does not match its recorded checksum")
        return joblib.load(path)
