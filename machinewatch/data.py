"""NASA C-MAPSS turbofan degradation data (Saxena et al., 2008), from the NASA Prognostics Center of
Excellence data repository. Downloaded on demand into data/raw/; never committed.

Each row is one operating cycle of one engine: unit id, cycle number, 3 operational settings and 21
sensor readings. Training engines run until failure. Test engines stop earlier, and RUL_FD00x.txt gives
how many cycles each one had left: that is what the model must predict.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
INNER = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]

# Standard practice for C-MAPSS: early in life an engine shows no measurable wear, so the target RUL is
# capped (piecewise-linear degradation). Without the cap a model is asked to tell 300 from 200 cycles left
# on data that looks identical.
RUL_CAP = 125


def download(raw: Path = RAW) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "cmapss.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    return archive


def _read(name: str, raw: Path = RAW) -> bytes:
    outer = zipfile.ZipFile(download(raw))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER)))
    return inner.read(name)


def _table(name: str, raw: Path = RAW) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(_read(name, raw)), sep=r"\s+", header=None, names=COLUMNS)
    return frame.astype({"unit": int, "cycle": int})


def load_train(subset: str = "FD001", raw: Path = RAW) -> pd.DataFrame:
    """Run-to-failure histories with a capped RUL target."""
    frame = _table(f"train_{subset}.txt", raw)
    last = frame.groupby("unit")["cycle"].transform("max")
    frame["rul"] = np.minimum(last - frame["cycle"], RUL_CAP)
    return frame


def load_test(subset: str = "FD001", raw: Path = RAW) -> tuple[pd.DataFrame, pd.Series]:
    """Truncated histories, and the true RUL after each engine's last cycle (indexed by unit)."""
    frame = _table(f"test_{subset}.txt", raw)
    rul = pd.read_csv(io.BytesIO(_read(f"RUL_{subset}.txt", raw)), header=None, names=["rul"])["rul"]
    rul.index = np.arange(1, len(rul) + 1)
    rul.index.name = "unit"
    return frame, rul


if __name__ == "__main__":
    train = load_train()
    test, rul = load_test()
    print(f"train: {train.unit.nunique()} engines, {len(train)} cycles; test: {test.unit.nunique()} engines, true RUL {rul.min()}-{rul.max()}")
