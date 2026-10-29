"""Replays C-MAPSS test engines into the API, one cycle at a time, as if their sensors were reporting live.

    python -m machinewatch.simulator --api http://localhost:8000 --engines 12 --rate 20

Engines are interleaved so the fleet ages together. When every engine has reported its last cycle, the
simulator resets the fleet (if the API allows it) and starts again.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time

import httpx

from . import data


def readings(frame, units):
    """Cycle-by-cycle readings of the chosen engines, interleaved: cycle 1 of every engine, then cycle 2..."""
    by_unit = {u: frame[frame.unit == u].sort_values("cycle") for u in units}
    for rows in itertools.zip_longest(*(g.itertuples(index=False) for g in by_unit.values())):
        for row in rows:
            if row is None:
                continue
            values = row._asdict()
            yield {
                "unit": int(values["unit"]),
                "cycle": int(values["cycle"]),
                "settings": [float(values[s]) for s in data.SETTINGS],
                "sensors": {s: float(values[s]) for s in data.SENSORS},
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=os.environ.get("MACHINEWATCH_API", "http://localhost:8000"))
    parser.add_argument("--engines", type=int, default=12)
    parser.add_argument("--rate", type=float, default=20.0, help="readings per second")
    parser.add_argument("--once", action="store_true", help="replay once instead of looping")
    args = parser.parse_args()

    test, true_rul = data.load_test()
    # The engines with the longest records show the whole arc from healthy to close to failure.
    units = test.groupby("unit")["cycle"].max().sort_values(ascending=False).index[: args.engines].tolist()
    headers = {"X-API-Key": os.environ["MACHINEWATCH_INGEST_KEY"]} if os.environ.get("MACHINEWATCH_INGEST_KEY") else {}

    with httpx.Client(base_url=args.api, headers=headers, timeout=10) as client:
        while True:
            for reading in readings(test, units):
                client.post("/api/readings", json=reading).raise_for_status()
                time.sleep(1 / args.rate)
            final = {f["unit"]: f["rul"] for f in client.get("/api/fleet").json()}
            for unit in units:
                print(f"engine {unit:3d}: predicted {final.get(unit, float('nan')):6.1f} cycles left, true {true_rul[unit]}")
            if args.once:
                break
            time.sleep(10)
            client.post("/api/reset")


if __name__ == "__main__":
    main()
