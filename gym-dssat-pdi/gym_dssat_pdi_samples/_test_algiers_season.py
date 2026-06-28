"""Quick smoke test: full CAPQL season with Algiers NASA measured weather."""
from __future__ import annotations

import sys

from capql_inference import load_actor, run_season

W = [1 / 3, 1 / 3, 1 / 3]
SEEDS = (123, 456)


def main() -> int:
    actor = load_actor()
    failed = 0
    for seed in SEEDS:
        try:
            r = run_season(W, seed=seed, actor=actor, weather_id='wgen-nasa-algiers')
            print(f'seed={seed} OK days={r.ep_length} yield={r.yield_kg_ha:.0f} kg/ha')
        except Exception as exc:
            failed += 1
            print(f'seed={seed} FAIL: {exc}')
    return failed


if __name__ == '__main__':
    sys.exit(main())
