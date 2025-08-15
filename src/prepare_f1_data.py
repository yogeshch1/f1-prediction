import json, os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

DATA_DIR = Path("data")                 # where your season .json files live
OUTPUT_CSV = Path("data/f1_results.csv")

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def parse_season_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    # Top-level is a dict keyed by round numbers: {"1": {...}, "2": {...}, ...}
    for _round_key, payload in data.items():
        mrdata = payload.get("MRData", {}) or {}
        race_table = mrdata.get("RaceTable", {}) or {}
        season = race_table.get("season")
        races = race_table.get("Races", []) or []

        for race in races:
            round_num = race.get("round")
            race_name = race.get("raceName")
            race_date = race.get("date")
            circuit = safe_get(race, "Circuit", "circuitName")
            results = race.get("Results", []) or []

            for res in results:
                driver = res.get("Driver", {}) or {}
                constructor = res.get("Constructor", {}) or {}
                fastest = res.get("FastestLap", {}) or {}
                fin_time = res.get("Time", {}) or {}
                rows.append({
                    "season": season,
                    "round": round_num,
                    "race_name": race_name,
                    "date": race_date,
                    "circuit": circuit,
                    "position": res.get("position"),
                    "position_text": res.get("positionText"),
                    "points": res.get("points"),
                    "driver_number": res.get("number"),
                    "driver_id": driver.get("driverId"),
                    "driver_code": driver.get("code"),
                    "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                    "constructor": constructor.get("name"),
                    "grid": res.get("grid"),
                    "laps": res.get("laps"),
                    "status": res.get("status"),
                    "finish_time": fin_time.get("time"),
                    "finish_time_millis": fin_time.get("millis"),
                    "fastest_lap_rank": fastest.get("rank"),
                    "fastest_lap_time": safe_get(fastest, "Time", "time"),
                    "fastest_lap_number": fastest.get("lap"),
                })
    return rows

def main():
    rows: List[Dict[str, Any]] = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".json"):
            rows.extend(parse_season_file(DATA_DIR / fname))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()