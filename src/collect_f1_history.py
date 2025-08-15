import os
import time
import json
import requests
from pathlib import Path

BASE_URL = "https://api.jolpi.ca/ergast/f1"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Seasons & Races range
START_YEAR = 2024
END_YEAR = 2025

# Retry settings
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds
REQUEST_DELAY = 1  # seconds between calls to avoid hitting rate limits

def fetch_with_retry(url):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = requests.get(url)
            if res.status_code == 429:
                print(f"Rate limit hit. Waiting {RETRY_DELAY} seconds... (Attempt {attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                continue
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            time.sleep(RETRY_DELAY)
    return None

def collect_f1_history():
    for year in range(START_YEAR, END_YEAR + 1):
        season_results = {}
        print(f"📅 Collecting data for {year}...")
        
        # Get races for the season
        schedule_url = f"{BASE_URL}/{year}.json"
        schedule_data = fetch_with_retry(schedule_url)
        if not schedule_data:
            print(f"❌ Failed to get schedule for {year}")
            continue

        races = schedule_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        print(f"Found {len(races)} races in {year}")

        for race in races:
            round_no = race.get("round")
            race_name = race.get("raceName")
            print(f"  🏁 {race_name} (Round {round_no})")

            results_url = f"{BASE_URL}/{year}/{round_no}/results.json"
            results_data = fetch_with_retry(results_url)
            if not results_data:
                print(f"  ❌ Failed to get results for {race_name}")
                continue

            season_results[round_no] = results_data
            time.sleep(REQUEST_DELAY)  # Avoid hitting the rate limit

        # Save season data
        file_path = DATA_DIR / f"{year}_season.json"
        with open(file_path, "w") as f:
            json.dump(season_results, f, indent=2)
        print(f"✅ Saved {year} season data to {file_path}")

if __name__ == "__main__":
    collect_f1_history()