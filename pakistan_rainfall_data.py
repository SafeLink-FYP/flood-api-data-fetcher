"""
Pakistan Flood Prediction — Data Collection Pipeline
=====================================================
Collects historical weather + river discharge data for all major Pakistani cities
using the Open-Meteo free API with PERSISTENT API call tracking.

Run this script multiple times across days — it auto-resumes where it left off.

Open-Meteo Free Tier Limits (non-commercial):
  ┌─────────────────────────────────────────────┐
  │  600 API calls / minute                     │
  │  5,000 API calls / hour                     │
  │  10,000 API calls / day                     │
  │  300,000 API calls / month                  │
  └─────────────────────────────────────────────┘

  API call counting (fractional):
    calls = max(1, variables/10) × max(1, days/14) × locations_per_request

Data collected:
  Phase 1 — Daily weather (11 vars, 2000-2025, 56 cities)
            precipitation, rain, snowfall, temperature, wind, ET₀, weather codes
  Phase 2 — River discharge from GloFAS Flood API (1997-2025, 56 cities)
            Crucial for flood prediction (target variable / labels)
            Note: seamless_v4 data starts 1997; pre-1997 returns all nulls

For Bi-LSTM Flood Prediction you will also need:
  - Flood event labels (derived from discharge thresholds or NDMA records)
  - Engineered features: rolling precipitation sums (3/7/14/30-day),
    antecedent precipitation index, seasonal indicators
  - Optional: hourly soil moisture, upstream city precipitation, elevation

API Docs:
  Historical Weather: https://open-meteo.com/en/docs/historical-weather-api
  Flood API:          https://open-meteo.com/en/docs/flood-api
  Terms:              https://open-meteo.com/en/terms
"""

import requests
import pandas as pd
import json
import os
import time
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
TRACKER_FILE = SCRIPT_DIR / "api_call_tracker.json"

# API endpoints
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_API_URL = "https://flood-api.open-meteo.com/v1/flood"

# Date ranges
WEATHER_START_YEAR = 2000
WEATHER_END_YEAR = 2025
FLOOD_START_YEAR = 1997  # seamless_v4 model has data from 1997 (pre-1997 returns all nulls)
FLOOD_END_YEAR = 2025    # seamless_v4 blends reanalysis + forecast; data available through 2025+

# Free tier rate limits
DAILY_LIMIT = 10_000
HOURLY_LIMIT = 5_000
MINUTE_LIMIT = 600
MONTHLY_LIMIT = 300_000

# Safety margin — stop at 90% of limits to avoid bans
SAFETY_FACTOR = 0.90

# Delay between HTTP requests (seconds)
REQUEST_DELAY = 3.0

# Temporarily disable startup probe if it causes false stops.
ENABLE_PREFLIGHT_CHECK = False

# Retry settings for 429/5xx errors
MAX_RETRIES = 6
RETRY_BASE_DELAY = 30  # seconds, doubles each retry

# Daily weather variables for flood prediction
DAILY_WEATHER_VARS = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "et0_fao_evapotranspiration",
]

# ═════════════════════════════════════════════════════════════════════════════
# PAKISTAN CITIES DATABASE — 56 cities with coordinates
# ═════════════════════════════════════════════════════════════════════════════

PAKISTAN_CITIES = [
    # ── Punjab (25 cities) ──
    {"city": "Lahore",            "province": "Punjab",          "lat": 31.5204, "lon": 74.3587},
    {"city": "Faisalabad",        "province": "Punjab",          "lat": 31.4504, "lon": 73.1350},
    {"city": "Rawalpindi",        "province": "Punjab",          "lat": 33.5651, "lon": 73.0169},
    {"city": "Multan",            "province": "Punjab",          "lat": 30.1575, "lon": 71.5249},
    {"city": "Gujranwala",        "province": "Punjab",          "lat": 32.1877, "lon": 74.1945},
    {"city": "Sialkot",           "province": "Punjab",          "lat": 32.4945, "lon": 74.5229},
    {"city": "Bahawalpur",        "province": "Punjab",          "lat": 29.3544, "lon": 71.6911},
    {"city": "Sargodha",          "province": "Punjab",          "lat": 32.0740, "lon": 72.6861},
    {"city": "Gujrat",            "province": "Punjab",          "lat": 32.5739, "lon": 74.0789},
    {"city": "Sahiwal",           "province": "Punjab",          "lat": 30.6682, "lon": 73.1114},
    {"city": "Jhang",             "province": "Punjab",          "lat": 31.2681, "lon": 72.3181},
    {"city": "Rahim Yar Khan",    "province": "Punjab",          "lat": 28.4202, "lon": 70.2952},
    {"city": "Okara",             "province": "Punjab",          "lat": 30.8138, "lon": 73.4534},
    {"city": "Kasur",             "province": "Punjab",          "lat": 31.1185, "lon": 74.4630},
    {"city": "Chiniot",           "province": "Punjab",          "lat": 31.7140, "lon": 72.9858},
    {"city": "Jhelum",            "province": "Punjab",          "lat": 32.9425, "lon": 73.7257},
    {"city": "Mianwali",          "province": "Punjab",          "lat": 32.5853, "lon": 71.5436},
    {"city": "Attock",            "province": "Punjab",          "lat": 33.7667, "lon": 72.3597},
    {"city": "Chakwal",           "province": "Punjab",          "lat": 32.9328, "lon": 72.8630},
    {"city": "Toba Tek Singh",    "province": "Punjab",          "lat": 30.9709, "lon": 72.4826},
    {"city": "Vehari",            "province": "Punjab",          "lat": 30.0445, "lon": 72.3556},
    {"city": "Khanewal",          "province": "Punjab",          "lat": 30.3017, "lon": 71.9321},
    {"city": "Lodhran",           "province": "Punjab",          "lat": 29.5432, "lon": 71.6369},
    {"city": "Muzaffargarh",      "province": "Punjab",          "lat": 30.0736, "lon": 71.1936},
    {"city": "Dera Ghazi Khan",   "province": "Punjab",          "lat": 30.0489, "lon": 70.6455},

    # ── Sindh (9 cities) ──
    {"city": "Karachi",           "province": "Sindh",           "lat": 24.8607, "lon": 67.0011},
    {"city": "Hyderabad",         "province": "Sindh",           "lat": 25.3960, "lon": 68.3578},
    {"city": "Sukkur",            "province": "Sindh",           "lat": 27.7052, "lon": 68.8574},
    {"city": "Larkana",           "province": "Sindh",           "lat": 27.5570, "lon": 68.2028},
    {"city": "Nawabshah",         "province": "Sindh",           "lat": 26.2483, "lon": 68.4098},
    {"city": "Jacobabad",         "province": "Sindh",           "lat": 28.2769, "lon": 68.4514},
    {"city": "Mirpur Khas",       "province": "Sindh",           "lat": 25.5276, "lon": 69.0159},
    {"city": "Thatta",            "province": "Sindh",           "lat": 24.7461, "lon": 67.9236},
    {"city": "Badin",             "province": "Sindh",           "lat": 24.6560, "lon": 68.8370},

    # ── Khyber Pakhtunkhwa (9 cities) ──
    {"city": "Peshawar",          "province": "KPK",             "lat": 34.0151, "lon": 71.5249},
    {"city": "Mardan",            "province": "KPK",             "lat": 34.1988, "lon": 72.0404},
    {"city": "Abbottabad",        "province": "KPK",             "lat": 34.1688, "lon": 73.2215},
    {"city": "Mingora (Swat)",    "province": "KPK",             "lat": 34.7717, "lon": 72.3609},
    {"city": "Kohat",             "province": "KPK",             "lat": 33.5869, "lon": 71.4414},
    {"city": "Bannu",             "province": "KPK",             "lat": 32.9889, "lon": 70.6042},
    {"city": "Dera Ismail Khan",  "province": "KPK",             "lat": 31.8626, "lon": 70.9019},
    {"city": "Chitral",           "province": "KPK",             "lat": 35.8518, "lon": 71.7864},
    {"city": "Dir",               "province": "KPK",             "lat": 35.2000, "lon": 71.8800},

    # ── Balochistan (7 cities) ──
    {"city": "Quetta",            "province": "Balochistan",     "lat": 30.1798, "lon": 66.9750},
    {"city": "Turbat",            "province": "Balochistan",     "lat": 26.0031, "lon": 63.0441},
    {"city": "Gwadar",            "province": "Balochistan",     "lat": 25.1264, "lon": 62.3225},
    {"city": "Zhob",              "province": "Balochistan",     "lat": 31.3515, "lon": 69.4493},
    {"city": "Khuzdar",           "province": "Balochistan",     "lat": 27.8000, "lon": 66.6167},
    {"city": "Sibi",              "province": "Balochistan",     "lat": 29.5430, "lon": 67.8770},
    {"city": "Kalat",             "province": "Balochistan",     "lat": 29.0225, "lon": 66.5900},

    # ── Islamabad Capital Territory ──
    {"city": "Islamabad",         "province": "ICT",             "lat": 33.6844, "lon": 73.0479},

    # ── Azad Jammu & Kashmir (2 cities) ──
    {"city": "Muzaffarabad",      "province": "AJK",             "lat": 34.3700, "lon": 73.4717},
    {"city": "Mirpur",            "province": "AJK",             "lat": 33.1483, "lon": 73.7514},

    # ── Gilgit-Baltistan (3 cities) ──
    {"city": "Gilgit",            "province": "Gilgit-Baltistan","lat": 35.9208, "lon": 74.3144},
    {"city": "Skardu",            "province": "Gilgit-Baltistan","lat": 35.2971, "lon": 75.6333},
    {"city": "Hunza (Karimabad)", "province": "Gilgit-Baltistan","lat": 36.3167, "lon": 74.6600},
]


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENT API CALL TRACKER
# ═════════════════════════════════════════════════════════════════════════════

class APICallTracker:
    """
    Tracks every API call with timestamps and estimated fractional cost.
    Persists to JSON so usage survives across program restarts.

    Open-Meteo fractional API call formula:
        estimated_calls = max(1, n_vars/10) * max(1, n_days/14) * n_locations
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> dict:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                print("  WARNING: Corrupt tracker file, creating backup and starting fresh.")
                backup = self.filepath.with_suffix(".json.bak")
                self.filepath.rename(backup)
        return {
            "calls": [],               # [{"ts": ..., "est": ..., "type": ..., "task": ...}]
            "completed_tasks": [],      # list of task_id strings
            "total_http_requests": 0,
            "created": datetime.now().isoformat(),
        }

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    # ── Recording ──

    def record_call(self, estimated_calls: float, call_type: str, task_id: str):
        """Record one HTTP request with its estimated fractional API call cost."""
        self.data["calls"].append({
            "ts": datetime.now().isoformat(),
            "est": round(estimated_calls, 2),
            "type": call_type,
            "task": task_id,
        })
        self.data["total_http_requests"] += 1
        # Auto-prune call log entries older than 35 days to keep file manageable
        cutoff = (datetime.now() - timedelta(days=35)).isoformat()
        self.data["calls"] = [c for c in self.data["calls"] if c["ts"] >= cutoff]
        self.save()

    def mark_completed(self, task_id: str):
        if task_id not in self.data["completed_tasks"]:
            self.data["completed_tasks"].append(task_id)
            self.save()

    def is_completed(self, task_id: str) -> bool:
        return task_id in self.data["completed_tasks"]

    # ── Usage Queries ──

    def _sum_est_since(self, cutoff_iso: str) -> float:
        # Use parsed datetimes and clamp at "now" so future timestamps
        # (e.g., from timezone shifts) do not keep limits blocked.
        cutoff = datetime.fromisoformat(cutoff_iso)
        now = datetime.now()
        total = 0.0
        for c in self.data["calls"]:
            try:
                ts = datetime.fromisoformat(c["ts"])
                est = float(c["est"])
            except (KeyError, TypeError, ValueError):
                continue
            if cutoff <= ts <= now:
                total += est
        return total

    def _count_since(self, cutoff_iso: str) -> int:
        cutoff = datetime.fromisoformat(cutoff_iso)
        now = datetime.now()
        count = 0
        for c in self.data["calls"]:
            try:
                ts = datetime.fromisoformat(c["ts"])
            except (KeyError, TypeError, ValueError):
                continue
            if cutoff <= ts <= now:
                count += 1
        return count

    def calls_today(self) -> float:
        return self._sum_est_since(date.today().isoformat())

    def calls_this_hour(self) -> float:
        return self._sum_est_since((datetime.now() - timedelta(hours=1)).isoformat())

    def calls_this_minute(self) -> float:
        return self._sum_est_since((datetime.now() - timedelta(minutes=1)).isoformat())

    def calls_this_month(self) -> float:
        return self._sum_est_since(date.today().replace(day=1).isoformat())

    def http_requests_today(self) -> int:
        return self._count_since(date.today().isoformat())

    # ── Limit Checks ──

    @staticmethod
    def estimate_api_calls(n_vars: int, n_days: int, n_locations: int = 1) -> float:
        """
        Estimate fractional API call cost per Open-Meteo's formula.
        calls = max(1, vars/10) * max(1, days/14) * locations
        """
        var_factor = max(1.0, n_vars / 10.0)
        time_factor = max(1.0, n_days / 14.0)
        return var_factor * time_factor * n_locations

    def can_make_call(self, estimated_calls: float) -> tuple:
        """Check all rate limits. Returns (ok: bool, reason: str)."""
        checks = [
            (self.calls_today() + estimated_calls, DAILY_LIMIT, "Daily"),
            (self.calls_this_hour() + estimated_calls, HOURLY_LIMIT, "Hourly"),
            (self.calls_this_minute() + estimated_calls, MINUTE_LIMIT, "Per-minute"),
            (self.calls_this_month() + estimated_calls, MONTHLY_LIMIT, "Monthly"),
        ]
        for current, limit, name in checks:
            safe_limit = limit * SAFETY_FACTOR
            if current > safe_limit:
                return False, (
                    f"{name} limit: {current:.0f} / {limit} "
                    f"(safe threshold: {safe_limit:.0f})"
                )
        return True, "OK"

    def wait_if_needed(self, estimated_calls: float):
        """Block until it's safe to make the next call. Exits if daily/monthly limit hit."""
        while True:
            ok, reason = self.can_make_call(estimated_calls)
            if ok:
                return

            if "Per-minute" in reason:
                wait = 62
                print(f"\n    ... Per-minute rate limit, pausing {wait}s ...")
                time.sleep(wait)
            elif "Hourly" in reason:
                wait = 300
                print(f"\n    ... Hourly rate limit, pausing {wait}s ...")
                time.sleep(wait)
            elif "Daily" in reason:
                print()
                print("  ╔══════════════════════════════════════════════════════════╗")
                print("  ║  DAILY API LIMIT REACHED — STOPPING FOR TODAY           ║")
                print("  ║  Re-run this script tomorrow to continue fetching.      ║")
                print("  ║  All progress has been saved automatically.             ║")
                print("  ╚══════════════════════════════════════════════════════════╝")
                self.print_status()
                sys.exit(0)
            elif "Monthly" in reason:
                print()
                print("  ╔══════════════════════════════════════════════════════════╗")
                print("  ║  MONTHLY API LIMIT REACHED — STOPPING                   ║")
                print("  ║  Re-run next month to continue. Progress saved.         ║")
                print("  ╚══════════════════════════════════════════════════════════╝")
                self.print_status()
                sys.exit(0)
            else:
                time.sleep(120)

    # ── Status Display ──

    def print_status(self):
        completed = len(self.data["completed_tasks"])
        weather_done = sum(1 for t in self.data["completed_tasks"] if t.startswith("weather_"))
        flood_done = sum(1 for t in self.data["completed_tasks"] if t.startswith("flood_"))

        total_weather = len(PAKISTAN_CITIES) * (WEATHER_END_YEAR - WEATHER_START_YEAR + 1)
        total_flood = len(PAKISTAN_CITIES) * (FLOOD_END_YEAR - FLOOD_START_YEAR + 1)

        print()
        print("  ╔════════════════════════════════════════════════════════════╗")
        print("  ║              API CALL TRACKER STATUS                      ║")
        print("  ╠════════════════════════════════════════════════════════════╣")
        print(f"  ║  Today:        {self.calls_today():>8.0f} / {DAILY_LIMIT:>8,}  est. calls   ║")
        print(f"  ║  This hour:    {self.calls_this_hour():>8.0f} / {HOURLY_LIMIT:>8,}  est. calls   ║")
        print(f"  ║  This minute:  {self.calls_this_minute():>8.0f} / {MINUTE_LIMIT:>8,}    est. calls   ║")
        print(f"  ║  This month:   {self.calls_this_month():>8.0f} / {MONTHLY_LIMIT:>8,}  est. calls   ║")
        print(f"  ║  HTTP reqs:    {self.http_requests_today():>8d}   today                ║")
        print("  ╠════════════════════════════════════════════════════════════╣")
        print(f"  ║  Weather tasks: {weather_done:>5d} / {total_weather:>5d}  completed          ║")
        print(f"  ║  Flood tasks:   {flood_done:>5d} / {total_flood:>5d}  completed          ║")
        print("  ╚════════════════════════════════════════════════════════════╝")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def fetch_with_retry(url: str, params: dict, tracker: APICallTracker,
                     est_calls: float, call_type: str, task_id: str) -> dict:
    """
    Make an API request with:
      - Pre-call rate limit waiting
      - Retry with exponential backoff on 429/5xx/timeout
      - Persistent call recording on success
    Returns parsed JSON response.
    """
    tracker.wait_if_needed(est_calls)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=120)

            if response.status_code == 429:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"\n    Rate limited (429). Retry {attempt+1}/{MAX_RETRIES} in {wait}s...", end=" ")
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"\n    Server error ({response.status_code}). Retry {attempt+1}/{MAX_RETRIES} in {wait}s...", end=" ")
                time.sleep(wait)
                continue

            response.raise_for_status()

            # Success — record the call
            tracker.record_call(est_calls, call_type, task_id)
            return response.json()

        except requests.exceptions.Timeout:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"\n    Timeout. Retry {attempt+1}/{MAX_RETRIES} in {wait}s...", end=" ")
            time.sleep(wait)
        except requests.exceptions.ConnectionError:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"\n    Connection error. Retry {attempt+1}/{MAX_RETRIES} in {wait}s...", end=" ")
            time.sleep(wait)
        except (requests.exceptions.RequestException, OSError, IOError) as e:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"\n    Network error ({type(e).__name__}). Retry {attempt+1}/{MAX_RETRIES} in {wait}s...", end=" ")
            time.sleep(wait)
        except KeyboardInterrupt:
            print("\n  Interrupted by user. Progress saved.")
            tracker.save()
            sys.exit(0)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries for task: {task_id}")


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def fetch_weather_year(city: dict, year: int, tracker: APICallTracker) -> pd.DataFrame:
    """Fetch one year of daily weather data for one city. Returns DataFrame or None."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    n_days = 366 if _is_leap_year(year) else 365

    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_WEATHER_VARS),
        "timezone": "Asia/Karachi",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
    }

    est_calls = APICallTracker.estimate_api_calls(len(DAILY_WEATHER_VARS), n_days)
    task_id = f"weather_{city['city']}_{year}"

    data = fetch_with_retry(WEATHER_API_URL, params, tracker, est_calls, "weather", task_id)

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        return None

    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    df.insert(0, "city", city["city"])
    df.insert(1, "province", city["province"])
    df.insert(2, "latitude", data.get("latitude", city["lat"]))
    df.insert(3, "longitude", data.get("longitude", city["lon"]))
    df.insert(4, "elevation_m", data.get("elevation", None))

    return df


def fetch_flood_year(city: dict, year: int, tracker: APICallTracker) -> pd.DataFrame:
    """Fetch one year of river discharge from GloFAS Flood API. Returns DataFrame or None."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    n_days = 366 if _is_leap_year(year) else 365

    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": "river_discharge",
        "models": "seamless_v4",
    }

    est_calls = APICallTracker.estimate_api_calls(1, n_days)
    task_id = f"flood_{city['city']}_{year}"

    data = fetch_with_retry(FLOOD_API_URL, params, tracker, est_calls, "flood", task_id)

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        return None

    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    df.insert(0, "city", city["city"])
    df.insert(1, "province", city["province"])
    df.insert(2, "latitude", data.get("latitude", city["lat"]))
    df.insert(3, "longitude", data.get("longitude", city["lon"]))

    return df


# ═════════════════════════════════════════════════════════════════════════════
# CSV HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def append_to_csv(df: pd.DataFrame, filepath: Path):
    """Append DataFrame to CSV, writing header only if file doesn't exist yet."""
    write_header = not filepath.exists() or filepath.stat().st_size == 0
    df.to_csv(filepath, mode="a", header=write_header, index=False)


# ═════════════════════════════════════════════════════════════════════════════
# COLLECTION PHASES
# ═════════════════════════════════════════════════════════════════════════════

def run_weather_collection(tracker: APICallTracker):
    """Phase 1: Collect daily weather data for all cities, year by year."""
    DATA_DIR.mkdir(exist_ok=True)
    weather_file = DATA_DIR / "pakistan_weather_daily.csv"

    years = list(range(WEATHER_START_YEAR, WEATHER_END_YEAR + 1))
    total_tasks = len(PAKISTAN_CITIES) * len(years)
    pending = []

    for city in PAKISTAN_CITIES:
        for year in years:
            task_id = f"weather_{city['city']}_{year}"
            if not tracker.is_completed(task_id):
                pending.append((city, year, task_id))

    if not pending:
        print("  All weather data already collected!")
        return

    done_count = total_tasks - len(pending)
    print(f"  Progress: {done_count}/{total_tasks} done, {len(pending)} remaining")
    print()

    for i, (city, year, task_id) in enumerate(pending):
        n_days = 366 if _is_leap_year(year) else 365
        est = APICallTracker.estimate_api_calls(len(DAILY_WEATHER_VARS), n_days)

        print(f"  [{done_count + i + 1}/{total_tasks}] "
              f"{city['city']:25s} {year} (~{est:.0f} est. calls) ", end="", flush=True)

        try:
            df = fetch_weather_year(city, year, tracker)
            if df is not None and len(df) > 0:
                append_to_csv(df, weather_file)
                tracker.mark_completed(task_id)
                print(f"OK  {len(df)} days")
            else:
                print("NO DATA")
        except RuntimeError as e:
            print(f"FAILED: {e}")
        except Exception as e:
            print(f"ERROR: {e}")

        # Rate-limit delay
        if i < len(pending) - 1:
            time.sleep(REQUEST_DELAY)

        # Show tracker status periodically
        if (i + 1) % 25 == 0:
            tracker.print_status()


def run_flood_collection(tracker: APICallTracker):
    """Phase 2: Collect river discharge data from GloFAS Flood API."""
    DATA_DIR.mkdir(exist_ok=True)
    flood_file = DATA_DIR / "pakistan_river_discharge.csv"

    years = list(range(FLOOD_START_YEAR, FLOOD_END_YEAR + 1))
    total_tasks = len(PAKISTAN_CITIES) * len(years)
    pending = []

    for city in PAKISTAN_CITIES:
        for year in years:
            task_id = f"flood_{city['city']}_{year}"
            if not tracker.is_completed(task_id):
                pending.append((city, year, task_id))

    if not pending:
        print("  All flood/discharge data already collected!")
        return

    done_count = total_tasks - len(pending)
    print(f"  Progress: {done_count}/{total_tasks} done, {len(pending)} remaining")
    print()

    for i, (city, year, task_id) in enumerate(pending):
        n_days = 212 if year == FLOOD_END_YEAR else (366 if _is_leap_year(year) else 365)
        est = APICallTracker.estimate_api_calls(1, n_days)

        print(f"  [{done_count + i + 1}/{total_tasks}] "
              f"{city['city']:25s} {year} (~{est:.0f} est. calls) ", end="", flush=True)

        try:
            df = fetch_flood_year(city, year, tracker)
            if df is not None and len(df) > 0:
                append_to_csv(df, flood_file)
                tracker.mark_completed(task_id)
                print(f"OK  {len(df)} days")
            else:
                print("NO DATA (city may not be near a major river)")
                tracker.mark_completed(task_id)  # don't retry
        except RuntimeError as e:
            print(f"FAILED: {e}")
        except Exception as e:
            print(f"ERROR: {e}")

        if i < len(pending) - 1:
            time.sleep(REQUEST_DELAY)

        if (i + 1) % 25 == 0:
            tracker.print_status()


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARIES & FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def compute_summaries():
    """Generate summary CSVs from complete weather data."""
    weather_file = DATA_DIR / "pakistan_weather_daily.csv"
    if not weather_file.exists():
        print("  No weather data file found. Skipping summaries.")
        return

    print("  Loading weather data...", end=" ", flush=True)
    df = pd.read_csv(weather_file, parse_dates=["date"])
    df = df.drop_duplicates(subset=["city", "date"], keep="last")
    # Re-save deduplicated version
    df.to_csv(weather_file, index=False)
    print(f"{len(df):,} records for {df['city'].nunique()} cities")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # 1. Annual rainfall by city
    annual = (
        df.groupby(["city", "province", "year"])
        .agg(
            total_precipitation_mm=("precipitation_sum", "sum"),
            total_rain_mm=("rain_sum", "sum"),
            total_snowfall_cm=("snowfall_sum", "sum"),
            total_precip_hours=("precipitation_hours", "sum"),
            rainy_days=("precipitation_sum", lambda x: (x > 0).sum()),
            heavy_rain_days_gt25mm=("precipitation_sum", lambda x: (x > 25).sum()),
            very_heavy_days_gt50mm=("precipitation_sum", lambda x: (x > 50).sum()),
            extreme_days_gt100mm=("precipitation_sum", lambda x: (x > 100).sum()),
            max_daily_precip_mm=("precipitation_sum", "max"),
            avg_daily_precip_mm=("precipitation_sum", "mean"),
        )
        .reset_index()
        .round(2)
    )
    annual.to_csv(DATA_DIR / "pakistan_rainfall_annual_by_city.csv", index=False)
    print(f"  Saved annual summary: {len(annual):,} rows")

    # 2. Monthly rainfall by city
    monthly = (
        df.groupby(["city", "province", "year", "month"])
        .agg(
            total_precipitation_mm=("precipitation_sum", "sum"),
            total_rain_mm=("rain_sum", "sum"),
            total_snowfall_cm=("snowfall_sum", "sum"),
            total_precip_hours=("precipitation_hours", "sum"),
            rainy_days=("precipitation_sum", lambda x: (x > 0).sum()),
            max_daily_precip_mm=("precipitation_sum", "max"),
            avg_daily_precip_mm=("precipitation_sum", "mean"),
        )
        .reset_index()
        .round(2)
    )
    monthly.to_csv(DATA_DIR / "pakistan_rainfall_monthly_by_city.csv", index=False)
    print(f"  Saved monthly summary: {len(monthly):,} rows")

    # 3. Monthly climatology
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    climatology = (
        monthly.groupby(["city", "province", "month"])
        .agg(
            avg_monthly_precip_mm=("total_precipitation_mm", "mean"),
            median_monthly_precip=("total_precipitation_mm", "median"),
            max_monthly_precip_mm=("total_precipitation_mm", "max"),
            min_monthly_precip_mm=("total_precipitation_mm", "min"),
            std_monthly_precip_mm=("total_precipitation_mm", "std"),
            avg_rainy_days=("rainy_days", "mean"),
        )
        .reset_index()
    )
    climatology["month_name"] = climatology["month"].map(month_names)
    climatology = climatology.round(2)
    climatology.to_csv(DATA_DIR / "pakistan_rainfall_monthly_climatology.csv", index=False)
    print(f"  Saved climatology: {len(climatology):,} rows")

    # 4. City summary
    city_summary = (
        annual.groupby(["city", "province"])
        .agg(
            years_of_data=("year", "count"),
            avg_annual_precip_mm=("total_precipitation_mm", "mean"),
            max_annual_precip_mm=("total_precipitation_mm", "max"),
            min_annual_precip_mm=("total_precipitation_mm", "min"),
            std_annual_precip_mm=("total_precipitation_mm", "std"),
            avg_rainy_days_per_year=("rainy_days", "mean"),
            avg_heavy_rain_days=("heavy_rain_days_gt25mm", "mean"),
            max_daily_precip_ever_mm=("max_daily_precip_mm", "max"),
        )
        .reset_index()
        .sort_values("avg_annual_precip_mm", ascending=False)
        .round(2)
    )
    city_summary.to_csv(DATA_DIR / "pakistan_rainfall_city_summary.csv", index=False)
    print(f"  Saved city summary: {len(city_summary):,} rows")

    # Print highlights
    print(f"\n  {'─'*60}")
    print("  TOP 10 WETTEST CITIES (Avg Annual Precipitation mm)")
    print(f"  {'─'*60}")
    for _, r in city_summary.head(10).iterrows():
        print(f"  {r['city']:25s} {r['province']:15s} {r['avg_annual_precip_mm']:8.1f} mm/yr")

    print(f"\n  {'─'*60}")
    print("  TOP 5 DRIEST CITIES")
    print(f"  {'─'*60}")
    for _, r in city_summary.tail(5).iloc[::-1].iterrows():
        print(f"  {r['city']:25s} {r['province']:15s} {r['avg_annual_precip_mm']:8.1f} mm/yr")


def merge_and_engineer_features():
    """
    Merge weather + flood data and compute derived features for Bi-LSTM.
    """
    weather_file = DATA_DIR / "pakistan_weather_daily.csv"
    flood_file = DATA_DIR / "pakistan_river_discharge.csv"

    if not weather_file.exists():
        print("  No weather data. Skipping.")
        return None

    print("  Loading weather data...", end=" ", flush=True)
    weather = pd.read_csv(weather_file, parse_dates=["date"])
    weather = weather.drop_duplicates(subset=["city", "date"], keep="last")
    print(f"{len(weather):,} rows")

    if flood_file.exists():
        print("  Loading flood data...", end=" ", flush=True)
        flood = pd.read_csv(flood_file, parse_dates=["date"])
        flood = flood.drop_duplicates(subset=["city", "date"], keep="last")
        print(f"{len(flood):,} rows")

        # Merge on city + date
        merged = weather.merge(
            flood[["city", "date", "river_discharge"]],
            on=["city", "date"],
            how="left"
        )
    else:
        print("  No flood data file yet — proceeding with weather only.")
        merged = weather.copy()
        merged["river_discharge"] = None

    # ── Derived features for Bi-LSTM flood prediction ──
    print("  Computing derived features...")
    merged = merged.sort_values(["city", "date"]).reset_index(drop=True)

    # Rolling precipitation sums
    for window in [3, 7, 14, 30]:
        col = f"precip_rolling_{window}d"
        merged[col] = (
            merged.groupby("city")["precipitation_sum"]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )

    # Antecedent Precipitation Index (exponential decay, k=0.85)
    def compute_api(series, k=0.85):
        result = series.values.copy().astype(float)
        for i in range(1, len(result)):
            if pd.notna(result[i]) and pd.notna(result[i-1]):
                result[i] = result[i] + k * result[i-1]
        return pd.Series(result, index=series.index)

    merged["antecedent_precip_index"] = (
        merged.groupby("city")["precipitation_sum"]
        .transform(compute_api)
    )

    # Temporal features
    merged["month"] = merged["date"].dt.month
    merged["day_of_year"] = merged["date"].dt.dayofyear
    merged["is_monsoon"] = merged["month"].isin([6, 7, 8, 9]).astype(int)
    merged["year"] = merged["date"].dt.year

    # Precipitation intensity category (IMD classification)
    merged["precip_category"] = pd.cut(
        merged["precipitation_sum"].fillna(0),
        bins=[-0.1, 0, 2.5, 7.5, 35.5, 64.5, 124.5, float("inf")],
        labels=["none", "trace", "light", "moderate", "heavy", "very_heavy", "extreme"],
    )

    # Temperature range (proxy for weather system strength)
    merged["temp_range"] = merged["temperature_2m_max"] - merged["temperature_2m_min"]

    # Days since last rain
    def days_since_rain(precip_series, threshold=0.1):
        result = []
        count = 0
        for val in precip_series:
            if pd.notna(val) and val > threshold:
                count = 0
            else:
                count += 1
            result.append(count)
        return result

    merged["days_since_rain"] = (
        merged.groupby("city")["precipitation_sum"]
        .transform(days_since_rain)
    )

    # Consecutive wet days
    def consecutive_wet(precip_series, threshold=0.1):
        result = []
        count = 0
        for val in precip_series:
            if pd.notna(val) and val > threshold:
                count += 1
            else:
                count = 0
            result.append(count)
        return result

    merged["consecutive_wet_days"] = (
        merged.groupby("city")["precipitation_sum"]
        .transform(consecutive_wet)
    )

    # Save final dataset
    output_file = DATA_DIR / "pakistan_flood_prediction_dataset.csv"
    merged.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_file.name} ({size_mb:.1f} MB, {len(merged):,} rows)")

    if "river_discharge" in merged.columns:
        has_discharge = merged["river_discharge"].notna().sum()
        print(f"  Rows with river discharge: {has_discharge:,} / {len(merged):,}")

    return merged


def print_flood_prediction_guide():
    """Print what the user needs next for the Bi-LSTM model."""
    print()
    print("  " + "=" * 62)
    print("  BI-LSTM FLOOD PREDICTION — NEXT STEPS GUIDE")
    print("  " + "=" * 62)
    print("""
  DATA COLLECTED:
  ───────────────
  1. Daily weather (2000-2025): precipitation, rain, snowfall,
     temperature, wind, ET0, weather codes — 56 cities
  2. River discharge (1984-2022): GloFAS simulated discharge
  3. Derived features: rolling precip sums (3/7/14/30-day),
     antecedent precip index, monsoon flag, consecutive wet days,
     days since rain, temperature range, precip category

  WHAT YOU STILL NEED:
  ────────────────────
  A) FLOOD LABELS (training target — pick one approach):
     - Threshold: river_discharge > 2yr return period -> "flood"
     - NDMA Pakistan flood records (historical events)
     - EM-DAT international disaster database
     - GloFAS alert threshold levels

  B) OPTIONAL DATA (improves accuracy):
     - Hourly precipitation intensity (max hourly rain per day)
     - Soil moisture (0-7cm, 7-28cm from Open-Meteo hourly API)
     - Snow depth (northern Pakistan glacial melt floods)
     - Upstream precipitation (aggregate precip from upstream)
     - DEM elevation raster (for watershed delineation)

  BI-LSTM ARCHITECTURE:
  ─────────────────────
  Input sequence:  30 days × ~25 features per city
  Features:        precip, rain, snow, temp_max, temp_min, wind,
                   ET0, rolling sums, API, discharge, monsoon...
  Model:           Bi-LSTM(128) → Dropout(0.3) → Bi-LSTM(64)
                   → Dense(32, relu) → Dense(1, sigmoid)
  Output:          flood probability (0-1) or discharge regression

  PAKISTAN-SPECIFIC NOTES:
  ───────────────────────
  • Monsoon (Jun-Sep) causes ~80% of annual flooding
  • 2010 mega-flood: 20M affected, Indus basin
  • 2022 catastrophic flood: 33M affected, national scale
  • Northern areas: glacial lake outburst floods (GLOF)
  • Sindh: riverine + coastal + urban flooding
  • Punjab: Indus + tributary overflow flooding
  """)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("  " + "=" * 62)
    print("  PAKISTAN FLOOD PREDICTION — DATA COLLECTION PIPELINE")
    print(f"  Cities: {len(PAKISTAN_CITIES)} | Weather: {WEATHER_START_YEAR}-{WEATHER_END_YEAR} "
          f"| Flood: {FLOOD_START_YEAR}-{FLOOD_END_YEAR}")
    print("  API: Open-Meteo Free Tier (non-commercial)")
    print("  Tracker: " + str(TRACKER_FILE))
    print("  " + "=" * 62)

    # Load persistent tracker
    tracker = APICallTracker(TRACKER_FILE)
    tracker.print_status()

    # Estimate total work
    total_weather = len(PAKISTAN_CITIES) * (WEATHER_END_YEAR - WEATHER_START_YEAR + 1)
    total_flood = len(PAKISTAN_CITIES) * (FLOOD_END_YEAR - FLOOD_START_YEAR + 1)

    weather_done = sum(1 for t in tracker.data["completed_tasks"] if t.startswith("weather_"))
    flood_done = sum(1 for t in tracker.data["completed_tasks"] if t.startswith("flood_"))

    # Rough estimate of API calls needed
    weather_est = (total_weather - weather_done) * 28.7  # ~28.7 est calls per city-year
    flood_est = (total_flood - flood_done) * 26.1         # ~26.1 est calls per city-year
    total_est = weather_est + flood_est
    days_needed = max(1, total_est / (DAILY_LIMIT * SAFETY_FACTOR))

    print(f"  Estimated remaining API calls: ~{total_est:,.0f}")
    print(f"  Estimated days to complete:    ~{days_needed:.1f}")
    print()

    # ── Pre-flight check: test if API is accessible ──
    if total_est > 0 and ENABLE_PREFLIGHT_CHECK:
        print("  Pre-flight API check...", end=" ", flush=True)
        try:
            # Respect local limits before probing the API.
            tracker.wait_if_needed(1.0)

            params = {
                "latitude": 33.68,
                "longitude": 73.05,
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
                "daily": "precipitation_sum",
                "timezone": "Asia/Karachi",
            }

            preflight_ok = False
            for attempt in range(3):
                test_resp = requests.get(WEATHER_API_URL, params=params, timeout=30)
                if test_resp.status_code == 200:
                    print("OK (API accessible)")
                    tracker.record_call(1.0, "preflight", "preflight_test")
                    preflight_ok = True
                    break

                if test_resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    if attempt < 2:
                        print(f"429 (retrying in {wait}s)...", end=" ", flush=True)
                        time.sleep(wait)
                        continue

                    # Do not abort here; request-level retry logic will keep working.
                    print("429 (continuing; retries enabled per request)")
                    break

                print(f"Unexpected status {test_resp.status_code}")
                break

            if not preflight_ok:
                print("  Proceeding anyway; fetch calls use exponential backoff and limit checks.")
        except Exception as e:
            print(f"FAILED ({e})")
            print("  Check your internet connection and try again.")
            sys.exit(1)
        print()

    # ── Phase 1: Weather Data ──
    print("  " + "-" * 62)
    print("  PHASE 1: Daily Weather Data")
    print("  " + "-" * 62)
    run_weather_collection(tracker)

    # ── Phase 2: Flood/Discharge Data ──
    print()
    print("  " + "-" * 62)
    print("  PHASE 2: River Discharge Data (GloFAS)")
    print("  " + "-" * 62)
    run_flood_collection(tracker)

    # ── Phase 3: Summaries & Features ──
    weather_done = sum(1 for t in tracker.data["completed_tasks"] if t.startswith("weather_"))

    print()
    print("  " + "-" * 62)
    print("  PHASE 3: Summaries & Feature Engineering")
    print("  " + "-" * 62)

    if weather_done >= total_weather:
        compute_summaries()
        merge_and_engineer_features()
        print_flood_prediction_guide()
    elif weather_done > 0:
        print(f"  Weather: {weather_done}/{total_weather} complete.")
        print(f"  Generating partial summaries...")
        compute_summaries()
        merge_and_engineer_features()
        print(f"\n  Re-run tomorrow to fetch remaining data.")
    else:
        print("  No data fetched yet. Check internet and try again.")

    # Final status
    tracker.print_status()

    print("  " + "=" * 62)
    print("  Output directory:", DATA_DIR)
    print("  Tracker file:    ", TRACKER_FILE)
    print("  " + "=" * 62)
    print()


if __name__ == "__main__":
    main()
