import requests
import time

# Check 2010 (major Pakistan flood year) for key Indus-basin cities
CITIES = [
    ("Sukkur",          27.7052, 68.8574),
    ("Hyderabad",       25.3960, 68.3578),
    ("Dera Ghazi Khan", 30.0489, 70.6455),
    ("Muzaffargarh",    30.0736, 71.1936),
    ("Jacobabad",       28.2769, 68.4514),
    ("Larkana",         27.5570, 68.2028),
    ("Thatta",          24.7461, 67.9236),
    ("Attock",          33.7667, 72.3597),
    ("Peshawar",        34.0151, 71.5249),
    ("Chitral",         35.8518, 71.7864),
    ("Gilgit",          35.9208, 74.3144),
    ("Muzaffarabad",    34.3700, 73.4717),
    ("Lahore",          31.5204, 74.3587),  # reference — known Ravi trickle
]

print(f"Probing 2010 peak flood season (Jul-Aug) — seamless_v4")
print(f"{'City':<22} {'Snapped Lat':>11} {'Snapped Lon':>11} {'Mean m3/s':>10} {'Max m3/s':>10}  Signal")
print("-" * 85)

for city, lat, lon in CITIES:
    r = requests.get("https://flood-api.open-meteo.com/v1/flood", params={
        "latitude": lat, "longitude": lon,
        "start_date": "2010-07-01", "end_date": "2010-08-31",
        "daily": "river_discharge",
        "models": "seamless_v4",
    }, timeout=30)
    d = r.json()
    vals = [v for v in d.get("daily", {}).get("river_discharge", []) if v is not None]
    snap_lat = d.get("latitude", lat)
    snap_lon = d.get("longitude", lon)
    if vals:
        mean_q = sum(vals) / len(vals)
        max_q = max(vals)
        signal = "STRONG" if max_q > 500 else ("OK" if max_q > 50 else "WEAK (small/dry river)")
    else:
        mean_q = max_q = 0
        signal = "NO DATA"
    print(f"{city:<22} {snap_lat:>11.4f} {snap_lon:>11.4f} {mean_q:>10.1f} {max_q:>10.1f}  {signal}")
    time.sleep(0.5)
