import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_data(n=200, seed=42):
    """Generate a small synthetic dataset for demonstration.
    Columns: timestamp, location_id, lat, lon, demand, temp_c, humidity, inventory_level, distance_km, is_perishable, spoilage
    """
    rng = np.random.RandomState(seed)
    base_time = datetime.now()
    rows = []
    for i in range(n):
        ts = base_time + timedelta(hours=int(i / 4))
        loc = int(rng.randint(1, 20))
        lat = 40.0 + rng.randn() * 0.1
        lon = -73.9 + rng.randn() * 0.1
        demand = max(0, int(rng.poisson(20) + rng.randint(-5, 5)))
        temp_c = 2.0 + rng.randn() * 5.0 + (0.5 if rng.rand() < 0.2 else 0)
        humidity = 50 + rng.randn() * 10
        inventory = max(0, int(rng.randint(0, 100)))
        distance_km = abs(rng.randn() * 10 + 20)
        is_perishable = int(rng.rand() < 0.6)
        # spoilage probability increases with temp and inventory age (simulated)
        spoil_prob = 0.01 * max(0, temp_c - 2) + 0.002 * (inventory) + (0.05 if is_perishable else 0.0)
        spoilage = int(rng.rand() < min(0.9, spoil_prob))
        rows.append({
            'timestamp': ts,
            'location_id': loc,
            'lat': lat,
            'lon': lon,
            'demand': demand,
            'temp_c': round(temp_c, 2),
            'humidity': round(humidity, 1),
            'inventory_level': inventory,
            'distance_km': round(distance_km, 2),
            'is_perishable': is_perishable,
            'spoilage': spoilage,
        })
    df = pd.DataFrame(rows)
    return df


def load_data(csv_path=None):
    """Load dataset from local CSV or generate synthetic if none provided."""
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=['timestamp'])
    return generate_synthetic_data()


def download_kaggle(dataset: str, file: str, dest: str):
    """Download a file from Kaggle dataset (requires KAGGLE_USERNAME/KAGGLE_KEY env vars and kaggle package).
    dataset: e.g. 'zynicide/wine-reviews' (owner/dataset)
    file: filename in dataset
    dest: destination path
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError("kaggle package required to download datasets; pip install kaggle")
    api = KaggleApi()
    api.authenticate()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    api.dataset_download_file(dataset, file, path=os.path.dirname(dest), unzip=True)
    if os.path.exists(dest):
        return dest
    # try to find downloaded file
    files = [f for f in os.listdir(os.path.dirname(dest)) if f.endswith('.csv')]
    if files:
        return os.path.join(os.path.dirname(dest), files[0])
    return None
