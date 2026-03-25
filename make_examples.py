"""
Generate example datasets for testing DataCleaner Pro.
Run:  python make_examples.py
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


# ── 1. Sales dataset (CSV) — missing values + outliers ────
def make_sales():
    n = 500
    df = pd.DataFrame({
        "order_id":    range(1, n + 1),
        "customer":    rng.choice(["Alice", "Bob", "Carol", "Dave", "Eve"], n),
        "product":     rng.choice(["Widget A", "Widget B", "Gadget X", "Gadget Y"], n),
        "region":      rng.choice(["North", "South", "East", "West"], n),
        "quantity":    rng.integers(1, 50, n).astype(float),
        "unit_price":  rng.uniform(5, 200, n).round(2),
        "discount":    rng.uniform(0, 0.3, n).round(2),
        "date":        pd.date_range("2023-01-01", periods=n, freq="12h").strftime("%Y-%m-%d"),
    })
    df["revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount"])).round(2)

    # Introduce issues
    mask_miss = rng.choice([True, False], n, p=[0.08, 0.92])
    df.loc[mask_miss, "quantity"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "unit_price"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "region"]     = None

    # Outliers
    df.loc[rng.choice(n, 5, replace=False), "unit_price"] = rng.uniform(800, 2000, 5).round(2)

    # Duplicates
    dupes = df.sample(15, random_state=42)
    df = pd.concat([df, dupes], ignore_index=True)

    path = OUT / "sales.csv"
    df.to_csv(path, index=False)
    print(f"✅ {path}  ({len(df)} rows)")
    return path


# ── 2. Employee dataset (Excel) — type mismatches ─────────
def make_employees():
    n = 300
    df = pd.DataFrame({
        "emp_id":       range(1001, 1001 + n),
        "name":         [f"Employee_{i}" for i in range(n)],
        "department":   rng.choice(["Engineering", "Sales", "HR", "Finance", "Marketing"], n),
        "salary":       rng.integers(30_000, 120_000, n).astype(str),   # stored as string
        "age":          rng.integers(22, 65, n),
        "years_exp":    rng.integers(0, 30, n).astype(float),
        "performance":  rng.choice(["A", "B", "C", "D"], n),
        "hire_date":    pd.date_range("2010-01-01", periods=n, freq="30d").strftime("%d/%m/%Y"),
        "active":       rng.choice(["Yes", "No", "yes", "no", "YES"], n),
    })

    # Missing values
    df.loc[rng.choice(n, 25, replace=False), "salary"]      = None
    df.loc[rng.choice(n, 15, replace=False), "years_exp"]   = np.nan
    df.loc[rng.choice(n, 10, replace=False), "performance"] = None

    path = OUT / "employees.xlsx"
    df.to_excel(path, index=False)
    print(f"✅ {path}  ({len(df)} rows)")
    return path


# ── 3. IoT sensor data (JSON) — many outliers ─────────────
def make_sensors():
    n = 400
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min")
    records = []
    for i, ts in enumerate(timestamps):
        temp    = 22 + rng.normal(0, 1.5)
        humidity= 55 + rng.normal(0, 5)
        vibration = abs(rng.normal(0.02, 0.005))
        # Inject outliers
        if i % 80 == 0:
            temp      = rng.choice([-50, 110])
            humidity  = rng.choice([-5, 110])
            vibration = rng.uniform(0.5, 2.0)
        records.append({
            "timestamp":   ts.isoformat(),
            "sensor_id":   f"S{rng.integers(1, 6):02d}",
            "temperature": round(float(temp), 2),
            "humidity":    round(float(humidity), 2),
            "vibration":   round(float(vibration), 4),
            "status":      rng.choice(["ok", "ok", "ok", "warning", None], p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        })
    path = OUT / "sensors.json"
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"✅ {path}  ({len(records)} rows)")
    return path


# ── 4. Tiny SQLite database ────────────────────────────────
def make_sqlite():
    try:
        from sqlalchemy import create_engine, text
        import sqlite3

        db_path = OUT / "store.db"
        engine  = create_engine(f"sqlite:///{db_path}")
        n = 200
        df = pd.DataFrame({
            "product_id":   range(1, n + 1),
            "product_name": [f"Product_{i}" for i in range(n)],
            "category":     rng.choice(["Electronics", "Clothing", "Food", "Books"], n),
            "price":        rng.uniform(1, 500, n).round(2),
            "stock":        rng.integers(0, 1000, n),
            "rating":       rng.uniform(1, 5, n).round(1),
        })
        df.loc[rng.choice(n, 15, replace=False), "price"]  = np.nan
        df.loc[rng.choice(n, 10, replace=False), "rating"] = np.nan
        with engine.connect() as conn:
            df.to_sql("products", conn, if_exists="replace", index=False)
            conn.commit()
        print(f"✅ {db_path}  (table: products, {n} rows)")
        return db_path
    except Exception as e:
        print(f"⚠️  SQLite creation failed: {e}")
        return None


if __name__ == "__main__":
    make_sales()
    make_employees()
    make_sensors()
    make_sqlite()
    print("\n🎉 All example datasets generated in ./data/")
