"""
Data Loader Module
==================
Supports loading from:
  - CSV  (.csv)
  - Excel (.xlsx / .xls)
  - JSON  (.json)
  - SQL   (via SQLAlchemy connection string)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
try:
    from sqlalchemy import create_engine, text
    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False

logger = logging.getLogger(__name__)


def _ext(path: str) -> str:
    return Path(path).suffix.lower()


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file."""
    logger.info(f"Loading CSV: {path}")
    return pd.read_csv(path, **kwargs)


def load_excel(path: str, sheet_name=0, **kwargs) -> pd.DataFrame:
    """Load an Excel file (first sheet by default)."""
    logger.info(f"Loading Excel: {path}, sheet={sheet_name}")
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


def load_json(path: str, **kwargs) -> pd.DataFrame:
    """
    Load JSON.  Handles both record-oriented lists and nested dicts
    by normalising with pd.json_normalize where needed.
    """
    logger.info(f"Loading JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return pd.json_normalize(data)
    elif isinstance(data, dict):
        # Try direct DataFrame construction; fall back to normalize
        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.json_normalize(data)
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")


def load_sql(
    connection_string: str,
    query_or_table: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Load data from a SQL database.

    Parameters
    ----------
    connection_string : str
        SQLAlchemy-compatible URL, e.g.
        'postgresql+psycopg2://user:pass@host:5432/db'
        'sqlite:///myfile.db'
    query_or_table : str
        Either a full SQL SELECT statement or a bare table name.
    """
    if not _HAS_SQLALCHEMY:
        raise ImportError("sqlalchemy is required for SQL connections: pip install sqlalchemy")
    logger.info(f"Connecting to SQL: {connection_string[:40]}…")
    engine = create_engine(connection_string)
    sql = query_or_table.strip()
    if not sql.lower().startswith("select"):
        sql = f"SELECT * FROM {sql}"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, **kwargs)
    logger.info(f"Fetched {len(df)} rows from SQL")
    return df


def load_auto(
    source: str,
    *,
    sheet_name=0,
    connection_string: Optional[str] = None,
    sql_query: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Auto-detect file type and load the DataFrame.

    For SQL pass connection_string + sql_query instead of a file path.
    """
    if connection_string:
        if not sql_query:
            raise ValueError("sql_query is required when connection_string is provided.")
        return load_sql(connection_string, sql_query, **kwargs)

    ext = _ext(source)
    if ext == ".csv":
        return load_csv(source, **kwargs)
    elif ext in (".xlsx", ".xls"):
        return load_excel(source, sheet_name=sheet_name, **kwargs)
    elif ext == ".json":
        return load_json(source, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Supported: .csv, .xlsx, .xls, .json, or SQL via connection_string."
        )


# ──────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV, Excel, or JSON based on file extension."""
    ext = _ext(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    elif ext == ".json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported save format '{ext}'.")
    logger.info(f"Saved cleaned data to {path}")
