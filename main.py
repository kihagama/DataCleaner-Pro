"""
DataCleaner Pro — CLI
=====================
Usage examples:
  python main.py --input data/sales.csv --output out/clean.csv
  python main.py --input data/orders.json --output out/orders.csv --outlier iqr cap
  python main.py --sql "sqlite:///db.sqlite" --query "SELECT * FROM orders" --output out/q.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── local imports ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from cleaner import (
    build_quality_report,
    detect_anomalies_isolation_forest,
    handle_outliers,
    impute_missing,
    fix_types,
    remove_duplicates,
    suggest_cleaning_strategies,
)
from loader import load_auto, save_dataframe
from reporter import generate_html_report
from visualizer import (
    plot_bar_static,
    plot_boxplot_static,
    plot_correlation_heatmap_static,
    plot_histogram_static,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("datacleaner.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="DataCleaner Pro CLI")
    p.add_argument("--input",   "-i",  help="Path to input file (CSV/Excel/JSON)")
    p.add_argument("--output",  "-o",  required=True, help="Path for cleaned output file")
    p.add_argument("--sql",     help="SQLAlchemy connection string")
    p.add_argument("--query",   help="SQL query or table name (use with --sql)")
    p.add_argument("--impute",  default="auto",
                   choices=["auto", "mean", "median", "mode", "skip"],
                   help="Imputation strategy")
    p.add_argument("--no-fix-types",  action="store_true", help="Skip type correction")
    p.add_argument("--no-dedup",      action="store_true", help="Skip duplicate removal")
    p.add_argument("--outlier",       nargs=2,
                   metavar=("METHOD", "ACTION"),
                   help="e.g. --outlier iqr cap  or  --outlier zscore remove")
    p.add_argument("--anomaly-detection", action="store_true",
                   help="Run Isolation Forest anomaly detection")
    p.add_argument("--report", "-r",  help="Path to save HTML quality report")
    p.add_argument("--charts",        help="Directory to save static chart images")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DataCleaner Pro — starting pipeline")
    if args.sql:
        if not args.query:
            logger.error("--query is required when using --sql")
            sys.exit(1)
        df = load_auto("", connection_string=args.sql, sql_query=args.query)
    elif args.input:
        df = load_auto(args.input)
    else:
        logger.error("Provide --input or --sql")
        sys.exit(1)

    logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Pre-cleaning report ───────────────────────────────
    raw_report = build_quality_report(df)
    logger.info(f"Missing: {raw_report['total_missing']:,}  |  "
                f"Duplicates: {raw_report['duplicate_rows']:,}")

    # ── Clean ─────────────────────────────────────────────
    if not args.no_fix_types:
        df = fix_types(df)

    if args.impute != "skip":
        df = impute_missing(df, strategy=args.impute)

    if not args.no_dedup:
        df = remove_duplicates(df)

    if args.outlier:
        method, action = args.outlier
        df = handle_outliers(df, method=method, action=action)

    if args.anomaly_detection:
        mask  = detect_anomalies_isolation_forest(df)
        n_anom = int(mask.sum())
        logger.info(f"Isolation Forest: {n_anom} anomalies detected "
                    f"({n_anom/len(df)*100:.1f}%)")
        anom_path = str(Path(args.output).parent / "anomalies.csv")
        df[mask].to_csv(anom_path, index=False)
        logger.info(f"Anomalous rows saved to {anom_path}")

    # ── Post-cleaning report ──────────────────────────────
    clean_report = build_quality_report(df)
    suggestions  = suggest_cleaning_strategies(clean_report)

    logger.info("=== Cleaning Suggestions ===")
    for tip in suggestions:
        logger.info(f"  {tip}")

    # ── Save cleaned data ─────────────────────────────────
    save_dataframe(df, args.output)
    logger.info(f"Cleaned data saved → {args.output}")

    # ── Static charts ─────────────────────────────────────
    chart_paths = []
    if args.charts:
        os.makedirs(args.charts, exist_ok=True)
        import numpy as np
        for col in df.select_dtypes(include=[np.number]).columns[:8]:
            try:
                chart_paths.append(plot_histogram_static(df, col, args.charts))
                chart_paths.append(plot_boxplot_static(df, col, args.charts))
            except Exception as e:
                logger.warning(f"Chart error for '{col}': {e}")
        for col in df.select_dtypes(exclude=[np.number]).columns[:4]:
            try:
                chart_paths.append(plot_bar_static(df, col, args.charts))
            except Exception as e:
                logger.warning(f"Bar chart error for '{col}': {e}")
        hmap = plot_correlation_heatmap_static(df, args.charts)
        if hmap:
            chart_paths.append(hmap)
        logger.info(f"Charts saved to {args.charts}/")

    # ── HTML Report ───────────────────────────────────────
    if args.report:
        generate_html_report(
            clean_report,
            suggestions,
            chart_paths=chart_paths,
            output_path=args.report,
        )
        logger.info(f"HTML report → {args.report}")

    logger.info("Pipeline complete ✅")


if __name__ == "__main__":
    main()
