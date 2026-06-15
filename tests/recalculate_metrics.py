"""
Пересчёт метрик из сохранённых responses/*.json без повторного вызова API.
Запуск: python recalculate_metrics.py
"""

import json
import pandas as pd
from pathlib import Path
from metrics import compute_all_metrics
from run_experiment import aggregate_per_building, compute_summary

BASE_DIR      = Path(__file__).parent
MANIFEST_PATH = BASE_DIR / "manifest.json"
RESPONSES_DIR = BASE_DIR / "responses"
RESULTS_DIR   = BASE_DIR / "results"
TESTS_CSV     = RESULTS_DIR / "tests.csv"

manifest  = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
buildings = {b["id"]: b for b in manifest["buildings"]}
tests_df  = pd.read_csv(TESTS_CSV)

new_rows = []
for _, row in tests_df.iterrows():
    response_path = RESPONSES_DIR / f"{row['test_id']}.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    building = buildings[row["building_id"]]

    metrics = compute_all_metrics(response, building, manifest)

    new_rows.append({**row.to_dict(), **metrics})
    print(f"{row['test_id']}  C={metrics['C']:.2f}  S={metrics['S']:.2f}"
          f"  D={metrics['D']:.2f}  Q={metrics['Q']:.2f}")

updated_tests_df = pd.DataFrame(new_rows)
updated_tests_df.to_csv(RESULTS_DIR / "tests.csv", index=False, encoding="utf-8")

present_ids = set(updated_tests_df["building_id"].unique())
buildings_for_agg = [b for b in manifest["buildings"] if b["id"] in present_ids]

per_building_df = aggregate_per_building(updated_tests_df, buildings_for_agg)
per_building_df.to_csv(RESULTS_DIR / "per_building.csv", index=False, encoding="utf-8")

summary_df = compute_summary(per_building_df)
summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False, encoding="utf-8")

print("\nDone. Files updated:")
print("  results/tests.csv")
print("  results/per_building.csv")
print("  results/summary.csv")