import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon

from metrics import compute_all_metrics

BASE_URL = "http://localhost:8000"
ENDPOINT = "/generate-schedule"
RUNS_PER_CONFIG = 3
REQUEST_TIMEOUT_SEC = 600  

CONFIGS: tuple[tuple[str, str], ...] = (
    ("BARE", "bare"),
    ("STRUCTURED", "structured"),
    ("RAG", "rag"),
)

DEFAULT_MODELS: tuple[str, ...] | None = None

TESTS_DIR = Path(__file__).parent
BUILDINGS_DIR = TESTS_DIR / "buildings"
MANIFEST_PATH = TESTS_DIR / "manifest.json"
RESPONSES_DIR = TESTS_DIR / "responses"
RESULTS_DIR = TESTS_DIR / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")


def call_api(body: dict, mode: str) -> tuple[dict, float, str | None]:
    url = f"{BASE_URL}{ENDPOINT}?mode={mode}"
    t_start = time.time()
    try:
        resp = requests.post(url, json=body, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.json(), time.time() - t_start, None
    except requests.exceptions.HTTPError as exc:
        body_excerpt = exc.response.text[:300] if exc.response is not None else ""
        return ({"error": f"HTTP {exc.response.status_code}: {body_excerpt}"},
                time.time() - t_start, str(exc))
    except Exception as exc:
        return {"error": str(exc)}, time.time() - t_start, str(exc)


def health_check() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        logger.info("API доступен: %s", body)
        return body.get("ollama_reachable", False)
    except Exception as exc:
        logger.error("API недоступен на %s/health: %s", BASE_URL, exc)
        return False


def run_experiment(
    manifest: dict,
    selected_buildings: list[dict],
    runs_per_config: int,
    preserved_rows: list[dict] | None = None,
    test_idx_start: int = 0,
) -> pd.DataFrame:
    RESPONSES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    rows: list[dict] = []
    total = len(selected_buildings) * len(CONFIGS) * runs_per_config
    test_idx = test_idx_start
    t_start = time.time()

    for building in selected_buildings:
        body_path = TESTS_DIR / building["filename"]
        body = json.loads(body_path.read_text(encoding="utf-8"))

        for config_name, mode_param in CONFIGS:
            for run in range(1, runs_per_config + 1):
                test_idx += 1
                test_id = f"t{test_idx:03d}"

                logger.info(
                    "[%d/%d] %s — %s %s (run %d/%d)",
                    test_idx - test_idx_start, total, test_id,
                    building["id"], config_name, run, runs_per_config,
                )

                response, latency, error = call_api(body, mode_param)

                response_path = RESPONSES_DIR / f"{test_id}.json"
                response_path.write_text(
                    json.dumps(response, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                if error:
                    logger.warning("  ошибка: %s", error)
                    metrics = {
                        "C": 0.0, "S": 0.0, "D": 0.0, "Q": 0.0, "Q_geo": 0.0,
                        "T_generated_days": 0, "T_norm_days": building["t_norm_days"],
                    }
                else:
                    try:
                        metrics = compute_all_metrics(response, building, manifest)
                    except Exception as exc:
                        logger.warning("  не удалось посчитать метрики: %s", exc)
                        metrics = {
                            "C": 0.0, "S": 0.0, "D": 0.0, "Q": 0.0, "Q_geo": 0.0,
                            "T_generated_days": 0, "T_norm_days": building["t_norm_days"],
                        }
                    logger.info(
                        "  C=%.2f S=%.2f D=%.2f Q=%.2f (T_gen=%d, T_norm=%d, latency=%.1fs)",
                        metrics["C"], metrics["S"], metrics["D"], metrics["Q"],
                        metrics["T_generated_days"], metrics["T_norm_days"], latency,
                    )

                rows.append({
                    "test_id": test_id,
                    "building_id": building["id"],
                    "group": building["group"],
                    "config": config_name,
                    "mode": mode_param,
                    "run": run,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "latency_sec": round(latency, 2),
                    "task_count": len(response.get("tasks") or []),
                    "T_generated_days": metrics["T_generated_days"],
                    "T_norm_days": metrics["T_norm_days"],
                    "C": round(metrics["C"], 3),
                    "S": round(metrics["S"], 3),
                    "D": round(metrics["D"], 3),
                    "Q": round(metrics["Q"], 3),
                    "Q_geo": round(metrics["Q_geo"], 3),
                    "rag_context_used": response.get("rag_context_used"),
                    "response_file": str(response_path.relative_to(TESTS_DIR)),
                    "error": error or "",
                })

                combined = (preserved_rows or []) + rows
                pd.DataFrame(combined).to_csv(
                    RESULTS_DIR / "tests.csv", index=False, encoding="utf-8",
                )

    elapsed_min = (time.time() - t_start) / 60
    logger.info("Все запросы выполнены за %.1f мин.", elapsed_min)
    return pd.DataFrame((preserved_rows or []) + rows)

def aggregate_per_building(
    tests_df: pd.DataFrame,
    selected_buildings: list[dict],
) -> pd.DataFrame:
    """Агрегация до уровня (building × config): среднее по прогонам."""
    rows: list[dict] = []
    config_names = [c[0] for c in CONFIGS]

    for building in selected_buildings:
        bid = building["id"]
        rec: dict = {
            "building_id": bid,
            "name": building["description"],
            "group": building["group"],
            "floors": building["floors"],
            "total_area": building["total_area"],
            "T_norm_days": building["t_norm_days"],
        }

        for config_name in config_names:
            subset = tests_df[
                (tests_df["building_id"] == bid) & (tests_df["config"] == config_name)
            ]
            if subset.empty:
                continue
            suffix = config_name.lower()
            rec[f"C_{suffix}"] = round(subset["C"].mean(), 3)
            rec[f"S_{suffix}"] = round(subset["S"].mean(), 3)
            rec[f"D_{suffix}"] = round(subset["D"].mean(), 3)
            rec[f"Q_{suffix}"] = round(subset["Q"].mean(), 3)
            rec[f"Q_geo_{suffix}"] = round(subset["Q_geo"].mean(), 3)
            rec[f"T_gen_{suffix}"] = int(round(subset["T_generated_days"].mean(), 0))

        rows.append(rec)

    return pd.DataFrame(rows)


def _safe_wilcoxon(a, b) -> float:
    try:
        _, p = wilcoxon(a, b, alternative="greater", zero_method="zsplit", method="exact")
        return float(p)
    except ValueError:
        return float("nan")


def _safe_ttest(a, b) -> float:
    """Парный t-критерий, односторонняя альтернатива a > b."""
    try:
        _, p = ttest_rel(a, b, alternative="greater")
        return float(p)
    except Exception:
        return float("nan")


def _cohen_dz(a, b) -> float:
    diffs = a - b
    std = diffs.std(ddof=1)
    return float(diffs.mean() / std) if std > 0 else float("nan")


def compute_summary(per_building_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in ("C", "S", "D", "Q"):
        col_b = f"{metric}_bare"
        col_s = f"{metric}_structured"
        col_r = f"{metric}_rag"

        if not all(c in per_building_df.columns for c in (col_b, col_s, col_r)):
            logger.warning("Колонки для метрики %s отсутствуют — пропуск", metric)
            continue

        joined = per_building_df[[col_b, col_s, col_r]].dropna()
        n = len(joined)
        if n < 2:
            logger.warning("Недостаточно данных для статистики метрики %s (N=%d)", metric, n)
            continue

        bare = joined[col_b].to_numpy()
        structured = joined[col_s].to_numpy()
        rag = joined[col_r].to_numpy()

        try:
            chi2, fp = friedmanchisquare(bare, structured, rag)
            friedman_chi2 = float(chi2)
            friedman_p = float(fp)
        except Exception:
            friedman_chi2 = float("nan")
            friedman_p = float("nan")

        p_s_vs_b = _safe_wilcoxon(structured, bare)
        p_r_vs_s = _safe_wilcoxon(rag, structured)
        p_r_vs_b = _safe_wilcoxon(rag, bare)

        t_s_vs_b = _safe_ttest(structured, bare)
        t_r_vs_s = _safe_ttest(rag, structured)
        t_r_vs_b = _safe_ttest(rag, bare)

        dz_s_vs_b = _cohen_dz(structured, bare)
        dz_r_vs_s = _cohen_dz(rag, structured)
        dz_r_vs_b = _cohen_dz(rag, bare)

        rows.append({
            "metric": metric,
            "N": n,

            "mean_BARE": round(float(bare.mean()), 3),
            "mean_STRUCTURED": round(float(structured.mean()), 3),
            "mean_RAG": round(float(rag.mean()), 3),
            "std_BARE": round(float(bare.std(ddof=1)), 3) if n > 1 else float("nan"),
            "std_STRUCTURED": round(float(structured.std(ddof=1)), 3) if n > 1 else float("nan"),
            "std_RAG": round(float(rag.std(ddof=1)), 3) if n > 1 else float("nan"),

            "delta_struct_vs_bare": round(float(structured.mean() - bare.mean()), 3),
            "delta_rag_vs_struct": round(float(rag.mean() - structured.mean()), 3),
            "delta_rag_vs_bare": round(float(rag.mean() - bare.mean()), 3),

            "friedman_chi2": round(friedman_chi2, 3) if friedman_chi2 == friedman_chi2 else float("nan"),
            "friedman_p": round(friedman_p, 4) if friedman_p == friedman_p else float("nan"),

            "wilcoxon_p_struct_vs_bare": round(p_s_vs_b, 4) if p_s_vs_b == p_s_vs_b else float("nan"),
            "wilcoxon_p_rag_vs_struct": round(p_r_vs_s, 4) if p_r_vs_s == p_r_vs_s else float("nan"),
            "wilcoxon_p_rag_vs_bare":   round(p_r_vs_b, 4) if p_r_vs_b == p_r_vs_b else float("nan"),

            "t_test_p_struct_vs_bare": round(t_s_vs_b, 4) if t_s_vs_b == t_s_vs_b else float("nan"),
            "t_test_p_rag_vs_struct": round(t_r_vs_s, 4) if t_r_vs_s == t_r_vs_s else float("nan"),
            "t_test_p_rag_vs_bare":   round(t_r_vs_b, 4) if t_r_vs_b == t_r_vs_b else float("nan"),

            "cohen_dz_struct_vs_bare": round(dz_s_vs_b, 3) if dz_s_vs_b == dz_s_vs_b else float("nan"),
            "cohen_dz_rag_vs_struct": round(dz_r_vs_s, 3) if dz_r_vs_s == dz_r_vs_s else float("nan"),
            "cohen_dz_rag_vs_bare":   round(dz_r_vs_b, 3) if dz_r_vs_b == dz_r_vs_b else float("nan"),

            "friedman_significant_005": bool(friedman_p < 0.05) if friedman_p == friedman_p else False,
        })

    return pd.DataFrame(rows)



def _select_buildings(manifest: dict, ids_csv: str | None) -> list[dict]:
    by_id = {b["id"]: b for b in manifest["buildings"]}

    if ids_csv:
        wanted = [s.strip() for s in ids_csv.split(",") if s.strip()]
    elif DEFAULT_MODELS:
        wanted = list(DEFAULT_MODELS)
    else:
        return list(manifest["buildings"])

    missing = [w for w in wanted if w not in by_id]
    if missing:
        logger.error("Не найдены в manifest: %s", ", ".join(missing))
        return []
    return [by_id[w] for w in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description="Трёхрукий эксперимент RAG vs no-RAG vs bare")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-тест: первая модель из выборки × 3 конфига × 1 прогон.",
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated building IDs (default: все модели из manifest.json)",
    )
    parser.add_argument(
        "--resume-models", default=None,
        help=(
            "Перезапустить эксперимент для указанных моделей: считывает существующий "
            "tests.csv, удаляет из него строки этих моделей, продолжает нумерацию "
            "test_id и дописывает новые результаты. Используется при сбоях/таймаутах. "
            "Пример: --resume-models m12,m13,m14,m15"
        ),
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        logger.error("Не найден %s", MANIFEST_PATH)
        return

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    preserved_rows: list[dict] | None = None
    test_idx_start: int = 0

    if args.resume_models:
        if args.smoke or args.models:
            logger.error("--resume-models несовместим с --smoke или --models.")
            return

        tests_csv = RESULTS_DIR / "tests.csv"
        if not tests_csv.exists():
            logger.error("--resume-models требует существующий %s", tests_csv)
            return

        resume_ids = [s.strip() for s in args.resume_models.split(",") if s.strip()]
        by_id = {b["id"]: b for b in manifest["buildings"]}
        missing = [r for r in resume_ids if r not in by_id]
        if missing:
            logger.error("Не найдены в manifest: %s", ", ".join(missing))
            return

        existing_df = pd.read_csv(tests_csv)
        preserved_df = existing_df[~existing_df["building_id"].isin(resume_ids)]
        preserved_rows = preserved_df.to_dict("records")
        n_removed = len(existing_df) - len(preserved_rows)

        if preserved_rows:
            test_idx_start = int(preserved_df["test_id"].str[1:].astype(int).max())

        selected = [by_id[r] for r in resume_ids]
        runs = RUNS_PER_CONFIG

        logger.info(
            "RESUME: сохраняем %d строк, удаляем %d (модели %s), "
            "продолжаем нумерацию с t%03d.",
            len(preserved_rows), n_removed, ",".join(resume_ids), test_idx_start + 1,
        )
        logger.info("Модели для перезапуска: %s", ", ".join(b["id"] for b in selected))
        logger.info("Дозаписываем %d запросов.", len(selected) * len(CONFIGS) * runs)
    else:
        selected = _select_buildings(manifest, args.models)
        if not selected:
            return

        if args.smoke:
            selected = selected[:1]
            runs = 1
            logger.info(
                "SMOKE-режим: 1 модель × %d конфига × 1 прогон = %d запросов",
                len(CONFIGS), len(CONFIGS),
            )
        else:
            runs = RUNS_PER_CONFIG
            total = len(selected) * len(CONFIGS) * runs
            logger.info(
                "Полный прогон: %d моделей × %d конфига × %d прогона = %d запросов",
                len(selected), len(CONFIGS), runs, total,
            )
            logger.info("Модели: %s", ", ".join(b["id"] for b in selected))

    if not health_check():
        logger.error("Запустите backend (cd .. && just run) перед прогоном.")
        return

    tests_df = run_experiment(manifest, selected, runs, preserved_rows, test_idx_start)

    present_ids = set(tests_df["building_id"].unique())
    buildings_for_agg = [b for b in manifest["buildings"] if b["id"] in present_ids]
    per_building_df = aggregate_per_building(tests_df, buildings_for_agg)
    per_building_df.to_csv(RESULTS_DIR / "per_building.csv", index=False, encoding="utf-8")
    logger.info("Сохранено: %s (%d строк)",
                (RESULTS_DIR / "per_building.csv").relative_to(TESTS_DIR), len(per_building_df))

    summary_df = compute_summary(per_building_df)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False, encoding="utf-8")
    logger.info("Сохранено: %s", (RESULTS_DIR / "summary.csv").relative_to(TESTS_DIR))

    logger.info("=" * 78)
    logger.info("ИТОГОВАЯ СТАТИСТИКА (одностороннее H₁: справа > слева):")
    for _, row in summary_df.iterrows():
        sig = "ЗНАЧИМО" if row["friedman_significant_005"] else "не значимо"
        logger.info(
            "  %s:  bare=%.3f  struct=%.3f  rag=%.3f  |  "
            "Δ(S-B)=%+.3f  Δ(R-S)=%+.3f  Δ(R-B)=%+.3f  |  friedman_p=%.4f  [%s]",
            row["metric"],
            row["mean_BARE"], row["mean_STRUCTURED"], row["mean_RAG"],
            row["delta_struct_vs_bare"], row["delta_rag_vs_struct"], row["delta_rag_vs_bare"],
            row["friedman_p"], sig,
        )
    logger.info("=" * 78)


if __name__ == "__main__":
    main()