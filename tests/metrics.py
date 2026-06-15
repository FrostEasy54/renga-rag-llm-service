from typing import Any


def compute_completeness(
    tasks: list[dict],
    mandatory_stages: list[dict],
) -> float:
    if not tasks or not mandatory_stages:
        return 0.0

    task_names_lower = [t.get("task_name", "").lower() for t in tasks]
    present = 0
    for stage in mandatory_stages:
        keywords = [kw.lower() for kw in stage["keywords"]]
        if any(kw in name for name in task_names_lower for kw in keywords):
            present += 1

    return present / len(mandatory_stages)


def _compute_start_times(tasks: list[dict]) -> dict[str, int]:
    task_by_name = {t.get("task_name", ""): t for t in tasks}
    start_times: dict[str, int] = {}

    def visit(name: str, visiting: set[str]) -> int:
        if name in start_times:
            return start_times[name]
        if name in visiting or name not in task_by_name:
            return 0

        visiting.add(name)
        task = task_by_name[name]
        deps = task.get("depends_on") or []

        max_end = 0
        for dep_name in deps:
            if dep_name in task_by_name:
                dep_start = visit(dep_name, visiting)
                dep_duration = task_by_name[dep_name].get("duration_days", 0)
                max_end = max(max_end, dep_start + dep_duration)

        visiting.remove(name)
        start_times[name] = max_end
        return max_end

    for task in tasks:
        visit(task.get("task_name", ""), set())

    return start_times


def compute_sequence(
    tasks: list[dict],
    precedence_pairs: list[dict],
    mandatory_stages: list[dict],
) -> float:
    if not tasks:
        return 0.0

    start_times = _compute_start_times(tasks)
    stage_keywords = {s["id"]: [kw.lower() for kw in s["keywords"]] for s in mandatory_stages}

    def earliest_start_for_stage(stage_id: str) -> int | None:
        if stage_id not in stage_keywords:
            return None
        keywords = stage_keywords[stage_id]
        matching_starts = [
            start_times.get(t.get("task_name", ""), 0)
            for t in tasks
            if any(kw in t.get("task_name", "").lower() for kw in keywords)
        ]
        return min(matching_starts) if matching_starts else None

    correct = 0
    total = 0
    for pair in precedence_pairs:
        s_before = earliest_start_for_stage(pair["before"])
        s_after = earliest_start_for_stage(pair["after"])
        if s_before is None or s_after is None:
            continue
        total += 1
        if s_before <= s_after:
            correct += 1

    return correct / total if total > 0 else 1.0


def compute_duration_realism(t_generated: int, t_norm: int) -> float:
    if t_norm <= 0:
        return 0.0
    delta = abs(t_generated - t_norm) / t_norm
    return max(0.0, 1.0 - min(1.0, delta))


def compute_composite(c: float, s: float, d: float) -> tuple[float, float]:
    q_arith = (c + s + d) / 3.0
    q_geo = (c * s * d) ** (1.0 / 3.0) if (c > 0 and s > 0 and d > 0) else 0.0
    return q_arith, q_geo


def compute_all_metrics(
    response: dict[str, Any],
    building: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, float]:
    tasks = response.get("tasks") or []
    t_gen = response.get("total_duration_days") or 0
    t_norm = building.get("t_norm_days", 0)

    c = compute_completeness(tasks, manifest["mandatory_stages"])
    s = compute_sequence(tasks, manifest["precedence_pairs"], manifest["mandatory_stages"])
    d = compute_duration_realism(t_gen, t_norm)
    q, q_geo = compute_composite(c, s, d)

    return {
        "C": c,
        "S": s,
        "D": d,
        "Q": q,
        "Q_geo": q_geo,
        "T_generated_days": t_gen,
        "T_norm_days": t_norm,
    }