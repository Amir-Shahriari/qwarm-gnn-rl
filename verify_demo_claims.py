#!/usr/bin/env python3
"""
verify_demo_claims.py — Read-only claim verification gate for ICDM 2026 demo.
Reads canonical sweep JSON files and checks paper's reported numbers.
Exit code: 0 if all pass, 1 if any fail.
"""

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SWEEP_25 = ROOT / "runs" / "sweep_phase3_final.json"
SWEEP_50_1X = ROOT / "runs" / "sweep_50x50" / "sweep_v1_50x50_1x.json"
SWEEP_50_4X = ROOT / "runs" / "sweep_50x50" / "sweep_v1_50x50_4x.json"
FLEET = ROOT / "runs" / "fleet_1779277545_seed42" / "fleet_results.json"
# Produced by scripts/run_shaping_control.py --train.
SHAPING_CONTROL = ROOT / "runs" / "shaping_control" / "aggregate.json"
# Fresh run's own reach counts (aggregate.json), keyed by lambda_shape —
# checked for exact match so any future rerun drifting silently fails loudly.
EXPECTED_SHAPING_REACH = {0.5: 18, 1.0: 12, 2.0: 16, 4.0: 14, 8.0: 15}
REACHABILITY_AUDIT = ROOT / "runs" / "eval_reachability_audit.json"
ABLATION_PARTIAL = ROOT / "runs" / "demo_source_ablation_partial.json"
ABLATION_LOG = ROOT / "runs" / "demo_source_ablation.log"

# Known-unsolvable 25x25 cell (destination node deactivated by perturbation) —
# the single expected exclusion when checking "reach over solvable cells".
KNOWN_UNSOLVABLE_25X25 = "seed518677876_s1"

_ABLATION_RUN_RE = re.compile(
    r"\[\s*\d+/50\]\s+(classical_only|quantum_only)\s+seed=(\d+)\s+(\S+)->(\S+)"
)
_ABLATION_RESULT_RE = re.compile(r"cost=(\S+)\s+strict=(True|False)")


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def win(rec):
    """warm_strict=True AND (cold_strict=False OR warm_cost < cold_cost)"""
    if not rec.get("warm_strict"):
        return False
    cold_strict = rec.get("cold_strict", False)
    if not cold_strict:
        return True
    # Both strict — compare costs
    warm_c = rec.get("warm_cost") if rec.get("warm_cost") is not None else math.inf
    cold_c = rec.get("cold_cost") if rec.get("cold_cost") is not None else math.inf
    return warm_c < cold_c


def mean_warm_ratio(records):
    """Mean warm_cost_ratio over records where warm_strict=True (reached cells only)."""
    ratios = [r["warm_cost_ratio"] for r in records
              if r.get("warm_strict") and r.get("warm_cost_ratio") is not None]
    if not ratios:
        return float("nan")
    return sum(ratios) / len(ratios)


def median_warm_ratio(records):
    """Median warm_cost_ratio over records where warm_strict=True."""
    ratios = sorted(r["warm_cost_ratio"] for r in records
                    if r.get("warm_strict") and r.get("warm_cost_ratio") is not None)
    if not ratios:
        return float("nan")
    n = len(ratios)
    mid = n // 2
    if n % 2 == 0:
        return (ratios[mid - 1] + ratios[mid]) / 2.0
    return ratios[mid]


def check_count(label, got, expected, tol, total, results):
    if isinstance(tol, (int, float)):
        abs_tol = tol
        passed = abs(got - expected) <= abs_tol
        tol_str = f"±{tol}"
    else:
        passed = False
        tol_str = str(tol)

    mark = "PASS" if passed else "FAIL"
    if passed:
        print(f"  [{mark}] {label}: got={got}/{total}  expected={expected}/{total}")
    else:
        print(f"  [{mark}] {label}: got={got}/{total}  expected={expected}/{total}  (tol={tol_str})")
    if not passed:
        results.append(f"{label}: got={got}, expected={expected}")


def check_ratio(label, got, expected, rel_tol, results):
    """Check a scalar value with relative tolerance."""
    if expected == 0:
        return got == 0
    passed = abs(got - expected) / expected <= rel_tol
    mark = "PASS" if passed else "FAIL"
    pct = rel_tol * 100
    if passed:
        print(f"  [{mark}] {label}: got={got:.3f}x  expected={expected:.3f}x")
    else:
        print(f"  [{mark}] {label}: got={got:.3f}x  expected={expected:.3f}x  (tol={pct:.0f}% rel)")
    if not passed:
        results.append(f"{label}: got={got:.3f}, expected={expected:.3f}")


def check_time(label, got, expected, rel_tol, unit, results):
    """Check a time scalar with relative tolerance."""
    passed = abs(got - expected) / expected <= rel_tol
    mark = "PASS" if passed else "FAIL"
    pct = rel_tol * 100
    if passed:
        print(f"  [{mark}] {label}: got={got:.4f}{unit}  expected={expected:.4f}{unit}")
    else:
        print(f"  [{mark}] {label}: got={got:.4f}{unit}  expected={expected:.4f}{unit}  (tol={pct:.0f}% rel)")
    if not passed:
        results.append(f"{label}: got={got:.4f}, expected={expected:.4f}")


def parse_ablation_log(log_path, results):
    """Parse runs/demo_source_ablation.log into per-run records.

    Full committed log of all 50 ablation training runs (25 classical_only +
    25 quantum_only), each an independently-run cell. Log format, one run =
    two lines:
        [ 12/50] classical_only seed=2024 Node_623->Node_50
                cost=inf strict=False  (185s)

    Fails loudly into `results` (gate FAIL) rather than silently on any parse
    drift: a missing result line after a run header, or a record count other
    than the expected 50, is reported as a failure rather than skipped.
    """
    lines = log_path.read_text().splitlines()
    records = []
    parse_failures = []
    for i, line in enumerate(lines):
        m = _ABLATION_RUN_RE.search(line)
        if not m:
            continue
        arm, seed, src, dst = m.groups()
        result_line = lines[i + 1] if i + 1 < len(lines) else None
        rm = _ABLATION_RESULT_RE.search(result_line) if result_line is not None else None
        if rm is None:
            parse_failures.append(f"no cost=/strict= line found after: {line!r}")
            continue
        records.append({
            "arm": arm, "seed": int(seed), "source": src, "destination": dst,
            "strict": rm.group(2) == "True",
        })

    if parse_failures:
        results.append(
            f"ablation log parse: {len(parse_failures)} run header(s) missing a result line: {parse_failures}"
        )
    if len(records) != 50:
        results.append(f"ablation log parse: expected 50 records, got {len(records)}")

    return records


def join_ablation_log_to_scenarios(log_records, sweep_25_records, results):
    """Attach scenario_id to each log record via (seed, source, destination).

    Fails loudly into `results` on any row that doesn't join against the
    headline 25x25 sweep — a silent None scenario_id would make every
    downstream solvable-cell check silently under-count instead of failing.
    """
    key_to_scenario = {
        (r["seed"], r["source"], r["destination"]): r["scenario_id"]
        for r in sweep_25_records
    }
    unmatched = []
    for r in log_records:
        key = (r["seed"], r["source"], r["destination"])
        scenario_id = key_to_scenario.get(key)
        if scenario_id is None:
            unmatched.append(dict(r))
        r["scenario_id"] = scenario_id

    if unmatched:
        results.append(
            f"ablation log join: {len(unmatched)} row(s) did not match any headline-sweep "
            f"scenario (seed, source, destination): {unmatched}"
        )
    return log_records


def check_ablation_full_reach(log_records, arm, solvable_ids, results):
    """Reach for `arm` over ALL structurally-solvable 25x25 cells, parsed from
    the full 50-run committed log (runs/demo_source_ablation.log) — not the
    9-row runs/demo_source_ablation_partial.json checkpoint, which only ever
    captured a subset because the pipeline's canonical join was aborted
    (traced-vs-published per-cell cost drift exceeded the documented
    reproduction baseline on 14/25 cells).

    Also asserts the single excluded non-solvable cell is exactly the known-
    unsolvable cell (KNOWN_UNSOLVABLE_25X25) — if a DIFFERENT cell turns out
    non-solvable, or more than one does, that's a real discrepancy, not
    something to silently absorb into the denominator.
    """
    arm_rows = [r for r in log_records if r["arm"] == arm]
    solvable_rows = [r for r in arm_rows if r["scenario_id"] in solvable_ids]
    non_solvable_rows = [r for r in arm_rows if r["scenario_id"] not in solvable_ids]

    den = len(solvable_rows)
    num = sum(1 for r in solvable_rows if r["strict"])
    expected_den = len(solvable_ids)

    label = f"ablation {arm} reach / solvable (source: runs/demo_source_ablation.log, full 50-run log)"
    mark = "PASS" if (num == expected_den and den == expected_den) else "FAIL"
    print(f"  [{mark}] {label}: got={num}/{den}  expected={expected_den}/{expected_den}")
    if not (num == expected_den and den == expected_den):
        results.append(f"{label}: got={num}/{den}, expected={expected_den}/{expected_den}")

    excluded_ids = {r["scenario_id"] for r in non_solvable_rows}
    if excluded_ids != {KNOWN_UNSOLVABLE_25X25}:
        results.append(
            f"ablation {arm}: expected the single non-solvable excluded cell to be "
            f"{KNOWN_UNSOLVABLE_25X25!r}, got {excluded_ids!r}"
        )

    return num, den


def check_ablation_subset_reach(rows, log_records, arm, expected_n, results):
    """Secondary consistency check: the 9-row partial-JSON checkpoint
    (runs/demo_source_ablation_partial.json) must independently show
    reach=100% over the `expected_n` cells it captured, AND every one of its
    strict values must agree with the full 50-run log parse for the same
    (arm, scenario_id). Two independent recordings of the same experiment
    agreeing is stronger evidence than either alone — this check exists
    precisely because these two artefacts were built by different code paths
    (an incrementally-checkpointed JSON vs. a stdout log capture) and could
    in principle have drifted from each other.
    """
    arm_rows = [r for r in rows if r.get("arm") == arm]
    n = len(arm_rows)
    reached = sum(1 for r in arm_rows if r.get("strict"))
    label = f"ablation {arm} reach (secondary check: {expected_n}-row partial-JSON checkpoint)"
    mark = "PASS" if (n == expected_n and reached == n) else "FAIL"
    print(f"  [{mark}] {label}: got={reached}/{n}  expected={expected_n}/{expected_n}")
    if not (n == expected_n and reached == n):
        results.append(f"{label}: got={reached}/{n}, expected={expected_n}/{expected_n}")

    log_strict_by_scenario = {
        (r["arm"], r["scenario_id"]): r["strict"] for r in log_records
    }
    disagreements = []
    for r in arm_rows:
        key = (arm, r.get("scenario_id"))
        log_strict = log_strict_by_scenario.get(key)
        if log_strict is None:
            disagreements.append((r.get("scenario_id"), "not found in full-log parse", r.get("strict")))
        elif log_strict != r.get("strict"):
            disagreements.append((r.get("scenario_id"), f"log={log_strict}", f"partial_json={r.get('strict')}"))

    if disagreements:
        results.append(f"ablation {arm}: partial-JSON vs full-log disagreement: {disagreements}")
    else:
        print(f"  [PASS] ablation {arm}: partial-JSON strict values agree with full-log parse for all {n} rows")


def check_solvable_reach(label, records, audit_solvable_ids, id_key, expected_num, expected_den, results):
    """Reach over structurally-solvable cells only (excludes known-unsolvable instances)."""
    solvable_records = [r for r in records if r.get(id_key) in audit_solvable_ids]
    den = len(solvable_records)
    num = sum(1 for r in solvable_records if r.get("warm_strict"))
    mark = "PASS" if (num == expected_num and den == expected_den) else "FAIL"
    print(f"  [{mark}] {label}: got={num}/{den}  expected={expected_num}/{expected_den}")
    if not (num == expected_num and den == expected_den):
        results.append(f"{label}: got={num}/{den}, expected={expected_num}/{expected_den}")


def check_shaping_arm(lm, arm, expected_reach, warm_reach, results):
    """Range-claim check for one reward-shaping-control arm.

    Certifies the fresh run's own numbers — a range claim over all five
    arms, not a selected "best arm" number. Four conditions, checked as one
    pass/fail per arm:
      - reach count matches this run's own recorded aggregate.json exactly
        (drift detector: a future rerun producing a different count fails
        loudly instead of silently passing a looser range).
      - reach strictly below the stored warm reference.
      - McNemar exact-binomial p vs warm <= 0.05 (shaping is statistically
        distinguishable from, and worse than, warm).
      - McNemar exact-binomial p vs cold >= 0.10 (shaping is NOT
        statistically distinguishable from cold at this cell count).
    """
    reach = arm["reach"]
    p_warm = arm["mcnemar_vs_warm_p"]
    p_cold = arm["mcnemar_vs_cold_p"]
    conditions = {
        f"reach == {expected_reach}": reach == expected_reach,
        f"reach < warm ({warm_reach})": reach < warm_reach,
        "p_vs_warm <= 0.05": p_warm is not None and p_warm <= 0.05,
        "p_vs_cold >= 0.10": p_cold is not None and p_cold >= 0.10,
    }
    passed = all(conditions.values())
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] lambda_shape={lm}: reach={reach}/{arm['n_cells']}  "
          f"p_vs_warm={p_warm:.4g}  p_vs_cold={p_cold:.4g}")
    if not passed:
        failed = [k for k, v in conditions.items() if not v]
        results.append(f"lambda_shape={lm} shaping-control range check failed: {failed}")


def main():
    failures = []

    # ── 25×25 ──────────────────────────────────────────────────────────────
    recs25 = load_json(SWEEP_25)
    n25 = len(recs25)
    warm_reach_25 = sum(1 for r in recs25 if r.get("warm_strict"))
    cold_reach_25 = sum(1 for r in recs25 if r.get("cold_strict"))
    win_25 = sum(1 for r in recs25 if win(r))
    mean_ratio_25 = mean_warm_ratio(recs25)

    print(f"25x25 ({n25} cells):")
    check_count("warm reach", warm_reach_25, 24, 1, n25, failures)
    check_count("cold reach", cold_reach_25, 13, 2, n25, failures)
    check_count("win count", win_25, 21, 2, n25, failures)
    check_ratio("mean warm cost ratio", mean_ratio_25, 4.42, 0.20, failures)

    # ── 50×50 1x ───────────────────────────────────────────────────────────
    recs1x = load_json(SWEEP_50_1X)
    n1x = len(recs1x)
    warm_reach_1x = sum(1 for r in recs1x if r.get("warm_strict"))
    cold_reach_1x = sum(1 for r in recs1x if r.get("cold_strict"))
    win_1x = sum(1 for r in recs1x if win(r))
    mean_ratio_1x = mean_warm_ratio(recs1x)
    med_ratio_1x = median_warm_ratio(recs1x)

    print(f"\n50x50 1x ({n1x} cells):")
    check_count("warm reach", warm_reach_1x, 22, 1, n1x, failures)
    check_count("cold reach", cold_reach_1x, 3, 1, n1x, failures)
    check_count("win count", win_1x, 21, 1, n1x, failures)
    check_ratio("mean warm ratio", mean_ratio_1x, 17.3, 0.20, failures)
    check_ratio("median warm ratio", med_ratio_1x, 10.5, 0.25, failures)

    # ── 50×50 4x ───────────────────────────────────────────────────────────
    recs4x = load_json(SWEEP_50_4X)
    n4x = len(recs4x)
    warm_reach_4x = sum(1 for r in recs4x if r.get("warm_strict"))
    cold_reach_4x = sum(1 for r in recs4x if r.get("cold_strict"))
    win_4x = sum(1 for r in recs4x if win(r))
    mean_ratio_4x = mean_warm_ratio(recs4x)
    med_ratio_4x = median_warm_ratio(recs4x)

    print(f"\n50x50 4x ({n4x} cells):")
    check_count("warm reach", warm_reach_4x, 25, 0, n4x, failures)
    check_count("cold reach", cold_reach_4x, 2, 1, n4x, failures)
    check_count("win count", win_4x, 24, 1, n4x, failures)
    check_ratio("mean warm ratio", mean_ratio_4x, 14.8, 0.20, failures)
    check_ratio("median warm ratio", med_ratio_4x, 4.3, 0.30, failures)

    # ── Reach over solvable cells only (excludes known-unsolvable instances) ──
    # NOTE: eval_reachability_audit.json's actual key for "structurally solvable"
    # is `reachable` (boolean), NOT `solvable` — the brief's guessed key name did
    # not match the real committed file. `scenario_id` and `scale` guesses were
    # correct and verified against the live file.
    audit = load_json(REACHABILITY_AUDIT)
    solvable_ids_25 = {r["scenario_id"] for r in audit if r.get("scale") == "25x25" and r.get("reachable")}
    solvable_ids_50 = {r["scenario_id"] for r in audit if r.get("scale") == "50x50" and r.get("reachable")}

    print("\nReach over solvable cells only:")
    check_solvable_reach("25x25 warm reach / solvable", recs25, solvable_ids_25, "scenario_id", 24, 24, failures)
    check_solvable_reach("50x50 1x warm reach / solvable", recs1x, solvable_ids_50, "scenario_id", 22, 22, failures)

    # ── Source ablation: full 24/24-per-arm reach, all three arms ─────────────
    print("\nSource ablation (all three arms, full solvable-cell coverage):")
    log_records = parse_ablation_log(ABLATION_LOG, failures)
    log_records = join_ablation_log_to_scenarios(log_records, recs25, failures)

    classical_num, classical_den = check_ablation_full_reach(
        log_records, "classical_only", solvable_ids_25, failures)
    quantum_num, quantum_den = check_ablation_full_reach(
        log_records, "quantum_only", solvable_ids_25, failures)
    check_solvable_reach(
        "ablation full_pool reach / solvable (source: runs/sweep_phase3_final.json, "
        "same warm arm as the headline sweep)",
        recs25, solvable_ids_25, "scenario_id", 24, 24, failures,
    )
    full_pool_solvable_rows = [r for r in recs25 if r.get("scenario_id") in solvable_ids_25]
    full_pool_num = sum(1 for r in full_pool_solvable_rows if r.get("warm_strict"))
    full_pool_den = len(full_pool_solvable_rows)

    print(
        f"\n  Ablation summary - classical_only {classical_num}/{classical_den} solvable, "
        f"quantum_only {quantum_num}/{quantum_den} solvable, full_pool {full_pool_num}/{full_pool_den} solvable"
    )

    # Secondary consistency check: the 9-row partial-JSON checkpoint should
    # independently agree with the full log parse above.
    ablation_rows = load_json(ABLATION_PARTIAL)
    check_ablation_subset_reach(ablation_rows, log_records, "classical_only", 5, failures)
    check_ablation_subset_reach(ablation_rows, log_records, "quantum_only", 4, failures)

    # ── Fleet ──────────────────────────────────────────────────────────────
    fleet = load_json(FLEET)
    n_q = fleet["n_queries"]
    astar_s = fleet["astar_total_s"]
    gnn_s = fleet["gnn_total_s"]
    tput = fleet["throughput_ratio"]

    print(f"\nFleet ({n_q} queries):")
    check_time("A* total_s", astar_s, 17.49, 0.30, "s", failures)
    check_time("GNN total_s", gnn_s, 0.126, 0.50, "s", failures)
    check_ratio("throughput_ratio", tput, 138.5, 0.50, failures)

    # ── Reward-shaping control (demonstration-free, lambda-sweep) ────────────
    # Certifies this run's own numbers as a range claim over all five arms,
    # not a single selected "best arm" number: reach 18/12/16/14/15 out of 25
    # for lambda_shape in {0.5,1,2,4,8}, each checked against the stored
    # warm/cold reference via exact binomial McNemar tests.
    shaping = load_json(SHAPING_CONTROL)
    warm_ref = shaping["warm_reach_reference"]
    print(f"\nReward-shaping control (demonstration-free, warm reference={warm_ref}/{shaping['n_cells_reference']}):")
    for lm_str, arm in sorted(shaping["arms"].items(), key=lambda kv: float(kv[0])):
        lm = float(lm_str)
        check_shaping_arm(lm, arm, EXPECTED_SHAPING_REACH[lm], warm_ref, failures)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n===")
    if failures:
        print(f"{len(failures)} checks failed")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
