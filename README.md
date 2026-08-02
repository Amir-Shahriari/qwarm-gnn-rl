# QWarm-RL — Persistent Expert Replay for GNN-RL Routing

An interactive browser demo and reproducibility package for a GraphSAGE-DQN
routing agent that learns to navigate a perturbing dynamic graph. A
**warm** (demonstration-assisted) agent — trained with a mixed pool of classical
(A\*/k-shortest-paths) and simulated-quantum (QAOA / noisy stochastic A\*)
demonstrations held in its replay buffer throughout training, under a
DQfD-style expert-margin loss — is raced head-to-head against a **cold**
(demonstration-free) agent trained on the identical graph and perturbation
sequence. Both agents start from identical random weights under the same seed:
*warm* and *cold* name the demonstration condition, not a parameter
initialisation. Alongside the demonstrations, the warm arm's offline buffer also
receives a small set of **goal-adjacent terminal transitions** (one per node one
hop from the destination, `reward = -step_cost + 100`); these are read off graph
adjacency rather than produced by an oracle, and they are warm-only, so they form
part of what the warm/cold contrast varies — see `train_gnn_dqn.py` (the block
guarded by `if oracles and re_seed_experts_each_iteration`).
The warm agent reaches 96% of 25×25 cells (52% cold) and 88% of
50×50 cells (12% cold); see
[Reproducing the paper's numbers](#reproducing-the-papers-numbers) below.

![Reach mode at 25×25: the warm agent reaches the goal in 13 steps, the cold agent exhausts its 500-step budget without arriving, and the Dijkstra reference path is shown alongside. Header panels report reach, warm optimality gap, and per-decision inference latency.](demo_ui.png)

*Reach mode at 25×25 — warm, cold, and Dijkstra on the same perturbed graph.
This instance is representative of the pattern the paper reports: the warm agent
arrives (13 steps, cost 295.2) while the cold agent wanders until its step budget
expires. Route quality is the standing limitation — the warm route costs 4.42×
the Dijkstra optimum (66.7) even though it reaches.*

Paper under review at IEEE ICDM 2026 (demo track).

## Quick start (one command)

```bash
git clone https://github.com/Amir-Shahriari/qwarm-gnn-rl.git
cd qwarm-gnn-rl
uv sync --extra dev
uv run python scripts/run_demo.py
```

The server precomputes all 10 scenarios at startup and does **not** accept
connections until that finishes, so **wait for the terminal to print
`Ready - open http://127.0.0.1:8765` before opening the page** — the launcher
also opens your browser automatically once the server is ready. During startup the
terminal prints progress (`Precomputing evaluated scenarios (k/10)...`). Opening
the URL before the `Ready` line will show a connection error until precompute
completes. Precompute is ~17-18s on a CPU-only dev machine; dependency install time
on top of that depends entirely on network speed and whether `uv`'s cache is warm.
(An in-page loading overlay is shown as a fallback if you reload mid-precompute.)

Works fully offline once dependencies are installed — no network calls at
runtime.

## Using the demo

The demo opens in a **Guided tour**: the warm and cold agents race the same
perturbed graph side by side (alongside the Dijkstra optimum), a first-run card
explains the setup, and "Try another scenario" steps through curated instances. A
persistent bar keeps the key point on screen — both agents run on the *same* graph
under the *same* perturbation, and the only difference is whether expert
demonstrations were available during training.

**Explore** (top-right toggle) exposes the full controls as three separate
selectors — graph scale (25×25 / 50×50), perturbation level, and evaluated
scenario — plus the oracle source (full pool / classical-only / quantum-sourced-only, on
25×25) and a **Sandbox** tab for authoring arbitrary source, destination, and
obstacle layouts. Warm, cold, and Dijkstra carry redundant encoding (solid /
dashed / dotted lines, circle / square markers) so the comparison reads without
relying on colour.

**Hardware:** CPU-only, no GPU required. ~4GB RAM, ~1GB disk (including
committed checkpoints). An NVIDIA GPU is used automatically if present, but
never required. On **Linux** specifically, the plain PyPI `torch` wheel
`uv sync` resolves to pulls in the full CUDA runtime as a transitive
dependency (no CPU-only variant exists on the plain index — that split only
exists on PyTorch's own `cpu`/`cu128` indices), so *install* size is larger
there even though the *runtime* path is genuinely CPU-only. macOS and
Windows wheels don't have this issue.

## Reproducing the paper's numbers

```bash
uv run python verify_demo_claims.py
```

Re-derives the paper's headline numbers from the committed per-cell
artefacts in `runs/` (no retraining required) and fails loudly if anything
diverges:

| Claim | Command / artefact |
|---|---|
| 25×25 warm 96% / cold 52%, 88%/12% at 50×50, 84% paired win-rate, mean gaps 4.42×/17.3× | `runs/sweep_phase3_final.json`, `runs/sweep_50x50/sweep_v1_50x50_{1x,4x}.json` |
| 24/24 and 22/22 reach over structurally-solvable cells | `runs/eval_reachability_audit.json` |
| Source-ablation reach (classical_only/quantum_only/full-pool) | `runs/demo_source_ablation.log` — the full 50-run log; all three arms reach 24/24 structurally-solvable 25×25 cells. `runs/demo_source_ablation_partial.json` is a 9-row checkpoint used as an independent cross-check that its strict values agree with the full-log parse (see the `check_ablation_full_reach` docstring in `verify_demo_claims.py`). Note `quantum_only` = QAOA **+** quantum-inspired stochastic, not QAOA alone |
| Fleet throughput (138×) | `runs/fleet_1779277545_seed42/fleet_results.json` |
| Reward-shaping control (demonstration-free λ-sweep, 5 arms) | `scripts/run_shaping_control.py`, `runs/shaping_control/{lambda_*.json,aggregate.json}` — a range claim over all five arms, not a single "best arm" number |
| Demonstration diversity (33.5 vs 4.8 unique paths/cell) | `runs/traces_25x25/*/{classical_only,quantum_only}/seeding_diversity.json` — means over the **24 structurally-solvable** cells; averaging over all 25 gives 33.2/4.7 and will not match the paper |
| 100×100 cold-start control at 4× budget (warm 10/15, cold 1/15, McNemar p=0.004) | `runs/cold_4x_control_{results,aggregate}.json`, `runs/sweep_v1_on_100x100.json` |
| Sample efficiency (warm 17/24 crossings, median 351 episodes; cold 1/24) | `runs/learning_curves_25x25.json` — first episode at which a rolling 25-episode window sustains 50% goal-reach, 500-episode budget |

Full training/eval provenance and seeds: `scripts/run_multi_seed_warm_vs_cold.py`
(25×25) and `scripts/run_sweep_50x50.py` (50×50), outer seeds
`[42, 1337, 2024, 7, 314159]`.

**Scope of this package.** The table above covers every quantitative claim the
paper makes about the demonstrated system. Figures the paper attributes to the
thesis — the within-trajectory adaptation result (6/6 at Δ∈{15,30,50}) and the
full 100×100 quality characterisation — are *not* reproduced here; the
within-trajectory code path exists (`realtime_perturb` in
`src/qwarm/env/pathfinding_env.py`) but its result artefacts are not part of
this repository.

**Run-to-run variance.** `verify_demo_claims.py` re-derives the headline numbers
from the committed artefacts; it does not retrain, and *retraining at the same
seeds does not reproduce the per-cell values*. The repository contains one
same-seed pair at 25×25 — `runs/sweep_phase3_final.json` (published) and
`runs/sweep_phase3_traced.json` (identical seeds, scenarios and graph
realisations, with per-episode tracing enabled). Between them: warm goal-reach
agrees on all 25 cells (24/25 in both, the single miss being the known-unsolvable
cell `seed518677876_s1`), cold goal-reach differs on **7 of 25** cells (13 vs 12
reached in aggregate — the close totals are coincidence, not stability), and path
costs differ on 14 warm and 15 cold cells, several by more than 2×. Determinism
flags are set in `src/qwarm/utils/seeding.py`, but with
`use_deterministic_algorithms(warn_only=True)`, so **no bit-level
reproducibility is claimed**. Treat reproduced per-cell numbers as a sample from
the same distribution, not as a match.

Note: `uv sync --extra dev` (the install path above) does not include
`qiskit`/`qiskit-aer`, so it cannot reproduce the `oracle_pool="full"` warm
arm used by these scripts — full-pool *training* raises an `ImportError` on
`FaithfulSimulatedQAOA` without it. `verify_demo_claims.py` and the live demo
don't need it (they read committed artefacts / load checkpoints); only
*fresh* full-pool retraining does — in that case run
`uv sync --extra dev --extra qiskit_backend` first.

## Experimental configurations

Three operating points appear in the results: 25×25, 50×50, and 100×100. They
differ in more than grid size, and the differences are not recorded in the
per-cell artefacts, so this table names the source of every value.

There is **no `configs/` directory in this repository**. The
`--config configs/default.yaml` invocation shown in the
`scripts/run_multi_seed_warm_vs_cold.py` docstring and argparse help refers to a
file that has never been committed. All configurations that *can* be read are
hardcoded Python constants: `GRID_TEMPLATE`/`TRAIN_CFG` in
`scripts/run_multi_seed_warm_vs_cold.py` (25×25) and `GRID_CFG`/`TRAIN_CFG` in
`scripts/run_sweep_50x50.py` (50×50). The 100×100 column below is recovered from
`runs/cold_4x_control_results.json`, which embeds `grid` and `training_config`
blocks — but those record the **cold** arm, so several 100×100 *warm* values are
genuinely unrecoverable and are marked as such rather than left blank.

Provenance: **[A]** read from a committed artefact · **[C]** code constant or
library default · **[D]** derived from other recorded values · **[?]** not
determinable from this repository. A compound mark means both apply: **[A/D]**
is a value computed from artefact-recorded quantities, **[C/D]** one computed
from code constants.

A **[C]** value is what the code says now, not a record of what executed. This
repository's history begins at its import commit and the sweeps predate it, so
for any code constant there is no way to confirm from here that the same value
was in effect when the artefacts were produced. That caveat applies to every
**[C]** cell below; it bears most on node deactivation at 25×25, where the value
comes from a class default the configuration never passes.

| Parameter | 25×25 | 50×50 (1×) | 100×100 |
|---|---|---|---|
| Nodes | 625 **[C]** | 2,500 **[C]** | 10,000 **[A]** |
| Hidden dimension | 64 **[C]** | 128 **[C]** | 128 **[A]** |
| Expert ratio ρ (warm) | 0.40 **[C]** | 0.30 **[C]** | **[?]** — cold arm records 0.0 **[A]** |
| Iterations N_it | 5 **[C]** | 10; 4× tier = 40 **[C]** | 40 at 4×, so 10 at 1× **[A/D]** |
| Episodes per iteration N_ep | 100 **[C]** | 200 **[C]** | 200 **[A]** |
| Gradient steps per episode | 4 **[C]** | 4 **[C]** | 4 **[A]** |
| Batch size | 64 **[C]** | 64 **[C]** | 64 **[A]** |
| Extra edges E_extra | 2 **[C]** | 3 **[C]** | 4 **[A]** |
| Edge deactivation p_deact | 0.15 **[C]** | 0.22 **[C]** | 0.30 **[A]** |
| Node deactivation | 0.05 **[C]** (class default, not passed) | 0.05 **[C]** (passed explicitly) | **[?]** — not in the recorded `grid` block |
| Pre-seed states N_ps | 5 **[C]** | 3 **[C]** | **[?]** — cold arm records 0 **[A]** |
| Pre-seed k-paths K | 10 **[C]** | 3 **[C]** | **[?]** — cold arm records 0 **[A]** |
| ε schedule | 0.40 → 0.05 over 5 iterations **[C/D]** | 0.40 → 0.05 over 10 (4×: 40) **[C/D]** | **[?]** — see note below |
| T_max, training | 400 **[C]** | 400 **[C]** | **[?]** — see note below |
| T_max, evaluation | 300 **[C]** | 300 and 1000, dual-budget **[C]** | 300 and 1000 **[A]** |
| Effective oracle set | A\* + stochastic + QAOA **[D]** | A\* + stochastic **[D]** | **[?]** — pool unknown; QAOA excluded by size **[D]** |

Batch size and gradient steps per episode are identical across all three, as are
γ (0.95), the learning rate (1e-4) and — between 25×25 and 50×50 — training
T_max and the ε endpoints. None of those identifies which configuration a run
used. Only the ε decay *rate* differs, and it follows from the iteration count
rather than being set independently
(`eps_decay = (0.40 - 0.05) / (N_it - 1)`).

The two **[?]** entries in the 100×100 column are marked so deliberately. ε and
training T_max are trainer defaults in
`src/qwarm/training/train_gnn_dqn.py`, and neither is exposed through
`TRAIN_CFG`, so any run driven by `scripts/run_multi_seed_warm_vs_cold.py` would
use 0.40 → 0.05 and 400. But the 100×100 values in this table come from
`runs/cold_4x_control_results.json`, whose `checkpoint` paths point outside this
repository, so the script that produced them is not available here and the
defaults cannot be confirmed to apply. They are most likely the same; they are
not evidenced.

**Training budget.** Total gradient steps are `N_it × N_ep × grad_steps × queries`,
with one query per cell:

| Configuration | Gradient steps | Artefact check |
|---|---|---|
| 25×25 | 2,000 **[D]** | not recorded |
| 50×50 1× | 8,000 **[D]** | warm median 8,000 **[A]** |
| 50×50 4× | 32,000 **[D]** | warm median 32,000 **[A]** |
| 100×100 4× | 32,000 **[D]** | `total_rollouts: 8000` (40 × 200) **[A]** |

"1×" and "4×" refer to these totals. The 50×50 rows are corroborated per cell:
those artefacts record `warm_grad_steps_total`/`cold_grad_steps_total`, which is
what the check column reports. The 25×25 row is arithmetic only — that artefact
records no gradient-step field.

No 100×100 run at 1× is present in this repository, so its 8,000-step budget is
not tabulated above: it would follow from dividing the recorded 40 iterations by
the recorded `compute_multiplier: 4`. On that inference — and only on it — the
50×50 1× budget would equal the 100×100 1× budget, making the 25×25 run a
quarter of both. Treat that as a plausible reading of two recorded numbers, not
as a measured correspondence.

Cold medians run ~40 steps short of warm (7,960 and 31,960). Those totals are
artefact-recorded; the explanation is not, but follows from the trainer, where
gradient steps begin only once the buffer holds `batch_size` transitions and the
cold arm has no pre-seeded transitions to start from.

**Oracle availability by scale.** `FaithfulSimulatedQAOA.solve` returns `inf`
immediately when the graph exceeds 1,000 nodes
(`src/qwarm/oracles/faithful_qaoa.py:104`). Both in-repository sweep scripts
construct the same three-oracle pool, so at 25×25 (625 nodes) all three
contribute, while at 50×50 (2,500) the effective pool reduces to
`ClassicalAStar` plus `QuantumInspiredStochasticOracle`. The reduction is
inferred from the size guard rather than observed — no artefact records which
oracle produced a given seeded path.

The same guard would apply at 100×100 (10,000 nodes), but the script that
produced those runs is not in this repository, so which oracles it constructed
is unknown; all that can be said is that QAOA could not have contributed at that
size. In every case this follows from graph size rather than from a
configuration choice, and it is not visible in the `oracle_pool` setting.

**Warm/cold pairing.** Within each configuration the two arms are matched: the
cold agent is built on a separate `DynamicGraph` constructed from the same
`scenario.grid_seed` (so it sees the identical perturbation sequence), with the
same seed, hidden dimension, γ, iterations, episodes, gradient steps and batch
size. The only intended differences are the expert pool (`oracles=[]`), the
expert ratio (`expert_ratio=0.0`), and pre-seeding
(`re_seed_experts_each_iteration=False`). See
`scripts/run_multi_seed_warm_vs_cold.py:141-170` (25×25) and
`scripts/run_sweep_50x50.py:333-354` (50×50). Pre-seeding and the goal-adjacent
terminal transitions sit behind a single gate at
`src/qwarm/training/train_gnn_dqn.py:87`
(`if oracles and re_seed_experts_each_iteration and pre_seed_n_states > 0`), so
both are strictly warm-only.

## Troubleshooting

- **Port 8765 already in use:** `uv run python scripts/run_demo.py --port 8766`.
- **Wrong Python version:** this project requires Python 3.11 (`requires-python = ">=3.11,<3.12"` in `pyproject.toml`). `uv python install 3.11` if you don't have it, then re-run `uv sync`.
- **"missing checkpoint" error at startup:** `demo_agents/*.pt` and `runs/traces_25x25/seed42_*/{classical_only,quantum_only}/agent.pt` must be present — these are committed to git, not downloaded; if they're missing you likely have a shallow/partial clone. Re-clone fully.
- **`uv sync` tries to reach the network and fails (fully offline machine):** dependencies must be resolved with network access at least once (to populate `uv`'s cache); after that, `uv sync --offline` works from cache.
- **CUDA/GPU errors:** the default install is CPU-only and never touches CUDA at runtime; if you manually installed a CUDA build of torch and it's misconfigured, `uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall`, or just re-run `uv sync` to fall back to the CPU wheel.
- **CI reference:** `.github/workflows/ci.yml` runs, in order: unit tests (`pytest -m "not slow" --cov=qwarm`), the offline-operation test (`tests/test_offline.py`), the headless demo smoke test (`tests/test_demo_smoke.py`), and the claims-verification gate (`verify_demo_claims.py`).

## License

MIT — see `LICENSE`.
