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

Note: `uv sync --extra dev` (the install path above) does not include
`qiskit`/`qiskit-aer`, so it cannot reproduce the `oracle_pool="full"` warm
arm used by these scripts — full-pool *training* raises an `ImportError` on
`FaithfulSimulatedQAOA` without it. `verify_demo_claims.py` and the live demo
don't need it (they read committed artefacts / load checkpoints); only
*fresh* full-pool retraining does — in that case run
`uv sync --extra dev --extra qiskit_backend` first.

## Troubleshooting

- **Port 8765 already in use:** `uv run python scripts/run_demo.py --port 8766`.
- **Wrong Python version:** this project requires Python 3.11 (`requires-python = ">=3.11,<3.12"` in `pyproject.toml`). `uv python install 3.11` if you don't have it, then re-run `uv sync`.
- **"missing checkpoint" error at startup:** `demo_agents/*.pt` and `runs/traces_25x25/seed42_*/{classical_only,quantum_only}/agent.pt` must be present — these are committed to git, not downloaded; if they're missing you likely have a shallow/partial clone. Re-clone fully.
- **`uv sync` tries to reach the network and fails (fully offline machine):** dependencies must be resolved with network access at least once (to populate `uv`'s cache); after that, `uv sync --offline` works from cache.
- **CUDA/GPU errors:** the default install is CPU-only and never touches CUDA at runtime; if you manually installed a CUDA build of torch and it's misconfigured, `uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall`, or just re-run `uv sync` to fall back to the CPU wheel.
- **CI reference:** `.github/workflows/ci.yml` runs, in order: unit tests (`pytest -m "not slow" --cov=qwarm`), the offline-operation test (`tests/test_offline.py`), the headless demo smoke test (`tests/test_demo_smoke.py`), and the claims-verification gate (`verify_demo_claims.py`).

## License

MIT — see `LICENSE`.
