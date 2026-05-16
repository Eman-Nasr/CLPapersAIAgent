"""
evaluate_member4.py  –  Member 4: Evaluation Outputs
=====================================================
Standalone script that reads the AutoML results produced by
autoML_Optuna.py and generates three deliverables:

    automl/automl_comparison_chart.png  — bar chart (quality + latency)
    automl/automl_comparison.csv        — flat comparison table
    automl/automl_eval_summary.md       — report-ready markdown summary

Run after autoML_Optuna.py has finished:
    python evaluate_member4.py                  # uses latest outputs/test* dir
    python evaluate_member4.py --output-dir outputs/test3/automl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Constants (must match autoML_Optuna.py / config.py) ───────────────────────
TOP_K              = 5
LATENCY_PERCENTILE = 95

# ── Colour palette (dark theme, matches online_learning.py style) ─────────────
_BG_FIGURE  = "#0f1117"
_BG_AXES    = "#161b22"
_GRID       = "#21262d"
_TICK       = "#c9d1d9"
_SPINE      = "#30363d"
_LABEL      = "#8b949e"
_TITLE      = "#e6edf3"
_COL_BASE   = "#4F8EF7"   # blue  — baseline
_COL_TUNED  = "#3fb950"   # green — AutoML tuned
_COL_WORSE  = "#ff7b72"   # red   — regression


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_automl_dir() -> Path:
    """Walk outputs/test* dirs newest-first and return the first that has
    an automl/automl_comparison.json file."""
    outputs = Path(__file__).parent / "outputs"
    if not outputs.exists():
        outputs = Path("outputs")          # try cwd
    tests = sorted(outputs.glob("test*"), reverse=True)
    for t in tests:
        candidate = t / "automl" / "automl_comparison.json"
        if candidate.exists():
            return t / "automl"
    raise FileNotFoundError(
        "No automl/automl_comparison.json found under outputs/test*. "
        "Run autoML_Optuna.py first."
    )


def load_comparison(automl_dir: Path) -> tuple[dict, dict, dict | None]:
    """
    Returns (baseline_metrics, tuned_metrics, best_config_card).
    best_config_card may be None if best_config.json is absent.
    """
    comp_path = automl_dir / "automl_comparison.json"
    if not comp_path.exists():
        raise FileNotFoundError(f"Missing {comp_path}")
    data = json.loads(comp_path.read_text(encoding="utf-8"))
    baseline = data["baseline"]
    tuned    = data["tuned"]

    cfg_card = None
    cfg_path = automl_dir / "best_config.json"
    if cfg_path.exists():
        cfg_card = json.loads(cfg_path.read_text(encoding="utf-8"))

    return baseline, tuned, cfg_card


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bar chart  (primary deliverable)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_chart(baseline: dict, tuned: dict,
                          out_dir: Path, k: int = TOP_K) -> Path:
    """
    Two-panel bar chart:
      Left  — Retrieval quality  (NDCG@k, Recall@k, MRR)   higher → better
      Right — Query latency      (avg_ms, p95_ms)            lower  → better

    Each AutoML bar is annotated with the signed absolute delta vs baseline.
    """
    lat_key = f"p{LATENCY_PERCENTILE}_latency_ms"

    # ── Data ─────────────────────────────────────────────────────────────────
    q_labels = [f"NDCG@{k}", f"Recall@{k}", "MRR"]
    q_base   = [baseline[f"NDCG@{k}"],  baseline[f"Recall@{k}"],  baseline["MRR"]]
    q_tuned  = [tuned[f"NDCG@{k}"],     tuned[f"Recall@{k}"],     tuned["MRR"]]

    l_labels = ["Avg Latency (ms)", "p95 Latency (ms)"]
    l_base   = [baseline["avg_latency_ms"], baseline[lat_key]]
    l_tuned  = [tuned["avg_latency_ms"],    tuned[lat_key]]

    # ── Layout ───────────────────────────────────────────────────────────────
    fig, (ax_q, ax_l) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(_BG_FIGURE)
    for ax in (ax_q, ax_l):
        ax.set_facecolor(_BG_AXES)
        ax.tick_params(colors=_TICK)
        for spine in ax.spines.values():
            spine.set_edgecolor(_SPINE)
        ax.grid(True, axis="y", color=_GRID, linewidth=0.7, zorder=0)

    bar_w = 0.35

    # ── Quality panel ─────────────────────────────────────────────────────────
    xs = list(range(len(q_labels)))
    ax_q.bar([x - bar_w / 2 for x in xs], q_base,
             bar_w, label="Baseline", color=_COL_BASE,  alpha=0.88, zorder=2)
    ax_q.bar([x + bar_w / 2 for x in xs], q_tuned,
             bar_w, label="AutoML",   color=_COL_TUNED, alpha=0.88, zorder=2)

    q_top = max(q_base + q_tuned)
    for xi, (bv, tv) in enumerate(zip(q_base, q_tuned)):
        diff  = tv - bv
        sign  = "+" if diff >= 0 else ""
        color = _COL_TUNED if diff >= 0 else _COL_WORSE
        ax_q.text(xi + bar_w / 2, tv + q_top * 0.02,
                  f"{sign}{diff:.3f}",
                  ha="center", va="bottom", fontsize=8.5, color=color, zorder=3)

    ax_q.set_xticks(xs)
    ax_q.set_xticklabels(q_labels, color=_TICK, fontsize=9.5)
    ax_q.set_ylim(0, q_top * 1.28)
    ax_q.set_ylabel("Score  (↑ higher is better)", color=_LABEL, fontsize=9)
    ax_q.set_title("Retrieval Quality", color=_TITLE, fontsize=10.5, pad=10)
    ax_q.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax_q.legend(facecolor=_BG_AXES, edgecolor=_SPINE,
                labelcolor=_TICK, fontsize=9)

    # ── Latency panel ─────────────────────────────────────────────────────────
    xl = list(range(len(l_labels)))
    ax_l.bar([x - bar_w / 2 for x in xl], l_base,
             bar_w, label="Baseline", color=_COL_BASE,  alpha=0.88, zorder=2)
    ax_l.bar([x + bar_w / 2 for x in xl], l_tuned,
             bar_w, label="AutoML",   color=_COL_TUNED, alpha=0.88, zorder=2)

    l_top = max(l_base + l_tuned)
    for xi, (bv, tv) in enumerate(zip(l_base, l_tuned)):
        diff  = tv - bv
        sign  = "+" if diff >= 0 else ""
        # For latency: negative diff = faster = good (green), positive = slower = bad (red)
        color = _COL_TUNED if diff <= 0 else _COL_WORSE
        ax_l.text(xi + bar_w / 2, tv + l_top * 0.02,
                  f"{sign}{diff:.2f} ms",
                  ha="center", va="bottom", fontsize=8.5, color=color, zorder=3)

    ax_l.set_xticks(xl)
    ax_l.set_xticklabels(l_labels, color=_TICK, fontsize=9.5)
    ax_l.set_ylim(0, l_top * 1.30)
    ax_l.set_ylabel("Milliseconds  (↓ lower is better)", color=_LABEL, fontsize=9)
    ax_l.set_title("Query Latency", color=_TITLE, fontsize=10.5, pad=10)
    ax_l.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_l.legend(facecolor=_BG_AXES, edgecolor=_SPINE,
                labelcolor=_TICK, fontsize=9)

    # ── Figure title & layout ─────────────────────────────────────────────────
    fig.suptitle(
        f"Baseline vs AutoML (Optuna TPE)  —  TF-IDF Retriever  |  k = {k}",
        color=_TITLE, fontsize=11.5, y=1.02,
    )
    plt.tight_layout()
    out_path = out_dir / "automl_comparison_chart.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[eval-m4] chart saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Comparison CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_csv(baseline: dict, tuned: dict,
                        out_dir: Path, k: int = TOP_K) -> Path:
    """
    Flat CSV: metric | baseline | automl_tuned | delta_pct | better_direction.
    Mirrors build_tfidf_retrieval's evaluation_results.csv for consistency.
    """
    lat_key = f"p{LATENCY_PERCENTILE}_latency_ms"

    def _pct_delta(b, a) -> str:
        if b == 0:
            return "n/a"
        return f"{(a - b) / b * 100:+.1f}%"

    rows = [
        # (metric, baseline_val, tuned_val, better_direction)
        (f"NDCG@{k}",        baseline[f"NDCG@{k}"],        tuned[f"NDCG@{k}"],        "higher"),
        (f"Recall@{k}",      baseline[f"Recall@{k}"],      tuned[f"Recall@{k}"],      "higher"),
        ("MRR",              baseline["MRR"],               tuned["MRR"],              "higher"),
        ("avg_latency_ms",   baseline["avg_latency_ms"],   tuned["avg_latency_ms"],   "lower"),
        (lat_key,            baseline[lat_key],             tuned[lat_key],            "lower"),
        ("build_time_s",     baseline["build_time_s"],     tuned["build_time_s"],     "lower"),
        ("n_evaluated",      baseline.get("n_evaluated","?"), tuned.get("n_evaluated","?"), "info"),
        ("matrix_shape",     str(baseline.get("matrix_shape","")),
                             str(tuned.get("matrix_shape","")),                        "info"),
    ]

    out_path = out_dir / "automl_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "automl_tuned", "delta_pct", "better_direction"])
        for metric, bv, tv, direction in rows:
            try:
                delta = _pct_delta(float(bv), float(tv))
            except (TypeError, ValueError):
                delta = "—"
            w.writerow([metric, bv, tv, delta, direction])

    print(f"[eval-m4] comparison CSV → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Markdown summary
# ─────────────────────────────────────────────────────────────────────────────

def save_eval_summary_md(baseline: dict, tuned: dict,
                         cfg_card: dict | None,
                         out_dir: Path, k: int = TOP_K) -> Path:
    """
    Concise, report-ready markdown that can be dropped straight into the
    D1 deliverable document. Includes:
      - baseline vs tuned table with signed deltas
      - winning hyperparameter configuration (if best_config.json found)
      - notes on metric direction
    """
    lat_key = f"p{LATENCY_PERCENTILE}_latency_ms"

    def _delta(b, a, lower_is_better: bool = False) -> str:
        if b == 0:
            return "n/a"
        pct = (a - b) / b * 100
        return f"{pct:+.1f}%"

    n_eval_b = baseline.get("n_evaluated", "?")
    n_eval_t = tuned.get("n_evaluated",   "?")

    cfg_block = ""
    if cfg_card:
        best_cfg = cfg_card.get("best_config", {})
        n_trials = cfg_card.get("n_trials", "?")
        n_folds  = cfg_card.get("n_folds",  "?")
        lat_pen  = cfg_card.get("latency_pen", "?")
        best_score = cfg_card.get("best_score", "?")
        best_trial = cfg_card.get("best_trial", "?")
        search_s   = cfg_card.get("search_wallclock_s", "?")
        cfg_block = f"""
## 3. Winning Configuration

| Setting           | Value |
|-------------------|-------|
| Trials run        | {n_trials} |
| CV folds          | {n_folds} |
| Latency penalty   | {lat_pen} per p{LATENCY_PERCENTILE} latency second |
| Best trial        | #{best_trial} |
| Best score        | {best_score} |
| Search time       | {search_s}s |

```json
{json.dumps(best_cfg, indent=2)}
```
"""

    md = f"""# Member 4 — Evaluation Summary: Baseline vs AutoML
## TF-IDF Retriever  |  k = {k}

> Δ = (AutoML − Baseline) / Baseline × 100.
> For ↓ metrics, a **negative Δ is an improvement**.

## 1. Retrieval Quality  (↑ higher is better)

| Metric      | Baseline | AutoML | Δ |
|-------------|:--------:|:------:|:---:|
| NDCG@{k}    | {baseline[f"NDCG@{k}"]} | {tuned[f"NDCG@{k}"]} | {_delta(baseline[f"NDCG@{k}"], tuned[f"NDCG@{k}"])} |
| Recall@{k}  | {baseline[f"Recall@{k}"]} | {tuned[f"Recall@{k}"]} | {_delta(baseline[f"Recall@{k}"], tuned[f"Recall@{k}"])} |
| MRR         | {baseline["MRR"]} | {tuned["MRR"]} | {_delta(baseline["MRR"], tuned["MRR"])} |

## 2. Latency & Build Cost  (↓ lower is better)

| Metric                  | Baseline | AutoML | Δ |
|-------------------------|:--------:|:------:|:---:|
| Avg latency (ms)        | {baseline["avg_latency_ms"]} | {tuned["avg_latency_ms"]} | {_delta(baseline["avg_latency_ms"], tuned["avg_latency_ms"], lower_is_better=True)} |
| p{LATENCY_PERCENTILE} latency (ms)      | {baseline[lat_key]} | {tuned[lat_key]} | {_delta(baseline[lat_key], tuned[lat_key], lower_is_better=True)} |
| Build time (s)          | {baseline["build_time_s"]} | {tuned["build_time_s"]} | {_delta(baseline["build_time_s"], tuned["build_time_s"], lower_is_better=True)} |
| Queries evaluated       | {n_eval_b} | {n_eval_t} | — |
| Matrix shape            | {baseline.get("matrix_shape","?")} | {tuned.get("matrix_shape","?")} | — |
{cfg_block}
---
*Generated by evaluate_member4.py — Member 4 evaluation stage only.*
"""
    out_path = out_dir / "automl_eval_summary.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"[eval-m4] markdown summary → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(automl_dir: Path, k: int = TOP_K) -> None:
    automl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval-m4] loading results from {automl_dir}")
    baseline, tuned, cfg_card = load_comparison(automl_dir)

    plot_comparison_chart(baseline, tuned, automl_dir, k)
    save_comparison_csv(baseline,  tuned, automl_dir, k)
    save_eval_summary_md(baseline, tuned, cfg_card, automl_dir, k)

    print("\n[eval-m4] Done. Outputs written to:", automl_dir)
    print("  automl_comparison_chart.png")
    print("  automl_comparison.csv")
    print("  automl_eval_summary.md")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Member 4 evaluation: bar chart + CSV + markdown for AutoML results."
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Path to the automl/ sub-directory (e.g. outputs/test3/automl). "
             "If omitted, the script finds the latest outputs/test*/automl automatically.",
    )
    p.add_argument("--k", type=int, default=TOP_K,
                   help=f"Top-k value used during retrieval (default {TOP_K}).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.output_dir is not None:
        automl_dir = Path(args.output_dir)
    else:
        automl_dir = _find_automl_dir()
        print(f"[eval-m4] auto-detected automl dir: {automl_dir}")

    run_evaluation(automl_dir, k=args.k)
