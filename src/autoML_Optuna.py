from __future__ import annotations

import json
import math
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import HyperbandPruner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition           import TruncatedSVD
from sklearn.preprocessing           import normalize
from sklearn.metrics.pairwise        import cosine_similarity

from src.config import OUTPUTS_DIR, TOP_K, LATENCY_PERCENTILE


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    rels  = [1 if pid in relevant else 0 for pid in retrieved[:k]]
    ideal = [1] * min(len(relevant), k)
    idcg  = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, pid in enumerate(retrieved, 1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


@dataclass
class TfidfConfig:
    max_features:   int
    ngram_range:    tuple[int, int]
    min_df:         int
    max_df:         float
    sublinear_tf:   bool
    norm:           str
    use_svd:        bool
    svd_components: int
    stop_words:     str | None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ngram_range"] = list(self.ngram_range)
        if not self.use_svd:
            d.pop("svd_components", None)
        return d


def build_index(texts: list[str], cfg: TfidfConfig):
    vectorizer = TfidfVectorizer(
        max_features  = cfg.max_features,
        ngram_range   = cfg.ngram_range,
        min_df        = cfg.min_df,
        max_df        = cfg.max_df,
        stop_words    = cfg.stop_words,
        sublinear_tf  = cfg.sublinear_tf,
        norm          = cfg.norm,
        strip_accents = "unicode",
    )
    doc_mat = vectorizer.fit_transform(texts)

    svd = None
    if cfg.use_svd:
        n_feats = doc_mat.shape[1]
        n_comp  = min(cfg.svd_components, max(2, n_feats - 1))
        svd     = TruncatedSVD(n_components=n_comp, random_state=42)
        doc_mat = svd.fit_transform(doc_mat)
        doc_mat = normalize(doc_mat, norm="l2")

    return vectorizer, svd, doc_mat


def retrieve(query_texts: list[str],
             vectorizer:  TfidfVectorizer,
             svd:         TruncatedSVD | None,
             doc_mat,
             k:           int) -> tuple[list[list[int]], list[float]]:
    top_indices: list[list[int]] = []
    latencies:   list[float]     = []

    for qtext in query_texts:
        t0 = time.time()
        q_vec = vectorizer.transform([qtext])
        if svd is not None:
            q_vec  = svd.transform(q_vec)
            q_vec  = normalize(q_vec, norm="l2")
            scores = (q_vec @ doc_mat.T).ravel()
        else:
            scores = cosine_similarity(q_vec, doc_mat).ravel()
        idx = np.argsort(scores)[::-1][:k]
        latencies.append((time.time() - t0) * 1000)
        top_indices.append(idx.tolist())

    return top_indices, latencies


def evaluate_split(top_indices: list[list[int]],
                   queries:     list[dict],
                   chunks:      list[dict],
                   k:           int) -> dict:
    recalls, ndcgs, mrrs = [], [], []
    for q, idxs in zip(queries, top_indices):
        relevant = set(q.get("relevant_paper_ids", []))
        if not relevant:
            continue
        retrieved_pids = [chunks[i]["paper_id"] for i in idxs]
        recalls.append(recall_at_k(retrieved_pids, relevant, k))
        ndcgs.append  (ndcg_at_k  (retrieved_pids, relevant, k))
        mrrs.append   (mrr        (retrieved_pids, relevant))

    def _mean(xs): return float(np.mean(xs)) if xs else 0.0
    return {
        f"Recall@{k}": round(_mean(recalls), 4),
        f"NDCG@{k}":   round(_mean(ndcgs),   4),
        "MRR":         round(_mean(mrrs),    4),
        "n_evaluated": len(recalls),
    }


def _kfold_indices(n: int, k: int, seed: int = 42) -> list[np.ndarray]:
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [perm[i::k] for i in range(k)]


def make_objective(chunks:      list[dict],
                   queries:     list[dict],
                   k:           int,
                   n_folds:     int,
                   latency_pen: float):
    labelled = [q for q in queries if q.get("relevant_paper_ids")]
    if len(labelled) < n_folds:
        raise ValueError(
            f"Not enough labelled queries ({len(labelled)}) for {n_folds}-fold CV."
        )
    folds = _kfold_indices(len(labelled), n_folds)
    texts = [c["text"] for c in chunks]

    def objective(trial: optuna.Trial) -> float:
        ngram_str = trial.suggest_categorical("ngram_range", ["1_1", "1_2", "1_3"])
        a, b = map(int, ngram_str.split("_"))

        stop_words = trial.suggest_categorical("stop_words", [None, "english"])
        use_svd    = trial.suggest_categorical("use_svd", [True, False])

        
        if use_svd:
            svd_components = trial.suggest_int("svd_components", 64, 512, log=True)
        else:
            svd_components = 0   

        cfg = TfidfConfig(
            max_features   = trial.suggest_int      ("max_features", 5_000, 100_000, log=True),
            ngram_range    = (a, b),
            min_df         = trial.suggest_int      ("min_df", 1, 5),
            max_df         = trial.suggest_float    ("max_df", 0.80, 1.00),
            sublinear_tf   = trial.suggest_categorical("sublinear_tf", [True, False]),
            norm           = trial.suggest_categorical("norm",         ["l1", "l2"]),
            use_svd        = use_svd,
            svd_components = svd_components,
            stop_words     = stop_words,
        )

        if cfg.max_df * len(texts) <= cfg.min_df:
            raise optuna.TrialPruned()

        try:
            vec, svd, doc_mat = build_index(texts, cfg)
        except ValueError as e:
            trial.set_user_attr("build_error", str(e))
            raise optuna.TrialPruned()

        fold_ndcgs:    list[float] = []
        fold_lat_p95s: list[float] = []
        for step, fold_idx in enumerate(folds):
            split_q = [labelled[i] for i in fold_idx]
            qtexts  = [q["text"] for q in split_q]
            top_indices, lats = retrieve(qtexts, vec, svd, doc_mat, k)
            metrics = evaluate_split(top_indices, split_q, chunks, k)
            fold_ndcgs.append   (metrics[f"NDCG@{k}"])
            fold_lat_p95s.append(float(np.percentile(lats, LATENCY_PERCENTILE)))

            running_mean = float(np.mean(fold_ndcgs))
            trial.report(running_mean, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_ndcg = float(np.mean(fold_ndcgs))
        p95_lat_s = float(np.mean(fold_lat_p95s)) / 1000.0
        score     = mean_ndcg - latency_pen * p95_lat_s

        trial.set_user_attr("mean_ndcg",      round(mean_ndcg, 4))
        trial.set_user_attr("p95_latency_ms", round(p95_lat_s * 1000, 3))
        trial.set_user_attr("matrix_shape",   list(doc_mat.shape))
        return score

    return objective


BASELINE_CONFIG = TfidfConfig(
    max_features   = 50_000,
    ngram_range    = (1, 2),
    min_df         = 2,
    max_df         = 1.0,
    sublinear_tf   = True,
    norm           = "l2",
    use_svd        = False,
    svd_components = 0,
    stop_words     = None,
)


def evaluate_full(cfg: TfidfConfig,
                  chunks:  list[dict],
                  queries: list[dict],
                  k:       int) -> dict:
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    vec, svd, doc_mat = build_index(texts, cfg)
    build_s = round(time.time() - t0, 3)

    labelled = [q for q in queries if q.get("relevant_paper_ids")]
    qtexts   = [q["text"] for q in labelled]
    top_indices, lats = retrieve(qtexts, vec, svd, doc_mat, k)

    metrics = evaluate_split(top_indices, labelled, chunks, k)
    metrics["avg_latency_ms"] = round(float(np.mean(lats)), 3)
    metrics[f"p{LATENCY_PERCENTILE}_latency_ms"] = round(
        float(np.percentile(lats, LATENCY_PERCENTILE)), 3
    )
    metrics["build_time_s"] = build_s
    metrics["matrix_shape"] = list(doc_mat.shape)
    return metrics


def _trials_to_records(study: optuna.Study) -> list[dict]:
    out = []
    for t in study.trials:
        out.append({
            "number":   t.number,
            "state":    t.state.name,
            "value":    t.value,
            "params":   t.params,
            "user_attrs": t.user_attrs,
            "duration_s": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start else None
            ),
        })
    return out


def _prequential_curve(study: optuna.Study) -> list[dict]:
    curve, best = [], -float("inf")
    for t in study.trials:
        if t.value is None:
            continue
        best = max(best, t.value)
        curve.append({"trial": t.number, "value": t.value, "best_so_far": round(best, 4)})
    return curve


def _write_report(out_dir: Path,
                  baseline:    dict,
                  tuned:       dict,
                  best_cfg:    TfidfConfig,
                  study:       optuna.Study,
                  total_time:  float,
                  n_trials:    int,
                  k:           int) -> None:
    n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_failed   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)

    def _delta(b, a):
        if b == 0: return "n/a"
        return f"{(a - b) / b * 100:+.1f}%"

    md = f"""# AutoML Tuning Report — TF-IDF Retriever (D1)

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Total wall-clock:** {total_time:.1f}s
**Trials:** {n_trials} requested · {n_complete} complete · {n_pruned} pruned · {n_failed} failed
**Sampler:** TPESampler (Bayesian / informed search)
**Pruner:** HyperbandPruner (multi-fidelity / Successive Halving over CV folds)

---

## 1. Search Space
| Hyperparameter   | Type        | Range / Choices                |
|------------------|-------------|--------------------------------|
| max_features     | int (log)   | 5 000 – 100 000                |
| ngram_range      | categorical | (1,1), (1,2), (1,3)            |
| min_df           | int         | 1 – 5                          |
| max_df           | float       | 0.80 – 1.00                    |
| sublinear_tf     | bool        | True / False                   |
| norm             | categorical | l1 / l2                        |
| stop_words       | categorical | None / "english"               |
| use_svd          | bool        | True / False                   |
| svd_components   | int (log)   | 64 – 512  (conditional on SVD) |


## 2. Baseline vs AutoML Results
| Metric                    | Baseline | AutoML (tuned) | Δ |
|---------------------------|---------:|---------------:|---:|
| Recall@{k}                | {baseline[f'Recall@{k}']} | {tuned[f'Recall@{k}']} | {_delta(baseline[f'Recall@{k}'], tuned[f'Recall@{k}'])} |
| NDCG@{k}                  | {baseline[f'NDCG@{k}']} | {tuned[f'NDCG@{k}']} | {_delta(baseline[f'NDCG@{k}'], tuned[f'NDCG@{k}'])} |
| MRR                       | {baseline['MRR']} | {tuned['MRR']} | {_delta(baseline['MRR'], tuned['MRR'])} |
| Avg Latency (ms) ↓        | {baseline['avg_latency_ms']} | {tuned['avg_latency_ms']} | {_delta(baseline['avg_latency_ms'], tuned['avg_latency_ms'])} |
| p{LATENCY_PERCENTILE} Latency (ms) ↓ | {baseline[f'p{LATENCY_PERCENTILE}_latency_ms']} | {tuned[f'p{LATENCY_PERCENTILE}_latency_ms']} | {_delta(baseline[f'p{LATENCY_PERCENTILE}_latency_ms'], tuned[f'p{LATENCY_PERCENTILE}_latency_ms'])} |
| Build Time (s) ↓          | {baseline['build_time_s']} | {tuned['build_time_s']} | {_delta(baseline['build_time_s'], tuned['build_time_s'])} |
| Matrix Shape              | {baseline['matrix_shape']} | {tuned['matrix_shape']} | N/A |

## 3. Winning Configuration
```json
{json.dumps(best_cfg.to_dict(), indent=2)}
```
"""
    (out_dir / "automl_report.md").write_text(md, encoding="utf-8")


def run_automl(output_dir:  Path,
               n_trials:    int   = 60,
               n_folds:     int   = 5,
               latency_pen: float = 0.05,
               seed:        int   = 42,
               k:           int   = TOP_K) -> dict:
    output_dir = Path(output_dir)
    automl_dir = output_dir / "automl"
    automl_dir.mkdir(parents=True, exist_ok=True)

    chunks_path  = output_dir / "all_chunks.json"
    queries_path = output_dir / "query_set.json"
    if not chunks_path.exists() or not queries_path.exists():
        raise FileNotFoundError(
            f"Expected {chunks_path} and {queries_path}. Run run_pipeline.py first."
        )

    chunks  = json.loads(chunks_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    print(f"[automl] corpus={len(chunks)} chunks  | queries={len(queries)} "
          f"(labelled={sum(1 for q in queries if q.get('relevant_paper_ids'))})")
    print(f"[automl] trials={n_trials}  folds={n_folds}  k={k}  "
          f"latency_pen={latency_pen}  seed={seed}")

    print("[automl] evaluating baseline …")
    t0       = time.time()
    baseline = evaluate_full(BASELINE_CONFIG, chunks, queries, k)
    print(f"[automl] baseline NDCG@{k}={baseline[f'NDCG@{k}']}  "
          f"Recall@{k}={baseline[f'Recall@{k}']}  "
          f"p95_lat={baseline[f'p{LATENCY_PERCENTILE}_latency_ms']}ms")

    sampler = TPESampler(seed=seed)
    pruner  = HyperbandPruner(min_resource=1, max_resource=n_folds, reduction_factor=3)
    study   = optuna.create_study(
        direction = "maximize",
        sampler   = sampler,
        pruner    = pruner,
        study_name = f"tfidf_automl_{datetime.now():%Y%m%d_%H%M%S}",
    )
    objective = make_objective(chunks, queries, k=k,
                               n_folds=n_folds, latency_pen=latency_pen)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t_search0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_s  = round(time.time() - t_search0, 2)

    n_pruned  = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"[automl] search done in {search_s}s "
          f"(best score={study.best_value:.4f}, pruned={n_pruned}/{n_trials})")

    best_params = study.best_params
    a, b = map(int, best_params["ngram_range"].split("_"))
    best_cfg = TfidfConfig(
        max_features   = best_params["max_features"],
        ngram_range    = (a, b),
        min_df         = best_params["min_df"],
        max_df         = best_params["max_df"],
        sublinear_tf   = best_params["sublinear_tf"],
        norm           = best_params["norm"],
        use_svd        = best_params["use_svd"],
        svd_components = best_params.get("svd_components", 0),
        stop_words     = best_params["stop_words"],
    )
    print("[automl] evaluating best config on full query set …")
    tuned = evaluate_full(best_cfg, chunks, queries, k)
    print(f"[automl] tuned    NDCG@{k}={tuned[f'NDCG@{k}']}  "
          f"Recall@{k}={tuned[f'Recall@{k}']}  "
          f"p95_lat={tuned[f'p{LATENCY_PERCENTILE}_latency_ms']}ms")

    run_card = {
    "created_at":         datetime.now().isoformat(),
    "k":                  k,
    "n_trials":           n_trials,
    "n_folds":            n_folds,
    "latency_pen":        latency_pen,
    "seed":               seed,
    "sampler":            f"TPESampler(seed={seed})",
    "pruner":             f"HyperbandPruner(min_resource=1, max_resource={n_folds}, reduction_factor=3)",
    "best_score":         round(study.best_value, 6),
    "best_trial":         study.best_trial.number,
    "best_config":        best_cfg.to_dict(),
    "tuned_full_eval":    tuned,
    "baseline_eval":      baseline,
    "search_wallclock_s": search_s,
    }
    (automl_dir / "best_config.json").write_text(
        json.dumps(run_card, indent=2), encoding="utf-8"
    )
    (automl_dir / "optuna_trials.json").write_text(
        json.dumps(_trials_to_records(study), indent=2, default=str),
        encoding="utf-8",
    )
    (automl_dir / "automl_comparison.json").write_text(
        json.dumps({"baseline": baseline, "tuned": tuned}, indent=2),
        encoding="utf-8",
    )

    curve = _prequential_curve(study)
    with open(automl_dir / "prequential_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "value", "best_so_far"])
        w.writeheader()
        w.writerows(curve)

    _write_report(automl_dir, baseline, tuned, best_cfg, study,
                  total_time=time.time() - t0, n_trials=n_trials, k=k)

    print(f"[automl] artifacts -> {automl_dir}")
    return run_card


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna AutoML for the TF-IDF retriever.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-trials",    type=int,   default=60)
    p.add_argument("--n-folds",     type=int,   default=5)
    p.add_argument("--latency-pen", type=float, default=0.05)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--k",           type=int,   default=TOP_K)
    return p.parse_args()


def _resolve_output_dir(arg_dir: Path | None) -> Path:
    if arg_dir is not None:
        return arg_dir
    tests = sorted(OUTPUTS_DIR.glob("test*"))
    if not tests:
        raise FileNotFoundError("No outputs/test* directories found. Run run_pipeline.py first.")
    return tests[-1]


if __name__ == "__main__":
    args   = _parse_args()
    outdir = _resolve_output_dir(args.output_dir)
    print(f"[automl] using output dir: {outdir}")
    run_automl(
        output_dir  = outdir,
        n_trials    = args.n_trials,
        n_folds     = args.n_folds,
        latency_pen = args.latency_pen,
        seed        = args.seed,
        k           = args.k,
    )
