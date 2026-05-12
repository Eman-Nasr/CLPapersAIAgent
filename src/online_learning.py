import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from river import linear_model
from river import preprocessing
from river import metrics
from river import drift

from src.config import OUTPUTS_DIR


# =========================================================
# Create online model
# =========================================================

def create_model():
    return (
        preprocessing.StandardScaler()
        | linear_model.LogisticRegression()
    )


model = create_model()

# Drift detector
adwin = drift.ADWIN(delta=0.01)

# Accuracy tracker
accuracy_metric = metrics.Accuracy()


# =========================================================
# Load latest retrieval results
# =========================================================

def load_latest_results():

    tests = sorted(OUTPUTS_DIR.glob("test*"))

    if not tests:
        raise FileNotFoundError(
            "No outputs/test* directories found."
        )

    latest = tests[-1]

    retrieval_path = latest / "retrieval_results.json"

    with open(retrieval_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    return results, latest


# =========================================================
# Convert retrieval output into streaming samples
# =========================================================

def create_streaming_samples(results):

    samples = []

    for query in results:

        relevant_ids = set(
            query.get("relevant_paper_ids", [])
        )

        for hit in query.get("top_k", []):

            score = hit.get("score", 0.0)
            rank = hit.get("rank", 0)
            word_count = hit.get("word_count", 0)

            x = {
                "score": score,
                "rank": rank,
                "word_count": word_count,
            }

            # Simulated feedback label
            # clicked helpful? y/n
            y = 1 if hit["paper_id"] in relevant_ids else 0

            samples.append((x, y))

    return samples


# =========================================================
# Online learning with drift handling
# =========================================================

def run_online_learning(samples):

    global model
    global accuracy_metric

    prequential_history = []

    drift_detected = False

    total_samples = len(samples)

    print(
        f"\n[online] Processing {total_samples} samples...\n"
    )

    for i, (x, y) in enumerate(samples, start=1):

        # =================================================
        # Inject artificial drift after halfway
        # =================================================

        if i > total_samples // 2:

            x["score"] = -9999
            x["rank"] = 9999
            x["word_count"] = 1

            if i == (total_samples // 2) + 1:
                print(
                    "\n[online] Artificial topic drift injected.\n"
                )

        # =================================================
        # Predict before learning
        # =================================================

        y_pred = model.predict_one(x)

        if y_pred is None:
            y_pred = 0

        # =================================================
        # Update running metric
        # =================================================

        accuracy_metric.update(y, y_pred)

        current_accuracy = accuracy_metric.get()

        prequential_history.append({
            "sample": i,
            "accuracy": current_accuracy
        })

        # =================================================
        # Drift detection
        # =================================================

        error_value = abs(x["score"])

        adwin.update(error_value)

        if adwin.drift_detected:

            print(
                f"\n[ADWIN] Drift detected at sample {i}"
            )

            drift_detected = True

            # =============================================
            # Drift response
            # Reset model
            # =============================================

            print(
                "[online] Resetting online learner...\n"
            )

            model = create_model()

        # =================================================
        # Online learning update
        # =================================================

        model.learn_one(x, y)

        print(
            f"[online] sample={i} "
            f"| pred={y_pred} "
            f"| actual={y} "
            f"| acc={current_accuracy:.3f}"
        )

    return prequential_history, drift_detected


# =========================================================
# Save metrics CSV
# =========================================================

def save_metrics_csv(history, out_dir):

    csv_path = out_dir / "prequential_metrics.csv"

    with open(csv_path, "w", newline="",
              encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["sample", "accuracy"])

        for row in history:
            writer.writerow([
                row["sample"],
                row["accuracy"]
            ])

    print(f"[online] CSV saved -> {csv_path}")


# =========================================================
# Plot prequential accuracy
# =========================================================

def plot_accuracy(history, out_dir):

    samples = [h["sample"] for h in history]
    accuracies = [h["accuracy"] for h in history]

    plt.figure(figsize=(10, 5))

    plt.plot(samples, accuracies)

    plt.xlabel("Streaming Sample")

    plt.ylabel("Prequential Accuracy")

    plt.title(
        "Online Learning Performance with Drift Handling"
    )

    plt.grid(True)

    plot_path = out_dir / "prequential_accuracy.png"

    plt.savefig(plot_path)

    plt.close()

    print(f"[online] Plot saved -> {plot_path}")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print("\n[online] Loading retrieval results ...")

    retrieval_results, latest_output = load_latest_results()

    print("[online] Creating streaming samples ...")

    samples = create_streaming_samples(
        retrieval_results
    )

    print(
        f"[online] {len(samples)} streaming samples created."
    )

    history, drift_detected = run_online_learning(
        samples
    )

    # =====================================================
    # Create output folder
    # =====================================================

    online_dir = latest_output / "online_learning"

    online_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save CSV
    save_metrics_csv(history, online_dir)

    # Save plot
    plot_accuracy(history, online_dir)

    # Save summary
    summary = {
        "samples_processed": len(samples),
        "final_accuracy": round(
            accuracy_metric.get(), 4
        ),
        "drift_detected": drift_detected,
    }

    summary_path = online_dir / "online_summary.json"

    with open(summary_path, "w",
              encoding="utf-8") as f:

        json.dump(summary, f, indent=2)

    print("\n==============================")
    print("ONLINE LEARNING COMPLETE")
    print("==============================")
    print(
        f"Samples processed: {len(samples)}"
    )
    print(
        f"Final Accuracy: "
        f"{accuracy_metric.get():.4f}"
    )
    print(
        f"Drift detected: {drift_detected}"
    )
    print("==============================\n")