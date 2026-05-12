from river import drift


adwin = drift.ADWIN()


def detect_drift(values):

    drift_found = False

    for i, value in enumerate(values):

        adwin.update(value)

        if adwin.drift_detected:

            drift_found = True

            print(
                f"[ADWIN] Drift detected at index {i}"
            )

    if not drift_found:
        print("[ADWIN] No drift detected.")