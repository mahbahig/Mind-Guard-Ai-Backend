"""
MindGuard Smartwatch Simulator
===============================
Simulates a full day+night cycle:
  1. Daytime: sends synthetic PPG windows to the HRV Stress API (port 8000)
  2. Night:   calls /simulate_night on the Sleep API (port 8001) which runs
              prediction on held-out test subjects never seen during training
  3. Correlation: sends both outputs to the Correlation Engine (port 8002)

Usage:
    python run_simulation.py           (launches servers + this script)
    python scripts/simulate_watch.py   (if servers are already running)
"""

import requests
import time
import sys
import numpy as np

try:
    import neurokit2 as nk
    _HAS_NK = True
except ImportError:
    _HAS_NK = False

STRESS_API = "http://localhost:8000/predict_stress"
SLEEP_API = "http://localhost:8001"
CORRELATE_API = "http://localhost:8002/correlate"


# ──────────────────────────────────────────────
#  Daytime stress simulation (synthetic PPG)
# ──────────────────────────────────────────────

def generate_synthetic_ppg(duration_sec=30, sampling_rate=256, heart_rate=75):
    if _HAS_NK:
        ppg = nk.ppg_simulate(duration=duration_sec, sampling_rate=sampling_rate, heart_rate=heart_rate)
        noise = np.random.normal(0, 0.1, len(ppg))
        return (ppg + noise).tolist()
    t = np.linspace(0, duration_sec, duration_sec * sampling_rate, endpoint=False)
    freq = heart_rate / 60.0
    ppg = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    ppg += np.random.normal(0, 0.15, len(ppg))
    return ppg.tolist()


def run_daytime_stress(n_windows=6):
    """Simulate daytime by sending n_windows of synthetic PPG to the Stress API."""
    print("\n" + "=" * 60)
    print("  DAYTIME SIMULATION -- HRV Stress Monitoring")
    print("=" * 60)

    stress_readings = []
    sampling_rate = 256

    for i in range(n_windows):
        hr = int(np.random.uniform(60, 110))
        ppg = generate_synthetic_ppg(duration_sec=30, sampling_rate=sampling_rate, heart_rate=hr)
        payload = {"ppg_signal": ppg, "sampling_rate": sampling_rate}

        try:
            t0 = time.time()
            resp = requests.post(STRESS_API, json=payload, timeout=30)
            latency_ms = (time.time() - t0) * 1000

            if resp.status_code == 200:
                result = resp.json()
                stress_readings.append({
                    "timestamp": f"window_{i+1}",
                    "stress_level": result["stress_level_int"],
                })
                sdrr = result["extracted_features"].get("SDRR", 0)
                lf_hf = result["extracted_features"].get("LF_HF", 0)
                print(
                    f"  [{i+1}/{n_windows}] HR~{hr}bpm | "
                    f"Stress: {result['stress_level_label']} ({result['stress_level_int']}) | "
                    f"SDRR={float(sdrr):.2f} LF/HF={float(lf_hf):.2f} | "
                    f"{latency_ms:.0f}ms"
                )
            else:
                print(f"  [{i+1}/{n_windows}] Error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            print(f"  [{i+1}/{n_windows}] Connection failed -- is the Stress API running?")
            break

        if i < n_windows - 1:
            time.sleep(1)

    return stress_readings


# ──────────────────────────────────────────────
#  Night sleep simulation (held-out real data)
# ──────────────────────────────────────────────

def run_night_simulation():
    """
    Call /simulate_night on the Sleep API to classify all epochs of a
    held-out test subject the model has never seen during training.
    """
    print("\n" + "=" * 60)
    print("  NIGHT SIMULATION -- Sleep Stage Classification")
    print("  (Held-out test subject, never seen during training)")
    print("=" * 60)

    try:
        print("  Requesting full-night prediction from Sleep API...")
        t0 = time.time()
        resp = requests.get(f"{SLEEP_API}/simulate_night", timeout=600)
        elapsed = time.time() - t0

        if resp.status_code == 200:
            result = resp.json()
            bio = result["biomarkers"]
            accuracy = result["accuracy"]
            n_epochs = result["n_epochs"]
            preds = result["epoch_predictions"]

            gt_wake = sum(1 for p in preds if p["true_label"] == 0)
            gt_sleep = n_epochs - gt_wake
            pred_wake = sum(1 for p in preds if p["predicted_stage"] == 0)
            pred_sleep = n_epochs - pred_wake
            mean_conf = np.mean([p["confidence"] for p in preds]) if preds else 0

            print(f"\n  {result['summary']}")
            print(f"\n  Ground Truth:      Wake={gt_wake:>4} | Sleep={gt_sleep:>4}")
            print(f"  Model Predictions: Wake={pred_wake:>4} | Sleep={pred_sleep:>4}")
            print(f"  Accuracy: {accuracy:.1%} | Mean Confidence: {mean_conf:.1%}")
            print(f"  Processed {n_epochs} epochs in {elapsed:.1f}s "
                  f"({elapsed/n_epochs*1000:.0f}ms/epoch)")

            print(f"\n  Sleep Biomarkers:")
            print(f"    TST:  {bio['tst_minutes']} min")
            print(f"    SOL:  {bio['sol_minutes']} min")
            print(f"    WASO: {bio['waso_minutes']} min")
            print(f"    SE:   {bio['sleep_efficiency_pct']}%")

            return bio
        else:
            print(f"  Error {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("  Connection failed -- is the Sleep API running on port 8001?")
        return None
    except requests.exceptions.ReadTimeout:
        print("  Request timed out (>600s). The subject may have too many epochs.")
        return None


# ──────────────────────────────────────────────
#  Correlation
# ──────────────────────────────────────────────

def run_correlation(sleep_biomarkers, stress_readings):
    """Send both outputs to the Correlation Engine."""
    print("\n" + "=" * 60)
    print("  MINDGUARD CORRELATION -- Health Risk Assessment")
    print("=" * 60)

    payload = {
        "sleep_biomarkers": sleep_biomarkers,
        "stress_readings": stress_readings,
    }

    try:
        resp = requests.post(CORRELATE_API, json=payload, timeout=30)

        if resp.status_code == 200:
            result = resp.json()
            risk = result["risk_level"]
            indicator = {"Low": "[OK]", "Moderate": "[!]", "High": "[!!!]"}.get(risk, "")
            print(f"\n  Risk Level: {risk} {indicator}")

            if result["flags"]:
                print(f"\n  Flags ({len(result['flags'])}):")
                for f in result["flags"]:
                    print(f"    - {f['flag']}: {f['description']}")

            if result["predictions"]:
                print(f"\n  Predictions:")
                for p in result["predictions"]:
                    print(f"    > {p}")

            print(f"\n  Recommendation:")
            print(f"    {result['recommendation']}")

            return result
        else:
            print(f"  Error {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("  Connection failed -- is the Correlation Engine running on port 8002?")
        return None


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MINDGUARD SMARTWATCH SIMULATOR")
    print("  Full Day+Night Cycle")
    print("=" * 60)

    stress_readings = run_daytime_stress(n_windows=6)

    sleep_biomarkers = run_night_simulation()

    if sleep_biomarkers and stress_readings:
        run_correlation(sleep_biomarkers, stress_readings)
    else:
        print("\n[!] Could not run correlation -- missing sleep or stress data.")
        if stress_readings and not sleep_biomarkers:
            print("    Sleep API may not be running on port 8001.")
        if sleep_biomarkers and not stress_readings:
            print("    Stress API may not be running on port 8000.")

    print("\n" + "=" * 60)
    print("  Simulation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
