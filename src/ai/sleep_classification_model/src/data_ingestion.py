"""
MESA PPG Data Ingestion — Sleep Stage Classification
======================================================
Downloads the MESA "Pleth" (PPG) channel and sleep stage labels
from the NSRR using the official `nsrr` python library.
Segments them into 30-second epochs and saves the result to
data/processed/mesa_ppg.h5 (or .npy).

Reference spec: 02_Modules/Data_Ingestion.md
NSRR Dataset   : https://sleepdata.org/datasets/mesa

Usage
-----
    python src/data_ingestion.py --n 5         # download and process 5 subjects
"""

import argparse
import os
import re
import struct
import subprocess
import sys
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────

TOKEN_FILE      = Path(__file__).resolve().parent.parent / "not important" / "token.txt"
PLETH_LABELS    = {"Pleth", "PPG", "SpO2", "pleth", "ppg"} 
EPOCH_SEC       = 30
MAX_SUBJECTS    = 5

ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data" / "processed"
RAW_DIR         = ROOT_DIR / "data" / "raw" / "mesa"
EDF_DIR         = RAW_DIR / "polysomnography" / "edfs"
XML_DIR         = RAW_DIR / "polysomnography" / "annotations-events-nsrr"

# MESA XML annotation values → 0-4 model labels
MESA_STAGE_MAP = {
    "0": 0,   # Wake
    "1": 1,   # NREM 1
    "2": 2,   # NREM 2
    "3": 3,   # NREM 3
    "4": 3,   # NREM 4 → merge with N3
    "5": 4,   # REM
    "9": -1,  # Unscored / artifact
}

# ─────────────────────────────────────────────────────────────
#  NSRR CLI Helpers
# ─────────────────────────────────────────────────────────────

def download_subject(subject_id):
    """
    Use nsrr CLI to download specific subject files.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "nsrr", "mesa",
        "--subject", subject_id,
        "-d",         "-t", str(TOKEN_FILE.resolve()),
    ]
    print(f"  Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [Error] Failed to download {subject_id}: {result.stderr}")
        return False
    return True

# ─────────────────────────────────────────────────────────────
#  Minimal EDF Reader (No dependencies)
# ─────────────────────────────────────────────────────────────

def _parse_edf_header(f):
    f.seek(0)
    # Skipping generic header info for brevity, focusing on signal info
    f.seek(236)
    n_signals = int(f.read(4).decode("ascii", errors="replace").strip())
    
    labels      = [f.read(16).decode("ascii", errors="replace").strip() for _ in range(n_signals)]
    f.read(80 * n_signals) # transducer
    f.read(8 * n_signals)  # phys_dim
    phys_min    = [float(f.read(8).decode("ascii", errors="replace").strip()) for _ in range(n_signals)]
    phys_max    = [float(f.read(8).decode("ascii", errors="replace").strip()) for _ in range(n_signals)]
    dig_min     = [int(f.read(8).decode("ascii", errors="replace").strip())   for _ in range(n_signals)]
    dig_max     = [int(f.read(8).decode("ascii", errors="replace").strip())   for _ in range(n_signals)]
    f.read(80 * n_signals) # prefilter
    n_samples   = [int(f.read(8).decode("ascii", errors="replace").strip())   for _ in range(n_signals)]
    # ... skip reserved
    
    # Calculate gain/offset for signal conversion
    gains   = [(phys_max[i] - phys_min[i]) / (dig_max[i] - dig_min[i]) if (dig_max[i]-dig_min[i]) != 0 else 1.0 for i in range(n_signals)]
    offsets = [phys_min[i] - gains[i] * dig_min[i] for i in range(n_signals)]

    f.seek(8, 0) # Read start time and date
    f.seek(236 + 16*n_signals + 80*n_signals + 8*n_signals + 8*n_signals + 8*n_signals + 8*n_signals + 8*n_signals + 80*n_signals + 8*n_signals + 32*n_signals)
    hdr_bytes = 256 + 256 * n_signals
    
    # Actually, let's just use the known structure from Step 307
    f.seek(0)
    f.read(236)
    n_signals = int(f.read(4).decode("ascii", errors="replace").strip())
    labels = [f.read(16).decode("ascii", errors="replace").strip() for _ in range(n_signals)]
    transducer = [f.read(80).decode("ascii", errors="replace").strip() for _ in range(n_signals)]
    phys_dim = [f.read(8).decode("ascii", errors="replace").strip() for _ in range(n_signals)]
    phys_min = [float(f.read(8).decode("ascii", errors="replace").strip()) for _ in range(n_signals)]
    phys_max = [float(f.read(8).decode("ascii", errors="replace").strip()) for _ in range(n_signals)]
    dig_min = [int(f.read(8).decode("ascii", errors="replace").strip()) for _ in range(n_signals)]
    dig_max = [int(f.read(8).decode("ascii", errors="replace").strip()) for _ in range(n_signals)]
    prefilter = [f.read(80).decode("ascii", errors="replace").strip() for _ in range(n_signals)]
    n_samples = [int(f.read(8).decode("ascii", errors="replace").strip()) for _ in range(n_signals)]
    
    f.seek(184)
    n_records = int(f.read(8).decode("ascii", errors="replace").strip())
    dur_record = float(f.read(8).decode("ascii", errors="replace").strip())

    return {
        "n_records": n_records,
        "dur_record": dur_record,
        "n_signals": n_signals,
        "labels": labels,
        "n_samples": n_samples,
        "gains": gains,
        "offsets": offsets,
        "hdr_bytes": 256 + 256 * n_signals,
    }

def read_edf_pleth(edf_path):
    with open(edf_path, "rb") as f:
        hdr = _parse_edf_header(f)
        pleth_idx = -1
        for i, lbl in enumerate(hdr["labels"]):
            if any(p.lower() in lbl.lower() for p in PLETH_LABELS):
                pleth_idx = i
                break
        
        if pleth_idx == -1:
            return None, None
            
        fs = hdr["n_samples"][pleth_idx] / hdr["dur_record"]
        gain = hdr["gains"][pleth_idx]
        offset = hdr["offsets"][pleth_idx]
        
        f.seek(hdr["hdr_bytes"])
        record_size = sum(hdr["n_samples"]) * 2
        pleth_data = []
        skip_bytes = sum(hdr["n_samples"][:pleth_idx]) * 2
        after_bytes = sum(hdr["n_samples"][pleth_idx+1:]) * 2
        
        for _ in range(hdr["n_records"]):
            f.read(skip_bytes)
            raw = f.read(hdr["n_samples"][pleth_idx] * 2)
            fmt = f"<{hdr['n_samples'][pleth_idx]}h"
            samples = struct.unpack(fmt, raw)
            pleth_data.extend([s * gain + offset for s in samples])
            f.read(after_bytes)
            
    return np.array(pleth_data, dtype=np.float32), fs

# ─────────────────────────────────────────────────────────────
#  MESA XML Parser
# ─────────────────────────────────────────────────────────────

def parse_mesa_xml(xml_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    stages = []
    for event in root.findall(".//ScoredEvent"):
        concept = event.findtext("EventConcept")
        if concept and "Stage" in concept:
            dur = float(event.findtext("Duration", 30))
            # Extract digit from "Stage X"
            match = re.search(r"\d", concept)
            stage_code = match.group() if match else "9"
            label = MESA_STAGE_MAP.get(stage_code, -1)
            # Repeat label for each 30s epoch in this block
            num_epochs = int(dur // 30)
            stages.extend([label] * num_epochs)
    return np.array(stages, dtype=np.int8)

# ─────────────────────────────────────────────────────────────
#  Processing
# ─────────────────────────────────────────────────────────────

def process_subject(subject_id):
    # Subject edf is mesa-sleep-[id].edf
    # Subject xml is mesa-sleep-[id]-nsrr.xml
    # Note: IDs in mesa file names are like 0001
    edf_files = list(EDF_DIR.glob(f"mesa-sleep-{subject_id}.edf"))
    xml_files = list(XML_DIR.glob(f"mesa-sleep-{subject_id}-nsrr.xml"))
    
    if not edf_files or not xml_files:
        print(f"  [Error] Missing files for {subject_id}")
        return None
        
    print(f"  Processing {subject_id}...")
    pleth, fs = read_edf_pleth(edf_files[0])
    labels = parse_mesa_xml(xml_files[0])
    
    if pleth is None:
        print(f"  [Error] No Pleth channel found for {subject_id}")
        return None
        
    samples_per_epoch = int(30 * fs)
    num_epochs = min(len(pleth) // samples_per_epoch, len(labels))
    
    epochs = []
    final_labels = []
    for i in range(num_epochs):
        if labels[i] == -1: continue # Skip unscored
        epoch = pleth[i*samples_per_epoch : (i+1)*samples_per_epoch]
        epochs.append(epoch.astype(np.float16)) # Save space
        final_labels.append(labels[i])
        
    return np.array(epochs), np.array(final_labels), fs

# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=MAX_SUBJECTS)
    args = parser.parse_args()
    
    # 1. We need subject IDs. For now let's use a small set of IDs if we can list them.
    # Since --list-subjects failed, let's try a heuristic list: 0001, 0002, 0003...
    subject_ids = [f"{i:04d}" for i in range(1, args.n + 1)]
    
    all_epochs = []
    all_labels = []
    all_subject_ids = []
    final_fs = None
    
    for sid in subject_ids:
        print(f"--- Subject {sid} ---")
        if not download_subject(sid):
            continue
            
        data = process_subject(sid)
        if data:
            epochs, labels, fs = data
            all_epochs.append(epochs)
            all_labels.append(labels)
            all_subject_ids.extend([sid] * len(labels))
            final_fs = fs
            
    if not all_epochs:
        print("No data collected.")
        return
        
    all_epochs = np.vstack(all_epochs)
    all_labels = np.concatenate(all_labels)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        import h5py
        output_file = DATA_DIR / "mesa_ppg.h5"
        with h5py.File(output_file, "w") as f:
            f.create_dataset("epochs", data=all_epochs, compression="gzip")
            f.create_dataset("labels", data=all_labels)
            f.create_dataset("subject_ids", data=np.array(all_subject_ids, dtype="S"))
            f.attrs["fs"] = final_fs
        print(f"Successfully saved to {output_file}")
    except ImportError:
        output_file = DATA_DIR / "mesa_ppg.npy"
        np.save(output_file, {
            "epochs": all_epochs,
            "labels": all_labels,
            "subject_ids": all_subject_ids,
            "fs": final_fs
        })
        print(f"h5py not found, saved to {output_file} instead.")

if __name__ == "__main__":
    main()
