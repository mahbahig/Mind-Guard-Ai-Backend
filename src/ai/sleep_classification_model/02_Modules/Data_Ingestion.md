# Data Ingestion Module — MESA PPG

**Follows Implementation Guide:** [[How_to_Segment_30s_Epochs]]

## 📝 Description
Downloads MESA "Pleth" (optical PPG) signal and AASM sleep stage labels from the NSRR, segments into 30-second epochs, and saves to `data/processed/mesa_ppg.npy`.

**Script:** `src/data_ingestion.py`

## 🔑 Connection

- **Dataset:** MESA (Multi-Ethnic Study of Atherosclerosis)
- **API:** `https://sleepdata.org/api/v1/datasets/mesa/`
- **Token:** `30672-tzKq562EPM9uzs8aKH5K`
- **Auth confirmed:** ✅ 2,056 EDF files listed successfully

## 📡 Pipeline Steps

| Step | Action |
|---|---|
| 1 | List EDF files via NSRR API (`polysomnography/edfs/`) |
| 2 | Download EDF + paired XML annotation per subject to `data/raw/` cache |
| 3 | Parse EDF with inline reader — extract **Pleth** channel, detect FS |
| 4 | Parse MESA NSRR XML → per-epoch AASM stage (0=W,1=N1,2=N2,3=N3,4=REM) |
| 5 | Segment signal into 30s epochs, align with labels |
| 6 | Save to `data/processed/mesa_ppg.npy` (+ `.h5` if h5py available) |

## ▶️ Run

```bash
# Download first 10 subjects (default)
python src/data_ingestion.py

# Download N subjects
python src/data_ingestion.py --n 5
```

## 📤 Output Format

```python
# NPY file — load with:
data = np.load('data/processed/mesa_ppg.npy', allow_pickle=True).item()
epochs      = data['epochs']       # float32 (N_epochs, epoch_samples)
labels      = data['labels']       # int8    (N_epochs,)  0-4
subject_ids = data['subject_ids']  # object  (N_epochs,)

# Or via helper:
subjects, fs = load_mesa_data()   # list of (epochs, labels) per subject
```

## 🐛 Known Issues / Notes
- **h5py not installed** (disk space during pip): NPY output used as fallback. HDF5 available once disk space freed.
- **Pleth channel name** varies per EDF record — fallback labels: `Pleth`, `PPG`, `SpO2`, `pleth`, `ppg`
- **AASM stage 9** = Unscored/artifact epochs are automatically discarded
