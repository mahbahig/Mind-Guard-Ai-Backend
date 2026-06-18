# 🎛️ Dataset Switchboard

Tell the Orchestrator which dataset configuration is currently active by checking ONLY ONE box below.

## Active Dataset
- [X] **SLPDB** (Blood Pressure - BP signal)
- [] **MESA** (Photoplethysmography - PPG signal)

## ⚙️ SLPDB Configuration (Active if Checked)
```python
# Save as config.py or load dynamically
CONFIG = {
    "DATASET_NAME": "slpdb",
    "SIGNAL_TYPE": "BP",
    "SAMPLING_RATE": 250,
    "EPOCH_SIZE_SEC": 30,
    "LSTM_UNITS": 64,
    "DATA_PATH": "./data/slpdb/raw/"
}
```

## ⚙️ MESA Configuration (Active if Checked)
```python
# Save as config.py or load dynamically
CONFIG = {
    "DATASET_NAME": "mesa",
    "SIGNAL_TYPE": "PPG",
    "SAMPLING_RATE": 256,
    "EPOCH_SIZE_SEC": 30,
    "LSTM_UNITS": 128,
    "DATA_PATH": "./data/mesa/raw/"
}
```
