

# mox-sensor-forecasting

PySpark pipeline for MOX gas sensor data — EDA, STL time series analysis, MLlib (LR & Random Forest) and a GRU neural network for 1-step resistance forecasting across 14 sensors.

---

## Dataset

[UCI Gas Sensor Array Under Dynamic Gas Mixtures](https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures)

16 features per timestep recorded by a metal-oxide (MOX) sensor array:

| Column | Description |
|---|---|
| `time_s` | Elapsed time (s) |
| `co_ppm` | CO concentration (ppm) |
| `humidity_rh` | Relative humidity (%) |
| `temp_c` | Temperature (°C) |
| `flow_ml_min` | Flow rate (ml/min) |
| `heater_v` | Heater voltage (V) |
| `R1–R14` | Sensor resistances (MΩ) |

Place all CSV files under `Data/` before running.

---

## Project Structure

```
mox-sensor-forecasting/
├── Data/               # Raw CSV files (not included — download from UCI)
├── figures/            # Auto-generated plots
├── DataSci.py          # Full pipeline script
├── TechReport.docx     # Technical report with analysis & results
└── README.md
```

---

## Pipeline Overview

```
Raw CSVs → PySpark Ingestion → Cleaning & Winsorization
    → EDA & Visualizations
    → Spark Queries (Accumulators, ReduceByKey, Windows)
    → Feature Engineering
    → STL / ACF / ADF Time Series Analysis
    → MLlib: Linear Regression & Random Forest (per sensor)
    → GRU Neural Network (multi-output, 12h sequence)
```

---

## Dependencies

```bash
pip install pyspark tensorflow scikit-learn statsmodels matplotlib pandas numpy
```

Tested with:
- Python 3.10+
- PySpark 3.5
- TensorFlow 2.15

---

## How to Run

1. Download the dataset and place the CSV files in `Data/`
2. Run the script:

```bash
python DataSci.py
```

Figures are saved automatically to `figures/`.

> **Note:** The script is configured for local Spark (`local[2]`). For a cluster, update the `.master()` config accordingly.

---

## Results

### Spark MLlib — per-sensor average (14 sensors)

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Linear Regression | — | — | — | — |
| Random Forest | — | — | — | — |

### GRU — per-sensor average (14 outputs)

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| GRU (seq=144) | — | — | — | — |
| Persistence Baseline | — | — | — | — |

> Fill in with your final test metrics.

---

## Figures

Generated plots include:
- Resistance boxplot per sensor
- CO, temperature, and heater voltage distributions
- Scatter: resistance vs CO / humidity (sensor R10)
- Mean resistance vs CO rolling mean (time segment)
- STL decomposition of `r_mean14`
- ACF & PACF of STL residuals
- True vs Predicted plots for LR, RF, and GRU (sensor R10)
