from pyspark.sql import SparkSession, functions as F, types as T
import os, sys
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

try:
    spark.stop()
except:
    pass

spark = (SparkSession.builder
    .appName("MOX-UCI")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "8")
    .config("spark.python.worker.reuse", "true")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext
print("AppId:", sc.applicationId)

print("CWD =", os.getcwd())

import glob
from pathlib import Path

pattern = r"/content/Data/*.csv"
files = glob.glob(pattern)

print("found =", len(files))
print("first 3 =", files[:3])

files_abs = [str(Path(p).resolve()).replace("\\", "/") for p in files]
print("first abs =", files_abs[0] if files_abs else None)

print("Current Working Directory:", os.getcwd())

found_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".csv"):
            found_files.append(os.path.join(root, file))

if found_files:
    print(f"Found {len(found_files)} CSV files:")
    for f in found_files[:10]:
        print(f)
    if len(found_files) > 10:
        print("... and more.")
else:
    print("No CSV files found in the current directory. You may need to upload them or mount Google Drive.")

schema = T.StructType([
    T.StructField("time_s", T.DoubleType(), True),
    T.StructField("co_ppm", T.DoubleType(), True),
    T.StructField("humidity_rh", T.DoubleType(), True),
    T.StructField("temp_c", T.DoubleType(), True),
    T.StructField("flow_ml_min", T.DoubleType(), True),
    T.StructField("heater_v", T.DoubleType(), True),
    *[T.StructField(f"R{i}", T.DoubleType(), True) for i in range(1,15)]
])

df = (spark.read
      .option("header","true")
      .schema(schema)
      .csv(files_abs)              # <-- εδώ
      .withColumn("file", F.input_file_name())
)

df.limit(5).show(truncate=False)

fname = F.regexp_extract(F.col("file"), r"(\d{8}_\d{6})", 1)
start_ts = F.to_timestamp(fname, "yyyyMMdd_HHmmss")

df_ts = (df
    .withColumn("start_ts", start_ts)
    .withColumn(
        "timestamp_raw",
        F.timestamp_millis(
            (F.unix_timestamp("start_ts") * 1000 + (F.col("time_s") * 1000)).cast("long")
        )
    )
)

df_ts.select("file","start_ts","time_s","timestamp_raw").show(5, truncate=False)

stack_expr = "stack(14, " + ", ".join([f"'R{i}', R{i}" for i in range(1,15)]) + ") as (sensor_id, resistance_mohm)"

df_long = df_ts.select(
    "timestamp_raw","co_ppm","humidity_rh","temp_c","flow_ml_min","heater_v",
    F.expr(stack_expr)
)

df_5m = (df_long
    .groupBy("sensor_id", F.window("timestamp_raw", "5 minutes").alias("w"))
    .agg(
        F.avg("co_ppm").alias("co_mean"),
        F.avg("humidity_rh").alias("rh_mean"),
        F.avg("temp_c").alias("temp_mean"),
        F.avg("flow_ml_min").alias("flow_mean"),
        F.avg("heater_v").alias("heater_v_mean"),
        F.avg("resistance_mohm").alias("res_mean"),
    )
    .withColumn("t5", F.col("w").start)
    .drop("w")
)

from pyspark.sql.window import Window
w = Window.partitionBy("sensor_id").orderBy("t5")
start_unix = F.unix_timestamp(F.lit("2025-09-01 00:00:00"))

df_norm = (df_5m
    .withColumn("idx", F.row_number().over(w) - 1)
    .withColumn("timestamp_norm", F.to_timestamp(F.from_unixtime(start_unix + F.col("idx")*300)))
    .drop("idx")
)

df_norm.select("sensor_id","t5","timestamp_norm","co_mean","rh_mean","heater_v_mean","res_mean") \
       .show(12, truncate=False)

cols_check = ["co_mean","rh_mean","temp_mean","heater_v_mean","res_mean"]

df_norm.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols_check]).show()

bad = df_norm.filter(
    (F.col("rh_mean") < 0) | (F.col("rh_mean") > 100) |
    (F.col("co_mean") < 0) |
    (F.col("heater_v_mean") <= 0) | (F.col("heater_v_mean") > 1.2) |
    (F.col("res_mean") <= 0)
)
print("bad rows:", bad.count())

"""Cleaning + Winsorization"""

vars_ = ["co_mean", "rh_mean", "temp_mean", "flow_mean", "heater_v_mean", "res_mean"]

df_clean = df_norm.filter(
    (F.col("rh_mean") >= 0) & (F.col("rh_mean") <= 100) &
    (F.col("co_mean") >= 0) &
    (F.col("temp_mean") >= -30) & (F.col("temp_mean") <= 80) &
    (F.col("flow_mean") > 0) &
    (F.col("heater_v_mean") > 0) & (F.col("heater_v_mean") <= 1.2) &
    (F.col("res_mean") > 0)
)

def winsorize(df, col, p_low=0.01, p_high=0.99, rel_err=0.01):
    lo, hi = df.approxQuantile(col, [p_low, p_high], rel_err)
    df2 = df.withColumn(
        col,
        F.when(F.col(col) < lo, lo)
         .when(F.col(col) > hi, hi)
         .otherwise(F.col(col))
    )
    return df2, lo, hi

bounds = {}
df_w = df_clean
for c in vars_:
    df_w, lo, hi = winsorize(df_w, c)
    bounds[c] = (lo, hi)

print("winsor bounds:", bounds)

import matplotlib.pyplot as plt

pdf = (df_w
       .select(["sensor_id", "timestamp_norm"] + vars_)
       .sample(False, 0.20, seed=42)
       .toPandas())

import numpy as np
import pandas as pd

os.makedirs("figures", exist_ok=True)

vars_ = ["co_mean", "rh_mean", "temp_mean", "flow_mean", "heater_v_mean", "res_mean"]

pdf = (df_w
       .select(["sensor_id", "timestamp_norm"] + vars_)
       .sample(False, 0.20, seed=42)
       .toPandas())

pdf["timestamp_norm"] = pd.to_datetime(pdf["timestamp_norm"])

"""### Boxplot Resistance ανα sensor"""

plt.figure()
pdf.boxplot(column="res_mean", by="sensor_id", grid=False, rot=45)
plt.title("Resistance by sensor")
plt.suptitle("")
plt.xlabel("Sensor ID")
plt.ylabel("Resistance (MΩ)")
plt.tight_layout()
plt.savefig("figures/box_res_by_sensor.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

"""### CO distribution (Histogram)"""

plt.figure()
plt.hist(pdf["co_mean"].dropna(), bins=30)
plt.title("CO concentration distribution")
plt.xlabel("CO concentration (ppm)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/hist_co_mean.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

"""### Temperature histogram"""

plt.figure()
plt.hist(pdf["temp_mean"].dropna(), bins=30)
plt.title("Temperature distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/hist_temp_mean.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

"""### Heater Voltage Histogram"""

plt.figure()
plt.hist(pdf["heater_v_mean"].dropna(), bins=30)
plt.title("Heater voltage distribution")
plt.xlabel("Heater Voltage (V)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/hist_heater_v_mean.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

"""### Scatter: CO vs Resistance για έναν αισθητήρα"""

sensor = "R10"
p = pdf[pdf["sensor_id"] == sensor].copy()

plt.figure()
plt.scatter(p["co_mean"], p["res_mean"], s=10)
plt.title(f"{sensor}: Resistance vs CO")
plt.xlabel("CO concentration (ppm)")
plt.ylabel("Resistance (MΩ)")
plt.tight_layout()
plt.savefig(f"figures/scatter_{sensor}_res_vs_co.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

"""### Scatter: Humidity vs Resistance για έναν αισθητήρα"""

plt.figure()
plt.scatter(p["rh_mean"], p["res_mean"], s=10)
plt.title(f"{sensor}: Resistance vs Humidity")
plt.xlabel("Relative Humidity (%RH)")
plt.ylabel("Resistance (MΩ)")
plt.tight_layout()
plt.savefig(f"figures/scatter_{sensor}_res_vs_rh.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

"""### (i) Τάση > 60% της μέγιστης και συνολική συγκέντρωση CO"""

max_v = df_ts.agg(F.max("heater_v").alias("max_v")).collect()[0]["max_v"]
thr = max_v * 0.6
print("max heater_v =", max_v)
print("threshold =", thr)

i_result = (df_ts
    .filter(F.col("heater_v") > thr)
    .agg(F.sum("co_ppm").alias("total_co_ppm"))
)
i_result.show()

i_debug = (df_ts
    .filter(F.col("heater_v") > thr)
    .agg(
        F.count("*").alias("n_rows"),
        F.avg("co_ppm").alias("avg_co_ppm"),
        F.sum("co_ppm").alias("sum_co_ppm"),
        F.max("co_ppm").alias("max_co_ppm")
    )
)
i_debug.show(truncate=False)

"""### (ii) Group by sensor_id: μέση τιμή + διακύμανση αντίστασης"""

ii_result = (df_long
    .groupBy("sensor_id")
    .agg(
        F.avg("resistance_mohm").alias("mean_res"),
        F.variance("resistance_mohm").alias("var_res")
    )
    .withColumn("sensor_n", F.regexp_extract("sensor_id", r"R(\d+)", 1).cast("int"))
    .orderBy("sensor_n")
    .drop("sensor_n")
)
ii_result.show(20, truncate=False)

"""### (iii) Accumulator: CO > 10 ppm ΚΑΙ RH < 30%"""

pattern = r"/content/Data/*.csv"
files = glob.glob(pattern)

files_abs = [os.path.abspath(f).replace("\\", "/") for f in files]
paths = ",".join([f"file:///{p}" for p in files_abs])

len(files_abs), files_abs[:2]

sc = spark.sparkContext
acc = sc.accumulator(0)

def update_partition(lines):
    c = 0
    for line in lines:
        # skip header
        if line.startswith("Time"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            co = float(parts[1])   # CO (ppm)
            rh = float(parts[2])   # Humidity (%r.h.)
        except:
            continue

        if (co > 10) and (rh < 30):   # αυστηρά όπως λέει η εκφώνηση
            c += 1

    acc.add(c)

rdd = sc.textFile(paths)
rdd.foreachPartition(update_partition)

print("Accumulator final value =", acc.value)

total_rows = rdd.filter(lambda line: not line.startswith("Time")).count()
print("Total data rows =", total_rows)
print("Condition count =", acc.value)
print("Percentage =", acc.value / total_rows * 100, "%")

"""### (iv) ReduceByKey: max R ανά heater voltage (RDD solution)"""

sc = spark.sparkContext
rdd = sc.textFile(paths)

def parse_line(line):
    if line.startswith("Time"):
        return None

    parts = line.split(",")
    # Heater voltage (V) => 6η στήλη => index 5
    if len(parts) <= 5:
        return None

    try:
        V = round(float(parts[5]), 2)  # Heater voltage (V)
        if V == 0:
            return None
        R = ((5.0 - V) / V) * 1_000_000.0
        return (V, R)  # (key, value)
    except:
        return None

pairs = rdd.map(parse_line).filter(lambda x: x is not None)

# reduceByKey για max R ανά V
maxR_by_V = pairs.reduceByKey(lambda a, b: a if a > b else b)

# Ταξινόμηση
result = maxR_by_V.sortByKey().collect()

print("heater_v\tmax_R")
for v, mr in result:
    print(f"{v:.2f}\t{mr}")

"""(v) 25 minute rolling window"""

sensor = "R10"

df_v = (
    df_norm
    .filter(F.col("sensor_id") == sensor)
    .groupBy(F.window("timestamp_norm", "25 minutes").alias("w"))
    .agg(
        F.avg("co_mean").alias("co_mean_25m"),
        F.avg("rh_mean").alias("rh_mean_25m"),
    )
    .select(
        F.col("w.start").alias("t_start"),
        F.col("w.end").alias("t_end"),
        "co_mean_25m", "rh_mean_25m"
    )
    .orderBy("t_start")
)

df_v.show(20, truncate=False)

"""## 2. FEATURE ENGINEERING

#### Spark to Pandas
"""

df_5m.printSchema()

df_norm = (df_5m
    .withColumn("idx", F.row_number().over(Window.partitionBy("sensor_id").orderBy("t5")) - 1)
    .withColumn("timestamp_norm",
                F.to_timestamp(F.from_unixtime(start_unix + F.col("idx")*300)))
    .drop("idx")
)

df_base = df_norm.select(
    "timestamp_norm", "t5", "sensor_id",
    "co_mean","rh_mean","temp_mean","flow_mean","heater_v_mean","res_mean"
)

df_wide = (df_base
    .groupBy("timestamp_norm")
    .pivot("sensor_id")
    .agg(F.first("res_mean"))
)

df_ctx = (df_base
    .groupBy("timestamp_norm")
    .agg(
        F.first("co_mean").alias("co_mean"),
        F.first("rh_mean").alias("rh_mean"),
        F.first("temp_mean").alias("temp_mean"),
        F.first("flow_mean").alias("flow_mean"),
        F.first("heater_v_mean").alias("heater_v_mean")
    )
)

df_final = df_ctx.join(df_wide, on="timestamp_norm", how="left").orderBy("timestamp_norm")
pdf = df_final.toPandas().set_index("timestamp_norm")

pdf.head()

pdf = pdf.copy()
pdf.index = pd.to_datetime(pdf.index)   # convert index to datetime
pdf = pdf.sort_index()
pdf = pdf.asfreq("5min")

import re

r_cols = sorted([c for c in pdf.columns if re.fullmatch(r"R\d+", c)],
                key=lambda x: int(x[1:]))

"""* Feature A - Διαφορά Θερμοκρασίας (ΔΤ)"""

pdf["temp_diff"] = pdf["temp_mean"].diff()

"""* Feature B - Heater Phase (Categorical variable low/high)

Κάνουμε threshold (=0.55)
"""

thr = pdf["heater_v_mean"].median()
pdf["heater_phase"] = np.where(pdf["heater_v_mean"] >= thr, "high", "low")

"""* Feature C - Μέσος Όρος των 14 αντιστάσεων (R_mean)"""

pdf["r_mean14"] = pdf[r_cols].mean(axis=1)

"""* Feature D - 5 minute rolling mean"""

pdf["co_roll_25m"] = pdf["co_mean"].rolling(window=5, min_periods=1).mean()

"""* Feature E - Sensor Variance"""

pdf["r_std14"] = pdf[r_cols].std(axis=1)

pdf["n_sensors"] = pdf[r_cols].notna().sum(axis=1)
bad = pdf[pdf["n_sensors"] == 0]
bad[["n_sensors","co_mean","rh_mean","temp_mean","heater_v_mean"]].head(30)

sensor_cols = [f"R{i}" for i in range(1, 15)]

# πόσοι αισθητήρες έχουν τιμή ανά t5
pdf["n_sensors"] = pdf[sensor_cols].notna().sum(axis=1)

pdf.loc[pdf["r_mean14"].isna(), ["n_sensors"] + sensor_cols[:14]].head(20)

"""#### σχέση μεταξύ του μ.ο. των αισθητήρων και ενός κυλιόμενου μ.ο. 25 λεπτών"""

segment = pdf.dropna(subset=["r_mean14","co_roll_25m"]).iloc[2000:2600].copy()

fig, ax1 = plt.subplots(figsize=(12,6))

ax1.plot(segment.index, segment["r_mean14"], color="tab:blue")
ax1.set_xlabel("Time")
ax1.set_ylabel("Mean resistance (MΩ)")

ax2 = ax1.twinx()
ax2.plot(segment.index, segment["co_roll_25m"], color="tab:red")
ax2.set_ylabel("CO rolling mean (ppm) [window=25 min]")

plt.title("Mean resistance (14 sensors) vs CO rolling mean (sample segment)")
plt.tight_layout()
plt.savefig("figures/mean_resistance_vs_co_roll_time.png", dpi=200, bbox_inches="tight")
plt.show()

print(segment.columns.tolist())
print(pdf.columns.tolist())

segment = pdf.iloc[2000:2600].copy()

plt.figure()
plt.scatter(segment["r_mean14"], segment["co_roll_25m"], s=10)
plt.xlabel("Mean Resistance (MΩ)")
plt.ylabel("CO rolling mean (ppm)")
plt.title("Mean Resistance vs CO rolling mean (Sample Segment)")
plt.tight_layout()
plt.show()

"""### Temporal Analysis

* Προετοιμασία Χρονοσειράς
"""

pdf = pdf.sort_index()

# Χρονοσειρά: μ.ο. 14 αισθητήρων
y = pdf["r_mean14"].copy()

# αν υπάρχουν NaNs
y = y.interpolate(method="time")

"""* 1) STL / Seasonal decomposition στο r_mean14 (με περίοδο heater)"""

from statsmodels.tsa.seasonal import STL

period = 11
stl = STL(y, period=period, robust=True)
res = stl.fit()

# 4 υποδιαγράμματα με κοινό άξονα χρόνου
fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

axes[0].plot(res.observed)
axes[0].set_title("Observed (r_mean14)")

axes[1].plot(res.trend)
axes[1].set_title("Trend")

axes[2].plot(res.seasonal)
axes[2].set_title(f"Seasonal (period={period} samples = {period*5} min)")

axes[3].plot(res.resid)
axes[3].set_title("Residuals")

for ax in axes:
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/stl_decomposition_r_mean14.png", dpi=200, bbox_inches="tight")
plt.show()

"""* 2) ACF & PACF των residuals με 95% confidence intervals"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

e = res.resid.dropna()
max_lag = 144   # 12 hours if sampling is 5 min

# ACF
fig = plot_acf(e, lags=max_lag, alpha=0.05, zero=False)
fig.set_size_inches(14, 4)
plt.xlabel("Lag (5-min steps)")
plt.ylabel("ACF")
plt.title("ACF of STL residuals (95% CI)")
plt.tight_layout()
plt.savefig("figures/acf_stl_residuals.png", dpi=200, bbox_inches="tight")
plt.show()

# PACF
fig = plot_pacf(e, lags=max_lag, alpha=0.05, zero=False, method="ywmle")
fig.set_size_inches(14, 4)
plt.xlabel("Lag (5-min steps)")
plt.ylabel("PACF")
plt.title("PACF of STL residuals (95% CI)")
plt.tight_layout()
plt.savefig("figures/pacf_stl_residuals.png", dpi=200, bbox_inches="tight")
plt.show()

"""* ποσοτικό αποτέλεσμα: ποια lags είναι significant"""

from statsmodels.tsa.stattools import acf, pacf

e = res.resid.dropna()
max_lag = 144

# ACF + 95% CI (alpha=0.05)
acf_vals, acf_ci = acf(e, nlags=max_lag, fft=True, alpha=0.05)

# PACF + 95% CI
pacf_vals, pacf_ci = pacf(e, nlags=max_lag, method="ywmle", alpha=0.05)

# “Significant” lag
sig_acf_lags  = np.where((acf_ci[:, 0] > 0) | (acf_ci[:, 1] < 0))[0]
sig_pacf_lags = np.where((pacf_ci[:, 0] > 0) | (pacf_ci[:, 1] < 0))[0]

# αγνοούμε lag 0
sig_acf_lags  = sig_acf_lags[sig_acf_lags != 0]
sig_pacf_lags = sig_pacf_lags[sig_pacf_lags != 0]

print("Significant ACF lags:", sig_acf_lags[:30], " ... total:", len(sig_acf_lags))
print("Significant PACF lags:", sig_pacf_lags[:30], " ... total:", len(sig_pacf_lags))

if len(sig_acf_lags) > 0:
    memory_minutes = sig_acf_lags.max() * 5
    print("Memory length (rough) =", memory_minutes, "minutes")

"""* 3) Έλεγχος στασιμότητας residuals (ADF test)"""

from statsmodels.tsa.stattools import adfuller

adf_stat, pval, used_lags, nobs, crit, icbest = adfuller(e, autolag="AIC")

print("ADF statistic:", adf_stat)
print("p-value:", pval)
print("used lags:", used_lags)
print("nobs:", nobs)
print("critical values:", crit)

"""#### MACHINE LEARNING MODEL"""

df_norm.select("sensor_id").distinct().orderBy("sensor_id").show(20, False)
df_norm.select("timestamp_norm","sensor_id","co_mean","heater_v_mean","res_mean").orderBy("timestamp_norm","sensor_id").show(5, False)
print("rows:", df_norm.count())

"""1-step target: “R του αισθητήρα στο επόμενο 5λεπτο”"""

w = Window.partitionBy("sensor_id").orderBy("timestamp_norm")

df_ml = (df_norm
    .withColumn("y_next", F.lead("res_mean", 1).over(w))   # target: next 5-min
)

# lag features
for L in [1,2,3,6,12]:
    df_ml = df_ml.withColumn(f"res_lag{L}", F.lag("res_mean", L).over(w))

df_ml = df_ml.dropna(subset=["y_next"] + [f"res_lag{L}" for L in [1,2,3,6,12]])

"""Time-based split 70/20/10 (ανά sensor_id)"""

time_col = "timestamp_norm"
w_order = Window.partitionBy("sensor_id").orderBy(time_col)
w_part  = Window.partitionBy("sensor_id")

df_split = (df_ml
    .withColumn("rn", F.row_number().over(w_order))
    .withColumn("n",  F.count("*").over(w_part))
    .withColumn("train_end", F.floor(F.col("n") * F.lit(0.70)))
    .withColumn("val_end",   F.floor(F.col("n") * F.lit(0.90)))
    .withColumn(
        "split",
        F.when(F.col("rn") <= F.col("train_end"), F.lit("train"))
         .when(F.col("rn") <= F.col("val_end"),   F.lit("val"))
         .otherwise(F.lit("test"))
    )
    .drop("rn","n","train_end","val_end")
)

#check counts
df_split.groupBy("sensor_id","split").count().orderBy("sensor_id","split").show(50, False)

train_df = df_split.filter(F.col("split")=="train").cache()
val_df   = df_split.filter(F.col("split")=="val").cache()
test_df  = df_split.filter(F.col("split")=="test").cache()

print("train:", train_df.count(), "val:", val_df.count(), "test:", test_df.count())

"""MLlib Pipeline + evaluation (ανά αισθητήρα)"""

import time
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

context_cols = ["co_mean","rh_mean","temp_mean","flow_mean","heater_v_mean"]
lag_cols = [f"res_lag{L}" for L in [1,2,3,6,12]]

feature_cols = context_cols + ["res_mean"] + lag_cols
label_col = "y_next"

eval_mae  = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")
eval_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
eval_r2   = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

def train_and_eval_for_sensor(sensor_id, algo="lr"):
    tr = train_df.filter(F.col("sensor_id")==sensor_id)
    te = test_df.filter(F.col("sensor_id")==sensor_id)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

    if algo == "lr":
        model = LinearRegression(featuresCol="features", labelCol=label_col, regParam=0.0, elasticNetParam=0.0)
    elif algo == "rf":
        model = RandomForestRegressor(featuresCol="features", labelCol=label_col, numTrees=120, maxDepth=12, seed=42)
    else:
        raise ValueError("algo must be 'lr' or 'rf'")

    pipe = Pipeline(stages=[assembler, scaler, model])

    t0 = time.time()
    fitted = pipe.fit(tr)
    train_time = time.time() - t0

    t1 = time.time()
    pred = fitted.transform(te).select(time_col, "sensor_id", F.col(label_col).alias("y"), F.col("prediction").alias("yhat"))
    pred.count()  # force execution για timing
    infer_time = time.time() - t1

    # Spark evaluators
    te_pred = fitted.transform(te)

    mae  = eval_mae.evaluate(te_pred)
    rmse = eval_rmse.evaluate(te_pred)
    r2   = eval_r2.evaluate(te_pred)

    # MAPE
    mape = (te_pred
        .select(F.col(label_col).alias("y"), F.col("prediction").alias("yhat"))
        .withColumn("den", F.when(F.abs(F.col("y")) < 1e-9, F.lit(1e-9)).otherwise(F.abs(F.col("y"))))
        .withColumn("ape", F.abs(F.col("yhat")-F.col("y")) / F.col("den"))
        .agg(F.avg("ape").alias("mape"))
        .first()["mape"]
    )

    return {
        "sensor_id": sensor_id,
        "model": algo,
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2,
        "train_s": train_time,
        "infer_s": infer_time,
        "pred_df": pred
    }

sensor_ids = [r["sensor_id"] for r in df_split.select("sensor_id").distinct().orderBy("sensor_id").collect()]

all_results = []
keep_pred_for = "R10"
pred_for_plot = {}

for algo in ["lr", "rf"]:
    for sid in sensor_ids:
        out = train_and_eval_for_sensor(sid, algo=algo)
        all_results.append({k:v for k,v in out.items() if k!="pred_df"})

        if sid == keep_pred_for:
            pred_for_plot[algo] = out["pred_df"]

res_df = pd.DataFrame(all_results)
res_df

table_per_sensor = (res_df
    .pivot(index="sensor_id", columns="model", values=["MAE","MAPE","RMSE","R2","train_s","infer_s"])
)
table_per_sensor

summary = (res_df
    .groupby("model")[["MAE","MAPE","RMSE","R2","train_s","infer_s"]]
    .agg(["mean","std"])
)
summary

for algo, spark_pred in pred_for_plot.items():
    pdf_pred = spark_pred.orderBy(time_col).toPandas()
    plt.figure(figsize=(14,5))
    plt.plot(pdf_pred[time_col], pdf_pred["y"], label="True")
    plt.plot(pdf_pred[time_col], pdf_pred["yhat"], label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Resistance (MΩ)")
    plt.title(f"True vs Predicted (1-step) for {keep_pred_for} [{algo}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/pred_vs_true_{keep_pred_for}_{algo}.png", dpi=200, bbox_inches="tight")
    plt.show()

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

pdf = pdf.sort_index()

sensor_cols = [f"R{i}" for i in range(1,15)]
context_cols = ["co_mean","rh_mean","temp_mean","flow_mean","heater_v_mean"]

# Features εισόδου
feat_cols = context_cols + sensor_cols

# Targets: next-step αντιστάσεις (14 outputs)
y_all = pdf[sensor_cols].shift(-1)

data = pdf[feat_cols].copy()
data["__ok__"] = 1

# κρατάμε κοινές γραμμές χωρίς NaN (και για X και για y)
mask = data[feat_cols].notna().all(axis=1) & y_all.notna().all(axis=1)
data = data.loc[mask, feat_cols]
y_all = y_all.loc[mask, sensor_cols]

print(data.shape, y_all.shape)

n = len(data)
i_train = int(n*0.70)
i_val   = int(n*0.90)

X_train_raw = data.iloc[:i_train].values
X_val_raw   = data.iloc[i_train:i_val].values
X_test_raw  = data.iloc[i_val:].values

y_train_raw = y_all.iloc[:i_train].values
y_val_raw   = y_all.iloc[i_train:i_val].values
y_test_raw  = y_all.iloc[i_val:].values

print(X_train_raw.shape, X_val_raw.shape, X_test_raw.shape)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_s = x_scaler.fit_transform(X_train_raw)
X_val_s   = x_scaler.transform(X_val_raw)
X_test_s  = x_scaler.transform(X_test_raw)

y_train_s = y_scaler.fit_transform(y_train_raw)
y_val_s   = y_scaler.transform(y_val_raw)
y_test_s  = y_scaler.transform(y_test_raw)

seq_len = 144  # 12 hours history (144*5min)

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i-1])  # target στο ίδιο index i
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

Xtr, ytr = make_sequences(X_train_s, y_train_s, seq_len)
Xva, yva = make_sequences(X_val_s, y_val_s, seq_len)
Xte, yte = make_sequences(X_test_s, y_test_s, seq_len)

print("Train seq:", Xtr.shape, ytr.shape)
print("Val seq:",   Xva.shape, yva.shape)
print("Test seq:",  Xte.shape, yte.shape)

tf.random.set_seed(42)

n_features = Xtr.shape[2]
n_outputs  = 14

model = models.Sequential([
    layers.Input(shape=(seq_len, n_features)),
    layers.GRU(64),
    layers.Dense(64, activation="relu"),
    layers.Dense(n_outputs)   # 14 outputs
])

model.compile(optimizer="adam", loss="mse")

cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

t0 = time.time()
hist = model.fit(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=20,
    batch_size=64,
    callbacks=[cb],
    verbose=1
)
train_time = time.time() - t0

t1 = time.time()
yhat_s = model.predict(Xte, verbose=0)  # scaled preds
infer_time = time.time() - t1

print("train_s:", train_time, "infer_s:", infer_time)

# invert scaling
yhat = y_scaler.inverse_transform(yhat_s)
ytrue = y_scaler.inverse_transform(yte)

eps = 1e-9

mae  = np.mean(np.abs(yhat - ytrue), axis=0)
rmse = np.sqrt(np.mean((yhat - ytrue)**2, axis=0))
mape = np.mean(np.abs((yhat - ytrue) / np.where(np.abs(ytrue) < eps, eps, np.abs(ytrue))), axis=0)

# R2 ανά αισθητήρα
ss_res = np.sum((ytrue - yhat)**2, axis=0)
ss_tot = np.sum((ytrue - np.mean(ytrue, axis=0))**2, axis=0)
r2 = 1 - ss_res / np.where(ss_tot < eps, eps, ss_tot)

rnn_res = pd.DataFrame({
    "sensor_id": sensor_cols,
    "MAE": mae,
    "MAPE": mape,
    "RMSE": rmse,
    "R2": r2
}).set_index("sensor_id")

# mean ± std row
mean_row = rnn_res.mean(axis=0)
std_row  = rnn_res.std(axis=0)

rnn_summary = pd.DataFrame({"mean": mean_row, "std": std_row})
print(rnn_summary)

rnn_res

sensor = "R10"
j = sensor_cols.index(sensor)

plt.figure(figsize=(14,5))
plt.plot(ytrue[:300, j], label="True")
plt.plot(yhat[:300, j], label="Pred")
plt.xlabel("Test time step (5min)")
plt.ylabel("Resistance (MΩ)")
plt.title(f"True vs Predicted (1-step) for {sensor} [GRU]")
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/pred_vs_true_{sensor}_GRU.png", dpi=200, bbox_inches="tight")
plt.show()

# baseline από το raw (unscaled) test slice:
test_idx = y_all.loc[mask].iloc[i_val:].index  # timestamps του test μετά το masking

# Οι yte αντιστοιχούν σε test_idx[seq_len:] περίπου
baseline_true = pdf.loc[test_idx, sensor_cols].iloc[seq_len:].values
baseline_pred = baseline_true.copy()

# για να σύγκριση με ytrue R(t+1):
baseline_pred = pdf.loc[test_idx, sensor_cols].iloc[seq_len:-1].values
baseline_true = pdf.loc[test_idx, sensor_cols].iloc[seq_len+1:].values

m = min(len(baseline_true), len(baseline_pred), len(ytrue))
baseline_true = baseline_true[:m]
baseline_pred = baseline_pred[:m]
ytrue_cut = ytrue[:m]
yhat_cut = yhat[:m]

def rmse(a,b): return np.sqrt(np.mean((a-b)**2, axis=0))

print("Baseline RMSE mean:", rmse(baseline_true, baseline_pred).mean())
print("GRU RMSE mean:", rmse(ytrue_cut, yhat_cut).mean())