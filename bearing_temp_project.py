import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
FREQ = "10min"              # sampling interval
MASK_RANDOM_PCT = 0.05      # 5% random missing
DROPOUT_PCT = 0.10          # ~10% missing as realistic dropouts
FFILL_LIMIT_MIN = 30        # forward fill limit (minutes)
FFILL_LIMIT_POINTS = FFILL_LIMIT_MIN // 10  # 30min / 10min = 3 points

# Now we read from CSV
CSV_PATH = "bearing_temp_6months_10min.csv"

np.random.seed(SEED)

# -----------------------------
# 1) LOAD DATA FROM CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

# Ensure uniform 10-minute index (handles any gaps / duplicates safely)
df = df.resample(FREQ).mean()

if "bearing_temp" not in df.columns:
    raise ValueError("CSV must contain 'bearing_temp' column.")

# Keep a copy of original
df_original = df.copy()

# -----------------------------
# 2) MASK 5% RANDOM MISSING DATA
# -----------------------------
n = len(df)
mask_random_count = int(MASK_RANDOM_PCT * n)

random_missing_idx = np.random.choice(df.index, size=mask_random_count, replace=False)
df.loc[random_missing_idx, "bearing_temp"] = np.nan

# -----------------------------
# 3) CREATE ~10% "DROPOUT" MISSING DATA (BLOCKS)
# -----------------------------
dropout_target = int(DROPOUT_PCT * n)

dropout_missing = set()
attempts = 0
while len(dropout_missing) < dropout_target and attempts < 20000:
    attempts += 1

    start_pos = np.random.randint(0, n - 1)
    block_len = np.random.randint(2, 25)  # 20 min to 4 hours (2 to 24 points)

    block_positions = range(start_pos, min(start_pos + block_len, n))
    for pos in block_positions:
        dropout_missing.add(df.index[pos])
        if len(dropout_missing) >= dropout_target:
            break

df.loc[list(dropout_missing), "bearing_temp"] = np.nan

# -----------------------------
# 4) FORWARD FILL WITH LIMIT = 30 MINUTES (3 POINTS)
# -----------------------------
df_ffill = df.copy()
df_ffill["bearing_temp_ffill_30m"] = df_ffill["bearing_temp"].ffill(limit=FFILL_LIMIT_POINTS)

# -----------------------------
# 5) REPORT MISSING STATS
# -----------------------------
def missing_stats(series: pd.Series, label: str):
    total = len(series)
    missing = int(series.isna().sum())
    pct = (missing / total) * 100
    print(f"{label}: missing={missing}/{total} ({pct:.2f}%)")

print("\n--- Missing Data Summary ---")
missing_stats(df_original["bearing_temp"], "Original")
missing_stats(df["bearing_temp"], "After masking (5% random + ~10% dropouts)")
missing_stats(df_ffill["bearing_temp_ffill_30m"], "After ffill(limit=30min)")

filled = df["bearing_temp"].isna() & df_ffill["bearing_temp_ffill_30m"].notna()
print(f"Filled points by bounded ffill: {int(filled.sum())}")

still_missing = df_ffill["bearing_temp_ffill_30m"].isna().sum()
print(f"Still missing after bounded ffill (gap > 30min): {int(still_missing)}")

# -----------------------------
# 6) BASIC ANALYSIS: SMOOTHING + ANOMALY FLAG (EXAMPLE)
# -----------------------------
df_ffill["temp_roll_1h"] = df_ffill["bearing_temp_ffill_30m"].rolling(6, min_periods=3).mean()

threshold = 75
df_ffill["anomaly_flag"] = df_ffill["bearing_temp_ffill_30m"] > threshold

# -----------------------------
# 7) PLOTS
# -----------------------------
window_days = 7
end_time = df_ffill.index.max()
start_time = end_time - pd.Timedelta(days=window_days)
view = df_ffill.loc[start_time:end_time].copy()

plt.figure(figsize=(12, 5))
plt.plot(view.index, view["bearing_temp"], label="With Missing (masked)")
plt.plot(view.index, view["bearing_temp_ffill_30m"], label="FFill (30-min limit)")
plt.title(f"Bearing Temperature - Last {window_days} Days (10-min data)")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(view.index, view["bearing_temp_ffill_30m"], label="FFill data")
plt.plot(view.index, view["anomaly_flag"].astype(int), label="Anomaly flag (1=above threshold)")
plt.title("Anomaly Indicator (Example)")
plt.xlabel("Time")
plt.tight_layout()
plt.legend()
plt.show()

# -----------------------------
# 8) EXPORT CLEANED DATA
# -----------------------------
df_ffill.to_csv("bearing_temp_cleaned_ffill_30m.csv")
print("\nSaved: bearing_temp_cleaned_ffill_30m.csv")
print("Done.")
