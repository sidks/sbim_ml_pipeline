import pandas as pd
import matplotlib.pyplot as plt

ACCEL_CSV = "/srv/repos/raddlab_datascience/cingo-db-data-extractor/output/test7_acceleration_2026-03-27_to_2026-06-15.csv"
output_file = "acceleration_sample_coverage_test7.png"
TIMEZONE = "America/Denver"

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
accel_df = pd.read_csv(ACCEL_CSV)

accel_df["captured_at"] = pd.to_datetime(
    accel_df["captured_at"],
    format="mixed",
    utc=True,
)

# Convert UTC -> participant local timezone
accel_df["local_time"] = (
    accel_df["captured_at"]
    .dt.tz_convert(TIMEZONE)
)

# Date and hour for plotting
accel_df["date"] = accel_df["local_time"].dt.date

accel_df["hour"] = (
    accel_df["local_time"].dt.hour
    + accel_df["local_time"].dt.minute / 60
    + accel_df["local_time"].dt.second / 3600
)

# ------------------------------------------------------------------
# Sleep window used by the algorithm:
#
# Previous day 18:00 -> current day 14:00
#
# On a per-hour basis this means:
# 18:00-24:00
# OR
# 00:00-14:00
# ------------------------------------------------------------------
accel_df["in_sleep_window"] = (
    (accel_df["hour"] >= 18)
    | (accel_df["hour"] < 14)
)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(figsize=(16, 8))

# Outside sleep window
outside = accel_df[~accel_df["in_sleep_window"]]

plt.scatter(
    outside["date"],
    outside["hour"],
    s=6,
    alpha=0.7,
    label="Outside Sleep Window (14:00-18:00)",
)

# Inside sleep window
inside = accel_df[accel_df["in_sleep_window"]]

plt.scatter(
    inside["date"],
    inside["hour"],
    s=6,
    alpha=0.8,
    label="Inside Sleep Window (18:00-14:00)",
)

# Shade sleep-analysis regions
plt.axhspan(
    0,
    14,
    alpha=0.12,
)

plt.axhspan(
    18,
    24,
    alpha=0.12,
)

plt.ylabel("Hour of Day (Local Time)")
plt.xlabel("Date")
plt.title("Acceleration Sample Coverage")

plt.ylim(0, 24)

plt.yticks(range(0, 25, 2))

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved plot to: {output_file}")

# ------------------------------------------------------------------
# Useful summary stats
# ------------------------------------------------------------------
print("\nSummary")
print("-" * 50)

print(f"Total samples: {len(accel_df):,}")

print(
    f"Inside sleep window: "
    f"{accel_df['in_sleep_window'].sum():,}"
)

print(
    f"Outside sleep window: "
    f"{(~accel_df['in_sleep_window']).sum():,}"
)

daily_counts = (
    accel_df.groupby("date")
    .size()
    .reset_index(name="num_samples")
)

print("\nSamples per day:")
print(daily_counts.to_string(index=False))
