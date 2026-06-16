import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

ACCEL_CSV = "/srv/repos/raddlab_datascience/cingo-db-data-extractor/output/iwellnt_acceleration_2026-03-27_to_2026-06-15.csv"
output_file = "acceleration_sample_coverage_iwellnt.png"
TIMEZONE = "America/Denver"

# ------------------------------------------------------------------
# Load acceleration data
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

# ------------------------------------------------------------------
# Create plotting columns
# ------------------------------------------------------------------
accel_df["date"] = accel_df["local_time"].dt.normalize()

accel_df["hour"] = (
    accel_df["local_time"].dt.hour
    + accel_df["local_time"].dt.minute / 60
    + accel_df["local_time"].dt.second / 3600
)

# ------------------------------------------------------------------
# Sleep analysis window used in original code:
#
# Previous day 18:00 -> Current day 14:00
#
# Hour-wise:
# 18:00-24:00 OR 00:00-14:00
# ------------------------------------------------------------------
accel_df["in_sleep_window"] = (
    (accel_df["hour"] >= 18)
    | (accel_df["hour"] < 14)
)

inside = accel_df[accel_df["in_sleep_window"]]
outside = accel_df[~accel_df["in_sleep_window"]]

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(figsize=(20, 8))

# Shade regions corresponding to sleep-analysis window
plt.axhspan(
    0,
    14,
    alpha=0.10,
)

plt.axhspan(
    18,
    24,
    alpha=0.10,
)

# Outside sleep window
plt.scatter(
    outside["date"],
    outside["hour"],
    s=3,
    alpha=0.3,
    label="Outside Sleep Window",
)

# Inside sleep window
plt.scatter(
    inside["date"],
    inside["hour"],
    s=12,
    alpha=0.9,
    label="Inside Sleep Window",
)

# ------------------------------------------------------------------
# X-axis formatting
# ------------------------------------------------------------------
ax = plt.gca()

ax.xaxis.set_major_locator(
    mdates.DayLocator(interval=1)
)

ax.xaxis.set_major_formatter(
    DateFormatter("%m/%d")
)

plt.xticks(rotation=90)

# ------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------
plt.ylabel("Hour of Day (Local Time)")
plt.xlabel("Date")
plt.title(
    f"Acceleration Sample Coverage ({TIMEZONE})"
)

plt.yticks(range(0, 25, 2))
plt.ylim(0, 24)

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
# Summary statistics
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total samples: {len(accel_df):,}")

print(
    f"Inside sleep window: "
    f"{accel_df['in_sleep_window'].sum():,}"
)

print(
    f"Outside sleep window: "
    f"{(~accel_df['in_sleep_window']).sum():,}"
)

print("\nSamples per day:")

daily_counts = (
    accel_df.groupby(accel_df["date"].dt.strftime("%m/%d"))
    .size()
    .reset_index(name="num_samples")
)

print(daily_counts.to_string(index=False))
