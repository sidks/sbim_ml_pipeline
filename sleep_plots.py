import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

# File paths
BASE_DIR = "/srv/repos/raddlab_datascience/cingo-db-data-extractor/output"
ACCEL_CSV = os.path.join(BASE_DIR, "iwellnt_acceleration_2026-03-27_to_2026-06-15.csv")
SURVEY_CSV = os.path.join(BASE_DIR, "iwellnt_survey_data_2026-03-27_to_2026-06-15.csv")
output_file = "acceleration_sample_coverage_iwellnt.png"
TIMEZONE = "America/Denver"

# ------------------------------------------------------------------
# Load and process acceleration data
# ------------------------------------------------------------------
accel_df = pd.read_csv(ACCEL_CSV)
accel_df["captured_at"] = pd.to_datetime(accel_df["captured_at"], format="mixed", utc=True)
accel_df["local_time"] = accel_df["captured_at"].dt.tz_convert(TIMEZONE)
accel_df["date"] = accel_df["local_time"].dt.normalize()
accel_df["hour"] = (
    accel_df["local_time"].dt.hour
    + accel_df["local_time"].dt.minute / 60
    + accel_df["local_time"].dt.second / 3600
)

accel_df["in_sleep_window"] = (accel_df["hour"] >= 18) | (accel_df["hour"] < 14)
inside = accel_df[accel_df["in_sleep_window"]]
outside = accel_df[~accel_df["in_sleep_window"]]

# ------------------------------------------------------------------
# Load and process survey data
# ------------------------------------------------------------------
survey_df = pd.read_csv(SURVEY_CSV)

# Convert completed_at to the local timezone date to properly index the sleep reports
survey_df["completed_at"] = pd.to_datetime(survey_df["completed_at"], format="mixed", utc=True)
survey_df["local_date"] = survey_df["completed_at"].dt.tz_convert(TIMEZONE).dt.normalize()

# Helper function to parse JSON responses safely
def extract_time_string(json_str, key):
    try:
        data = json.loads(json_str)
        if isinstance(data, list) and len(data) > 0:
            return data[0].get(key)
    except Exception:
        return None

def time_to_hours(time_str):
    if not time_str:
        return None
    h, m = map(int, time_str.split(":"))
    return h + m / 60.0

# Extract specific survey response hours
survey_df["bedtime_str"] = survey_df.apply(lambda r: extract_time_string(r["responses"], "bed_time_military") if r["check_in_type"] == "bedtime" else None, axis=1)
survey_df["wakeup_str"] = survey_df.apply(lambda r: extract_time_string(r["responses"], "wake_up_time_military") if r["check_in_type"] == "wake-up" else None, axis=1)

survey_df["bedtime_hour"] = survey_df["bedtime_str"].apply(time_to_hours)
survey_df["wakeup_hour"] = survey_df["wakeup_str"].apply(time_to_hours)

# Group by the day the survey was completed to line up bedtime ("last night") and wakeup ("this morning")
daily_sleep = survey_df.groupby("local_date").agg({
    "bedtime_hour": "first",
    "wakeup_hour": "last"
}).dropna(how="all").reset_index()

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(figsize=(20, 8))

# Shade regions corresponding to general sleep-analysis window
plt.axhspan(0, 14, alpha=0.05, color="gray", label="Analysis Window (00:00-14:00)")
plt.axhspan(18, 24, alpha=0.05, color="gray", label="Analysis Window (18:00-24:00)")

# Plot actual self-reported sleep intervals as vertical bars
# We map them to the correct days based on when the sleep actually occurred
for _, row in daily_sleep.iterrows():
    survey_date = row["local_date"]
    
    # 1. Post-midnight sleep: From 00:00 to wake up time on the survey completion day
    if pd.notna(row["wakeup_hour"]):
        plt.vlines(
            x=survey_date, 
            ymin=0, 
            ymax=row["wakeup_hour"], 
            colors="crimson", 
            linewidth=4, 
            alpha=0.8,
            label="Self-Reported Sleep" if "Self-Reported Sleep" not in plt.gca().get_legend_handles_labels()[1] else ""
        )
        
    # 2. Pre-midnight sleep: From bedtime to 24:00 on the *previous* night
    if pd.notna(row["bedtime_hour"]):
        bedtime_date = survey_date - pd.Timedelta(days=1)
        plt.vlines(
            x=bedtime_date, 
            ymin=row["bedtime_hour"], 
            ymax=24, 
            colors="crimson", 
            linewidth=4, 
            alpha=0.8,
            label="Self-Reported Sleep" if "Self-Reported Sleep" not in plt.gca().get_legend_handles_labels()[1] else ""
        )

# Outside sleep window points
plt.scatter(
    outside["date"],
    outside["hour"],
    s=3,
    alpha=0.3,
    color="tab:blue",
    label="Outside Sleep Window",
)

# Inside sleep window points
plt.scatter(
    inside["date"],
    inside["hour"],
    s=12,
    alpha=0.7,
    color="tab:orange",
    label="Inside Sleep Window",
)

# ------------------------------------------------------------------
# X-axis formatting
# ------------------------------------------------------------------
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
plt.xticks(rotation=90)

# ------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------
plt.ylabel("Hour of Day (Local Time)")
plt.xlabel("Date")
plt.title(f"Acceleration Sample Coverage & Self-Reported Sleep ({TIMEZONE})")

plt.yticks(range(0, 25, 2))
plt.ylim(0, 24)
plt.grid(True, alpha=0.3)

# Deduplicate legend items
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper right")

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot to: {output_file}")
