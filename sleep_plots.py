import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

# File paths
BASE_DIR = "/srv/repos/raddlab_datascience/cingo-db-data-extractor/output"
ACCEL_CSV = os.path.join(BASE_DIR, "iwellnt_light_2026-03-27_to_2026-06-27.csv")
SURVEY_CSV = os.path.join(BASE_DIR, "iwellnt_survey_data_2026-03-27_to_2026-06-27.csv")
output_file = "light_sample_coverage_iwellnt.png"
TIMEZONE = "America/Denver"

# ------------------------------------------------------------------
# Load and process acceleration/device data
# ------------------------------------------------------------------
accel_df = pd.read_csv(ACCEL_CSV)

event_col = "event_type" if "event_type" in accel_df.columns else accel_df.columns[2]
json_col = "payload" if "payload" in accel_df.columns else accel_df.columns[3]

has_device_usage = (accel_df[event_col] == "SK Device Usage").any()
has_battery_status = (accel_df[event_col] == "Battery Status").any()

# Initialize data structures for localized plotting conditions
plugged_in_df = pd.DataFrame()
unplugged_df = pd.DataFrame()

# Scenario A: Handle Device Unlocks
if has_device_usage:
    print("Found 'SK Device Usage' data. Filtering for active unlocks...")
    accel_df = accel_df[accel_df[event_col] == "SK Device Usage"].copy()
    
    def extract_unlocks(json_str):
        try:
            return int(json.loads(json_str).get("total_unlocks", 0))
        except Exception:
            return 0

    accel_df["total_unlocks"] = accel_df[json_col].apply(extract_unlocks)
    accel_df = accel_df[accel_df["total_unlocks"] > 0].copy()
    
    plot_title = f"Device Unlock Events & Self-Reported Sleep ({TIMEZONE})"
    outside_label, inside_label = "Device Unlocks (Outside Window)", "Device Unlocks (Inside Window)"
    marker_size_out, marker_size_in = 25, 35
    marker_alpha_out, marker_alpha_in = 0.6, 0.8

# Scenario B: Handle Battery State Changes (Plugged In / Unplugged)
elif has_battery_status:
    print("Found 'Battery Status' data. Isolating transition milestones...")
    accel_df = accel_df[accel_df[event_col] == "Battery Status"].copy()
    
    # Chronological sort ensures correct chronological delta calculation
    accel_df["captured_at_dt"] = pd.to_datetime(accel_df["captured_at"], format="mixed", utc=True)
    accel_df = accel_df.sort_values("captured_at_dt")
    
    def extract_charging_status(json_str):
        try:
            data = json.loads(json_str)
            return bool(data.get("batteryIsCharging", data.get("battery_is_charging", False)))
        except Exception:
            return False

    accel_df["is_charging"] = accel_df[json_col].apply(extract_charging_status)
    accel_df["state_changed"] = accel_df["is_charging"].diff().fillna(False)
    
    # Keep only target transition instances
    accel_df = accel_df[accel_df["state_changed"] == True].copy()
    
    plot_title = f"Battery Charging Transitions & Self-Reported Sleep ({TIMEZONE})"

# Scenario C: Fallback to Raw Pings
else:
    print("No target telemetry found. Proceeding with raw ping coverage mapping...")
    plot_title = f"Acceleration Sample Coverage & Self-Reported Sleep ({TIMEZONE})"
    outside_label, inside_label = "Outside Sleep Window", "Inside Sleep Window"
    marker_size_out, marker_size_in = 3, 12
    marker_alpha_out, marker_alpha_in = 0.3, 0.7

# Process timestamps for whatever valid data remains after filtering strategies above
if not accel_df.empty:
    accel_df["captured_at"] = pd.to_datetime(accel_df["captured_at"], format="mixed", utc=True)
    accel_df["local_time"] = accel_df["captured_at"].dt.tz_convert(TIMEZONE)
    accel_df["date"] = accel_df["local_time"].dt.normalize()
    accel_df["hour"] = (
        accel_df["local_time"].dt.hour
        + accel_df["local_time"].dt.minute / 60
        + accel_df["local_time"].dt.second / 3600
    )

    if has_battery_status:
        # Segment processed battery dataset into dedicated directional frames
        plugged_in_df = accel_df[accel_df["is_charging"] == True]
        unplugged_df = accel_df[accel_df["is_charging"] == False]
    else:
        accel_df["in_sleep_window"] = (accel_df["hour"] >= 18) | (accel_df["hour"] < 14)
        inside = accel_df[accel_df["in_sleep_window"]]
        outside = accel_df[~accel_df["in_sleep_window"]]

# ------------------------------------------------------------------
# Load and process survey data
# ------------------------------------------------------------------
survey_df = pd.read_csv(SURVEY_CSV)
survey_df["completed_at"] = pd.to_datetime(survey_df["completed_at"], format="mixed", utc=True)
survey_df["local_date"] = survey_df["completed_at"].dt.tz_convert(TIMEZONE).dt.normalize()

def extract_time_string(json_str, key):
    try:
        data = json.loads(json_str)
        if isinstance(data, list) and len(data) > 0:
            return data[0].get(key)
    except Exception:
        return None

def time_to_hours(time_str):
    if not isinstance(time_str, str) or not time_str:
        return None
    h, m = map(int, time_str.split(":"))
    return h + m / 60.0

survey_df["bedtime_str"] = survey_df.apply(lambda r: extract_time_string(r["responses"], "bed_time_military") if r["check_in_type"] == "bedtime" else None, axis=1)
survey_df["wakeup_str"] = survey_df.apply(lambda r: extract_time_string(r["responses"], "wake_up_time_military") if r["check_in_type"] == "wake-up" else None, axis=1)

survey_df["bedtime_hour"] = survey_df["bedtime_str"].apply(time_to_hours)
survey_df["wakeup_hour"] = survey_df["wakeup_str"].apply(time_to_hours)

daily_sleep = survey_df.groupby("local_date").agg({
    "bedtime_hour": "first",
    "wakeup_hour": "last"
}).dropna(how="all").reset_index()

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(22, 8))

# Shade regions corresponding to general sleep-analysis window
ax.axhspan(0, 14, alpha=0.05, color="gray", label="Analysis Window (00:00-14:00)", zorder=1)
ax.axhspan(18, 24, alpha=0.05, color="gray", label="Analysis Window (18:00-24:00)", zorder=1)

# Plot actual self-reported sleep intervals as vertical bars
for _, row in daily_sleep.iterrows():
    survey_date = row["local_date"]
    if pd.notna(row["wakeup_hour"]):
        ax.vlines(x=survey_date, ymin=0, ymax=row["wakeup_hour"], colors="crimson", linewidth=4, alpha=0.8, zorder=2,
                  label="Self-Reported Sleep" if "Self-Reported Sleep" not in ax.get_legend_handles_labels()[1] else "")
    if pd.notna(row["bedtime_hour"]):
        bedtime_date = survey_date - pd.Timedelta(days=1)
        ax.vlines(x=bedtime_date, ymin=row["bedtime_hour"], ymax=24, colors="crimson", linewidth=4, alpha=0.8, zorder=2,
                  label="Self-Reported Sleep" if "Self-Reported Sleep" not in ax.get_legend_handles_labels()[1] else "")

# Render data based on telemetry context discovered above
if has_battery_status:
    # Plot Plugged In events (Green upward triangles)
    if not plugged_in_df.empty:
        ax.scatter(
            plugged_in_df["date"], plugged_in_df["hour"],
            marker="^", s=70, color="forestgreen", alpha=0.9, zorder=3,
            label="Plugged In (Charging Started)"
        )
    # Plot Unplugged events (Red downward triangles)
    if not unplugged_df.empty:
        ax.scatter(
            unplugged_df["date"], unplugged_df["hour"],
            marker="v", s=70, color="firebrick", alpha=0.9, zorder=3,
            label="Unplugged (Charging Stopped)"
        )
else:
    # Scatter layouts for Standard Device Unlocks or Basic Ping Coverages
    if not accel_df.empty:
        ax.scatter(outside["date"], outside["hour"], s=marker_size_out, alpha=marker_alpha_out, color="tab:blue", zorder=3, label=outside_label)
        ax.scatter(inside["date"], inside["hour"], s=marker_size_in, alpha=marker_alpha_in, color="tab:orange", zorder=3, label=inside_label)

# X-axis formatting
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
plt.xticks(rotation=90)

# Labels & Limits
ax.set_ylabel("Hour of Day (Local Time)")
ax.set_xlabel("Date")
ax.set_title(plot_title)
ax.set_yticks(range(0, 25, 2))
ax.set_ylim(0, 24)
ax.grid(True, alpha=0.3, zorder=0)

# Layout and Legend Management
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.subplots_adjust(right=0.82)
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.)

plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved plot to: {output_file}")
