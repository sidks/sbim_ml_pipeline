import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

# File paths
BASE_DIR = "/srv/repos/raddlab_datascience/cingo-db-data-extractor/output"
# This can now be test7_light_..., test7_acceleration_..., or test7_device_usage_...
ACCEL_CSV = os.path.join(BASE_DIR, "test7_device_usage_2026-03-27_to_2026-06-27.csv")
SURVEY_CSV = os.path.join(BASE_DIR, "test7_survey_data_2026-03-27_to_2026-06-27.csv")
TIMEZONE = "America/Denver"

# ------------------------------------------------------------------
# Dynamic Data & Output Mapping (Parsed from CSV Name)
# ------------------------------------------------------------------
csv_filename = os.path.basename(ACCEL_CSV)
filename_parts = csv_filename.split("_")

participant_prefix = filename_parts[0]  # Extracts 'test7', 'iwellnt', etc.
data_type = filename_parts[1]           # Extracts 'light', 'acceleration', or 'device'

# Clean up 'device_usage' name mapping if 'device' is extracted
if data_type == "device":
    data_type = "device_usage"

# Standard target output names built dynamically
output_file_main = f"{data_type}_sample_coverage_{participant_prefix}.png"
output_file_battery = f"battery_sample_coverage_{participant_prefix}.png"

print(f"Detected Participant: {participant_prefix} | Data Type: {data_type}")

# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------
raw_df = pd.read_csv(ACCEL_CSV)

event_col = "event_type" if "event_type" in raw_df.columns else raw_df.columns[2]
json_col = "payload" if "payload" in raw_df.columns else raw_df.columns[3]

# ------------------------------------------------------------------
# Load and process survey data (Shared by all plots)
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


# Helper to convert raw dataframe records to structured times
def format_timestamps(df):
    df["captured_at"] = pd.to_datetime(df["captured_at"], format="mixed", utc=True)
    df["local_time"] = df["captured_at"].dt.tz_convert(TIMEZONE)
    df["date"] = df["local_time"].dt.normalize()
    df["hour"] = (
        df["local_time"].dt.hour
        + df["local_time"].dt.minute / 60
        + df["local_time"].dt.second / 3600
    )
    return df

# Helper to draw the base plot background and survey vertical bars
def create_base_plot(title_text):
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.axhspan(0, 14, alpha=0.05, color="gray", label="Analysis Window (00:00-14:00)", zorder=1)
    ax.axhspan(18, 24, alpha=0.05, color="gray", label="Analysis Window (18:00-24:00)", zorder=1)

    for _, row in daily_sleep.iterrows():
        survey_date = row["local_date"]
        if pd.notna(row["wakeup_hour"]):
            ax.vlines(x=survey_date, ymin=0, ymax=row["wakeup_hour"], colors="crimson", linewidth=4, alpha=0.8, zorder=2,
                      label="Self-Reported Sleep" if "Self-Reported Sleep" not in ax.get_legend_handles_labels()[1] else "")
        if pd.notna(row["bedtime_hour"]):
            bedtime_date = survey_date - pd.Timedelta(days=1)
            ax.vlines(x=bedtime_date, ymin=row["bedtime_hour"], ymax=24, colors="crimson", linewidth=4, alpha=0.8, zorder=2,
                      label="Self-Reported Sleep" if "Self-Reported Sleep" not in ax.get_legend_handles_labels()[1] else "")
    
    ax.set_ylabel("Hour of Day (Local Time)")
    ax.set_xlabel("Date")
    ax.set_title(title_text)
    ax.set_yticks(range(0, 25, 2))
    ax.set_ylim(0, 24)
    ax.grid(True, alpha=0.3, zorder=0)
    return fig, ax

# Helper to save and handle legends safely
def finalize_and_save_plot(fig, ax, filename):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
    plt.xticks(rotation=90)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.subplots_adjust(right=0.82)
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.)
    
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {filename}")


# ==================================================================
# EXECUTION ROUTINE
# ==================================================================

# CONDITION 1: Dataset explicitly targets Device Usage telemetry
if data_type == "device_usage" and (raw_df[event_col] == "SK Device Usage").any():
    print("Processing active device unlock interactions...")
    
    # --- 1. DEVICE UNLOCKS PLOT ---
    fig, ax = create_base_plot(f"Device Unlock Events & Self-Reported Sleep ({TIMEZONE})")
    accel_df_usage = raw_df[raw_df[event_col] == "SK Device Usage"].copy()
    
    def extract_unlocks(json_str):
        try: return int(json.loads(json_str).get("total_unlocks", 0))
        except: return 0

    accel_df_usage["total_unlocks"] = accel_df_usage[json_col].apply(extract_unlocks)
    accel_df_usage = accel_df_usage[accel_df_usage["total_unlocks"] > 0].copy()
    
    if not accel_df_usage.empty:
        accel_df_usage = format_timestamps(accel_df_usage)
        accel_df_usage["in_sleep_window"] = (accel_df_usage["hour"] >= 18) | (accel_df_usage["hour"] < 14)
        inside = accel_df_usage[accel_df_usage["in_sleep_window"]]
        outside = accel_df_usage[~accel_df_usage["in_sleep_window"]]
        
        ax.scatter(outside["date"], outside["hour"], s=25, alpha=0.6, color="tab:blue", zorder=3, label="Device Unlocks (Outside Window)")
        ax.scatter(inside["date"], inside["hour"], s=35, alpha=0.8, color="tab:orange", zorder=3, label="Device Unlocks (Inside Window)")
    
    finalize_and_save_plot(fig, ax, output_file_main)

    # --- 2. BATTERY TREND PLOT ---
    print("Processing battery delta trends from device usage JSON...")
    fig, ax = create_base_plot(f"Battery Level Trends from Device Usage & Self-Reported Sleep ({TIMEZONE})")
    accel_df_battery = raw_df[raw_df[event_col] == "SK Device Usage"].copy()
    
    accel_df_battery["captured_at_dt"] = pd.to_datetime(accel_df_battery["captured_at"], format="mixed", utc=True)
    accel_df_battery = accel_df_battery.sort_values("captured_at_dt")
    
    def extract_battery_level(json_str):
        try: return float(json.loads(json_str).get("battery_level", 0.0))
        except: return None

    accel_df_battery["battery_level"] = accel_df_battery[json_col].apply(extract_battery_level)
    accel_df_battery = accel_df_battery.dropna(subset=["battery_level"]).copy()
    
    accel_df_battery["level_delta"] = accel_df_battery["battery_level"].diff()
    accel_df_battery = accel_df_battery.dropna(subset=["level_delta"]).copy()
    
    if not accel_df_battery.empty:
        accel_df_battery = format_timestamps(accel_df_battery)
        
        charging_df = accel_df_battery[accel_df_battery["level_delta"] > 0]
        discharging_df = accel_df_battery[accel_df_battery["level_delta"] < 0]
        constant_df = accel_df_battery[accel_df_battery["level_delta"] == 0]
        
        if not constant_df.empty:
            ax.scatter(constant_df["date"], constant_df["hour"], s=15, color="darkgray", alpha=0.4, zorder=3, label="Battery Constant")
        if not charging_df.empty:
            ax.scatter(charging_df["date"], charging_df["hour"], s=35, color="forestgreen", alpha=0.8, zorder=4, label="Battery Increasing (Charging)")
        if not discharging_df.empty:
            ax.scatter(discharging_df["date"], discharging_df["hour"], s=35, color="firebrick", alpha=0.8, zorder=4, label="Battery Decreasing (In Use)")
            
    finalize_and_save_plot(fig, ax, output_file_battery)

# CONDITION 2: Dataset targets Light, Acceleration, or fallbacks
else:
    title_label = data_type.replace("_", " ").title()
    print(f"Processing raw timestamp coverage mapping for: {title_label} data...")
    
    fig, ax = create_base_plot(f"{title_label} Sample Coverage & Self-Reported Sleep ({TIMEZONE})")
    
    if not raw_df.empty:
        fallback_df = format_timestamps(raw_df.copy())
        fallback_df["in_sleep_window"] = (fallback_df["hour"] >= 18) | (fallback_df["hour"] < 14)
        inside = fallback_df[fallback_df["in_sleep_window"]]
        outside = fallback_df[~fallback_df["in_sleep_window"]]
        
        ax.scatter(outside["date"], outside["hour"], s=3, alpha=0.3, color="tab:blue", zorder=3, label="Outside Sleep Window")
        ax.scatter(inside["date"], inside["hour"], s=12, alpha=0.7, color="tab:orange", zorder=3, label="Inside Sleep Window")
        
    finalize_and_save_plot(fig, ax, output_file_main)
