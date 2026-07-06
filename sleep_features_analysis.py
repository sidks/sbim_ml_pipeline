#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import pytz

###############################################################################
# CONFIG: PARTICIPANT "iwellnt" FILES
###############################################################################
BASE_DIR = Path("/srv/repos/raddlab_datascience/cingo-db-data-extractor/output")

ACCEL_CSV = BASE_DIR / "iwellnt_acceleration_2026-03-27_to_2026-06-27.csv"
LIGHT_CSV = BASE_DIR / "iwellnt_light_2026-03-27_to_2026-06-27.csv"
DEVICE_CSV = BASE_DIR / "iwellnt_device_usage_2026-03-27_to_2026-06-27.csv"
SURVEY_CSV = BASE_DIR / "iwellnt_survey_data_2026-03-27_to_2026-06-27.csv"

OUTPUT_CSV = "/srv/repos/sbim_ml_pipeline/sleep_analysis_iwellnt_new_logic.csv"

###############################################################################
# SURVEY GROUND TRUTH PARSING WITH DAYTIME ERROR CORRECTION
###############################################################################
def extract_time_from_response(response_str, key):
    if pd.isna(response_str):
        return None
    try:
        data = json.loads(response_str)
        if isinstance(data, list) and len(data) > 0:
            return data[0].get(key)
    except Exception:
        pass
    return None

def build_ground_truth_sleep_table(survey_df):
    survey_df["completed_at"] = pd.to_datetime(survey_df["completed_at"], utc=True, format="mixed")
    rows = []

    for survey_id, grp in survey_df.groupby("survey_id"):
        bedtime_row = grp[grp["check_in_type"] == "bedtime"]
        wake_row = grp[grp["check_in_type"] == "wake-up"]

        if len(bedtime_row) == 0 or len(wake_row) == 0:
            continue

        bedtime_row = bedtime_row.iloc[0]
        wake_row = wake_row.iloc[0]
        timezone_name = bedtime_row["user_timezone"]

        try:
            tz = pytz.timezone(timezone_name)
        except Exception:
            continue

        bed_time_str = extract_time_from_response(bedtime_row["responses"], "bed_time_military")
        wake_time_str = extract_time_from_response(wake_row["responses"], "wake_up_time_military")

        if bed_time_str is None or wake_time_str is None:
            continue

        completed_local = bedtime_row["completed_at"].tz_convert(tz)
        survey_date = completed_local.date()

        bed_hour, bed_min = map(int, bed_time_str.split(":"))
        wake_hour, wake_min = map(int, wake_time_str.split(":"))

        # --- AUTO-CORRECT DAYTIME BEDTIME ERRORS ---
        # Initialize the day offset for bedtime (usually -1 day relative to survey day)
        bed_day_offset = -1

        if 9 <= bed_hour <= 17:
            original_str = f"{bed_hour:02d}:{bed_min:02d}"
            
            if bed_hour == 12:
                bed_hour = 0       # 12:00 PM -> 00:00 midnight
                bed_day_offset = 0 # June 15th survey midnight means 00:00 on June 15th
            elif bed_hour < 12:
                bed_hour += 12     # 10:45 AM -> 22:45 PM (still previous night, offset remains -1)
            else:
                bed_hour -= 12     # 13:00 -> 01:00 AM 
                bed_day_offset = 0 # 1:00 AM belongs to the survey morning itself
                
            print(f"[{survey_date}] Corrected bedtime: {original_str} -> {bed_hour:02d}:{bed_min:02d} (Offset day shift: {bed_day_offset})")

        # Apply the dynamic bed day offset safely
        true_bed = tz.localize(pd.Timestamp(survey_date.year, survey_date.month, survey_date.day, bed_hour, bed_min)) + timedelta(days=bed_day_offset)
        true_wake = tz.localize(pd.Timestamp(survey_date.year, survey_date.month, survey_date.day, wake_hour, wake_min))

        rows.append({
            "survey_id": survey_id,
            "survey_date": survey_date,
            "timezone": timezone_name,
            "true_bed": true_bed,
            "true_wake": true_wake,
        })
    return pd.DataFrame(rows)

###############################################################################
# TELEMETRY LOADING HELPER
###############################################################################
def load_and_localize_stream(file_path, tz):
    if not Path(file_path).exists():
        return pd.DataFrame(columns=["captured_at", "local_time", "hour"])
        
    df = pd.read_csv(file_path)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, format="mixed")
    df["local_time"] = df["captured_at"].dt.tz_convert(tz)
    df["hour"] = df["local_time"].dt.hour + df["local_time"].dt.minute / 60.0 + df["local_time"].dt.second / 3600.0
    return df

###############################################################################
# WINDOWED MULTI-SENSOR HEURISTIC LOGIC
###############################################################################
def compute_sleep_prediction_new(accel_df, light_df, unlock_df, window_start, window_end):
    a_win = accel_df[(accel_df["captured_at"] >= window_start) & (accel_df["captured_at"] <= window_end)]
    l_win = light_df[(light_df["captured_at"] >= window_start) & (light_df["captured_at"] <= window_end)]
    u_win = unlock_df[(unlock_df["captured_at"] >= window_start) & (unlock_df["captured_at"] <= window_end)]

    bed_cond = lambda df: (df["hour"] >= 20.0) | (df["hour"] <= 1.0)
    wake_cond = lambda df: (df["hour"] >= 4.0) & (df["hour"] <= 10.0)

    bed_candidates = []
    wake_candidates = []

    # 1. Acceleration
    a_bed = a_win[bed_cond(a_win)]["captured_at"].max()
    a_wake = a_win[wake_cond(a_win)]["captured_at"].min()
    if pd.notna(a_bed): bed_candidates.append(a_bed)
    if pd.notna(a_wake): wake_candidates.append(a_wake)

    # 2. Device Unlock
    u_bed = u_win[bed_cond(u_win)]["captured_at"].max()
    u_wake = u_win[wake_cond(u_win)]["captured_at"].min()
    if pd.notna(u_bed): bed_candidates.append(u_bed)
    if pd.notna(u_wake): wake_candidates.append(u_wake)

    # 3. Light Sensor
    l_bed = l_win[bed_cond(l_win)]["captured_at"].max()
    if pd.notna(l_bed): bed_candidates.append(l_bed)
    
    l_wake_pts = l_win[wake_cond(l_win)].sort_values("captured_at")["captured_at"].tolist()
    if len(l_wake_pts) >= 2:
        ts_avg = l_wake_pts[0] + (l_wake_pts[1] - l_wake_pts[0]) / 2
        wake_candidates.append(ts_avg)
    elif len(l_wake_pts) == 1:
        wake_candidates.append(l_wake_pts[0])

    # Ensemble rules
    pred_bed = max(bed_candidates) if bed_candidates else None
    pred_wake = min(wake_candidates) if wake_candidates else None

    return pred_bed, pred_wake

###############################################################################
# MAIN PIPELINE
###############################################################################
def main():
    if not SURVEY_CSV.exists():
        print(f"Survey file not found at: {SURVEY_CSV}")
        return

    survey_df = pd.read_csv(SURVEY_CSV)
    gt_df = build_ground_truth_sleep_table(survey_df)
    print(f"Loaded {len(gt_df)} processed ground truth records.\n")

    if gt_df.empty:
        return

    target_tz = pytz.timezone(gt_df.iloc[0]["timezone"])

    print("Loading sensor streams...")
    raw_accel = load_and_localize_stream(ACCEL_CSV, target_tz)
    raw_light = load_and_localize_stream(LIGHT_CSV, target_tz)
    
    # Process Device usage file for unlocks
    if DEVICE_CSV.exists():
        raw_device = pd.read_csv(DEVICE_CSV)
        raw_device["captured_at"] = pd.to_datetime(raw_device["captured_at"], utc=True, format="mixed")
        event_col = "event_type" if "event_type" in raw_device.columns else raw_device.columns[2]
        json_col = "payload" if "payload" in raw_device.columns else raw_device.columns[3]
        
        def isolate_unlocks(row):
            if row[event_col] == "SK Device Usage":
                try:
                    return int(json.loads(row[json_col]).get("total_unlocks", 0)) > 0
                except: return False
            return False
            
        raw_device["is_unlock"] = raw_device.apply(isolate_unlocks, axis=1)
        filtered_unlocks = raw_device[raw_device["is_unlock"] == True].copy()
        filtered_unlocks["local_time"] = filtered_unlocks["captured_at"].dt.tz_convert(target_tz)
        filtered_unlocks["hour"] = filtered_unlocks["local_time"].dt.hour + filtered_unlocks["local_time"].dt.minute / 60.0 + filtered_unlocks["local_time"].dt.second / 3600.0
    else:
        filtered_unlocks = pd.DataFrame(columns=["captured_at", "hour"])

    results = []

    for _, row in gt_df.iterrows():
        tz = pytz.timezone(row["timezone"])
        survey_date = row["survey_date"]

        window_start = (tz.localize(pd.Timestamp(survey_date.year, survey_date.month, survey_date.day, 18, 0)).astimezone(pytz.UTC)) - timedelta(days=1)
        window_end = tz.localize(pd.Timestamp(survey_date.year, survey_date.month, survey_date.day, 14, 0)).astimezone(pytz.UTC)

        pred_bed, pred_wake = compute_sleep_prediction_new(
            raw_accel, raw_light, filtered_unlocks, window_start, window_end
        )

        true_bed_utc = row["true_bed"].astimezone(pytz.UTC)
        true_wake_utc = row["true_wake"].astimezone(pytz.UTC)
        true_duration = (true_wake_utc - true_bed_utc).total_seconds() / 3600

        bed_error_min, wake_error_min, pred_duration_hours, duration_error_hours = None, None, None, None

        if pred_bed is not None and pred_wake is not None:
            bed_error_min = (pred_bed - true_bed_utc).total_seconds() / 60
            wake_error_min = (pred_wake - true_wake_utc).total_seconds() / 60
            pred_duration_hours = (pred_wake - pred_bed).total_seconds() / 3600
            duration_error_hours = pred_duration_hours - true_duration

        results.append({
            "survey_date": survey_date,
            "true_bed": row["true_bed"],
            "true_wake": row["true_wake"],
            "pred_bed": pred_bed.astimezone(tz) if pred_bed else None,
            "pred_wake": pred_wake.astimezone(tz) if pred_wake else None,
            "bed_error_min": bed_error_min,
            "wake_error_min": wake_error_min,
            "actual_sleep_duration_hours": true_duration,
            "predicted_sleep_duration_hours": pred_duration_hours,
            "sleep_duration_error_hours": duration_error_hours
        })

    results_df = pd.DataFrame(results).sort_values("survey_date").reset_index(drop=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved metrics dataset to: {OUTPUT_CSV}")
    print("\n" + "="*50 + "\nERROR METRICS WITH SURVEY TIME-CORRECTIONS\n" + "="*50)
    
    clean_metrics = results_df.dropna(subset=["bed_error_min", "wake_error_min"])
    if not clean_metrics.empty:
        print(f"Mean Abs Bedtime Error : {clean_metrics['bed_error_min'].abs().mean():.2f} minutes")
        print(f"Mean Abs Waketime Error: {clean_metrics['wake_error_min'].abs().mean():.2f} minutes")
        print(f"Mean Abs Duration Error: {clean_metrics['sleep_duration_error_hours'].abs().mean():.2f} hours")
    else:
        print("No matches available to compute statistics.")
    print("="*50)

if __name__ == "__main__":
    main()
