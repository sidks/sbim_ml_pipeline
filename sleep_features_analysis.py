#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import pytz


###############################################################################
# CONFIG
###############################################################################

ACCEL_CSV = "/srv/repos/raddlab_datascience/cingo-db-data-extractor/output/iwellas_acceleration_2026-03-27_to_2026-06-15.csv"
SURVEY_CSV = "/srv/repos/raddlab_datascience/cingo-db-data-extractor/output/iwellas_survey_data_2026-03-27_to_2026-06-15.csv"

OUTPUT_CSV = "/srv/repos/sbim_ml_pipeline/sleep_acceleration_analysis_iwellas_2420.csv"


###############################################################################
# SURVEY PARSING
###############################################################################

def extract_time_from_response(response_str, key):
    """
    Example response:

    '[{"bed_time_military":"21:30", ...}]'
    """

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
    """
    Build one row per survey day:

    survey_date
    true_bed
    true_wake
    timezone
    """

    survey_df["completed_at"] = pd.to_datetime(
        survey_df["completed_at"],
        utc=True,
        format="mixed",
    )

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

        bed_time_str = extract_time_from_response(
            bedtime_row["responses"],
            "bed_time_military",
        )

        wake_time_str = extract_time_from_response(
            wake_row["responses"],
            "wake_up_time_military",
        )

        if bed_time_str is None or wake_time_str is None:
            continue

        completed_local = (
            bedtime_row["completed_at"]
            .tz_convert(tz)
        )

        survey_date = completed_local.date()

        bed_hour, bed_min = map(int, bed_time_str.split(":"))
        wake_hour, wake_min = map(int, wake_time_str.split(":"))

        true_bed = tz.localize(
            pd.Timestamp(
                year=survey_date.year,
                month=survey_date.month,
                day=survey_date.day,
                hour=bed_hour,
                minute=bed_min,
            ).to_pydatetime()
        ) - timedelta(days=1)

        true_wake = tz.localize(
            pd.Timestamp(
                year=survey_date.year,
                month=survey_date.month,
                day=survey_date.day,
                hour=wake_hour,
                minute=wake_min,
            ).to_pydatetime()
        )

        rows.append(
            {
                "survey_id": survey_id,
                "survey_date": survey_date,
                "timezone": timezone_name,
                "true_bed": true_bed,
                "true_wake": true_wake,
            }
        )

    return pd.DataFrame(rows)


###############################################################################
# ACCELERATION ANALYSIS
###############################################################################

def compute_sleep_prediction(timestamps):
    """
    Reproduce compute_sleep_features_from_acceleration()
    """

    if len(timestamps) < 2:
        return {
            "method_used": "insufficient_data",
            "pred_bed": None,
            "pred_wake": None,
            "largest_gap_start": None,
            "largest_gap_end": None,
            "largest_gap_hours": None,
        }

    gaps = []

    for i in range(len(timestamps) - 1):

        gap_hours = (
            timestamps[i + 1] - timestamps[i]
        ).total_seconds() / 3600

        gaps.append(
            {
                "left": timestamps[i],
                "right": timestamps[i + 1],
                "gap_hours": gap_hours,
            }
        )

    largest_gap = max(gaps, key=lambda x: x["gap_hours"])

    # Original logic
    if largest_gap["gap_hours"] > 3:

        pred_bed = largest_gap["left"]
        pred_wake = largest_gap["right"] - timedelta(hours=1)

        return {
            "method_used": "gap",
            "pred_bed": pred_bed,
            "pred_wake": pred_wake,
            "largest_gap_start": largest_gap["left"],
            "largest_gap_end": largest_gap["right"],
            "largest_gap_hours": largest_gap["gap_hours"],
        }

    # Fallback logic
    first_point = timestamps[0]
    last_point = timestamps[-1]

    pred_bed = None
    pred_wake = None

    if first_point.hour < 2:
        pred_bed = first_point

    if last_point.hour >= 6:
        pred_wake = last_point - timedelta(hours=1)

    return {
        "method_used": "fallback",
        "pred_bed": pred_bed,
        "pred_wake": pred_wake,
        "largest_gap_start": largest_gap["left"],
        "largest_gap_end": largest_gap["right"],
        "largest_gap_hours": largest_gap["gap_hours"],
    }


def nearest_sample_metrics(timestamps, target_time):

    if len(timestamps) == 0:
        return None, None

    before = [t for t in timestamps if t <= target_time]
    after = [t for t in timestamps if t >= target_time]

    prev_min = None
    next_min = None

    if len(before):
        prev_min = (
            target_time - before[-1]
        ).total_seconds() / 60

    if len(after):
        next_min = (
            after[0] - target_time
        ).total_seconds() / 60

    return prev_min, next_min


def compute_overlap_ratio(
    true_bed,
    true_wake,
    gap_start,
    gap_end,
):
    """
    Overlap between:
        GT sleep interval      [true_bed, true_wake]
        Largest gap interval   [gap_start, gap_end]

    Returns:
        overlap_hours
        overlap_ratio
    """

    if (
        true_bed is None
        or true_wake is None
        or gap_start is None
        or gap_end is None
    ):
        return None, None

    overlap_start = max(true_bed, gap_start)
    overlap_end = min(true_wake, gap_end)

    overlap_seconds = max(
        0,
        (overlap_end - overlap_start).total_seconds(),
    )

    overlap_hours = overlap_seconds / 3600

    true_sleep_hours = (
        true_wake - true_bed
    ).total_seconds() / 3600

    if true_sleep_hours <= 0:
        return overlap_hours, None

    overlap_ratio = overlap_hours / true_sleep_hours

    return overlap_hours, overlap_ratio


###############################################################################
# MAIN
###############################################################################

def main():

    accel_df = pd.read_csv(ACCEL_CSV)
    survey_df = pd.read_csv(SURVEY_CSV)

    accel_df["captured_at"] = pd.to_datetime(
        accel_df["captured_at"],
        utc=True,
        format="mixed",
    )

    gt_df = build_ground_truth_sleep_table(survey_df)

    print(f"Found {len(gt_df)} sleep surveys")

    results = []

    for _, row in gt_df.iterrows():

        tz = pytz.timezone(row["timezone"])

        survey_date = row["survey_date"]

        sleep_start_local = tz.localize(
            pd.Timestamp(
                survey_date.year,
                survey_date.month,
                survey_date.day,
                18,
                0,
                0,
            ).to_pydatetime()
        ) - timedelta(days=1)

        sleep_end_local = tz.localize(
            pd.Timestamp(
                survey_date.year,
                survey_date.month,
                survey_date.day,
                14,
                0,
                0,
            ).to_pydatetime()
        )

        sleep_start_utc = sleep_start_local.astimezone(pytz.UTC)
        sleep_end_utc = sleep_end_local.astimezone(pytz.UTC)

        window = accel_df[
            (accel_df["captured_at"] >= sleep_start_utc)
            &
            (accel_df["captured_at"] <= sleep_end_utc)
        ]

        timestamps = sorted(window["captured_at"].tolist())

        prediction = compute_sleep_prediction(timestamps)

        overlap_hours, overlap_ratio = compute_overlap_ratio(
            row["true_bed"].astimezone(pytz.UTC),
            row["true_wake"].astimezone(pytz.UTC),
            prediction["largest_gap_start"],
            prediction["largest_gap_end"],
        )

        pred_bed = prediction["pred_bed"]
        pred_wake = prediction["pred_wake"]

        bed_error = None
        wake_error = None

        if pred_bed is not None:
            bed_error = (
                pred_bed - row["true_bed"]
            ).total_seconds() / 60

        if pred_wake is not None:
            wake_error = (
                pred_wake - row["true_wake"]
            ).total_seconds() / 60

        bed_prev, bed_next = nearest_sample_metrics(
            timestamps,
            row["true_bed"].astimezone(pytz.UTC),
        )

        wake_prev, wake_next = nearest_sample_metrics(
            timestamps,
            row["true_wake"].astimezone(pytz.UTC),
        )

        actual_sleep_duration_hours = (
            row["true_wake"] - row["true_bed"]
        ).total_seconds() / 3600

        predicted_sleep_duration_hours = None

        if pred_bed is not None and pred_wake is not None:
            predicted_sleep_duration_hours = (
                pred_wake - pred_bed
            ).total_seconds() / 3600

        sleep_duration_error_hours = None

        if predicted_sleep_duration_hours is not None:
            sleep_duration_error_hours = (
                predicted_sleep_duration_hours
                - actual_sleep_duration_hours
            )

        sleep_duration_error_min = None

        if sleep_duration_error_hours is not None:
            sleep_duration_error_min = (
                sleep_duration_error_hours * 60
            )

        results.append(
            {
                "survey_date": survey_date,

                "true_bed": row["true_bed"],
                "true_wake": row["true_wake"],

                "num_accel_samples": len(timestamps),

                "method_used": prediction["method_used"],

                "largest_gap_start":
                    prediction["largest_gap_start"],

                "largest_gap_end":
                    prediction["largest_gap_end"],

                "largest_gap_hours":
                    prediction["largest_gap_hours"],

                "pred_bed": pred_bed,
                "pred_wake": pred_wake,

                "bed_error_min": bed_error,
                "wake_error_min": wake_error,

                "bed_prev_sample_min": bed_prev,
                "bed_next_sample_min": bed_next,

                "wake_prev_sample_min": wake_prev,
                "wake_next_sample_min": wake_next,

                "overlap_hours": overlap_hours,
                "overlap_ratio": overlap_ratio,

                "actual_sleep_duration_hours":
                    actual_sleep_duration_hours,
                
                "predicted_sleep_duration_hours":
                    predicted_sleep_duration_hours,
                
                "sleep_duration_error_hours":
                    sleep_duration_error_hours,
                
                "sleep_duration_error_min":
                    sleep_duration_error_min,
            }
        )

    results_df = pd.DataFrame(results)

    results_df = (
        results_df
        .sort_values("survey_date")
        .reset_index(drop=True)
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_CSV)

    print("\nSummary")
    print("=" * 60)

    if len(results_df):

        print(
            "Mean abs bed error (min):",
            results_df["bed_error_min"].abs().mean(),
        )

        print(
            "Mean abs wake error (min):",
            results_df["wake_error_min"].abs().mean(),
        )

        print("\nMethod counts:")
        print(
            results_df["method_used"]
            .value_counts(dropna=False)
        )

        print("\nLargest gap statistics:")
        print(
            results_df["largest_gap_hours"]
            .describe()
        )

        print(
            "\nMean overlap ratio:",
            results_df["overlap_ratio"].mean(),
        )
        
        print(
            "Median overlap ratio:",
            results_df["overlap_ratio"].median(),
        )
        
        print(
            "Mean overlap hours:",
            results_df["overlap_hours"].mean(),
        )

        print(
            "\nMean abs sleep duration error (hours):",
            results_df[
                "sleep_duration_error_hours"
            ].abs().mean(),
        )
        
        print(
            "Median abs sleep duration error (hours):",
            results_df[
                "sleep_duration_error_hours"
            ].abs().median(),
        )


if __name__ == "__main__":
    main()
