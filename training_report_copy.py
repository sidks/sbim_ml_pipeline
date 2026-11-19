import warnings
warnings.filterwarnings("ignore")

import os
import copy
from pathlib import Path
import re
import glob
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datetime import datetime
import random
import math
import pickle

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance


# Directory containing the CSV files
data_dir = Path("/srv/repos/raddlab_datascience/cingo-pipeline/output/")

# Pattern to match files like:
# input-data-for-ai-models-2026-03-27.csv
pattern = re.compile(r"input-data-for-ai-models-(\d{4}-\d{2}-\d{2})\.csv")

# Find all matching CSV files
csv_files = []
dates = []

for file in data_dir.glob("input-data-for-ai-models-*.csv"):
    match = pattern.match(file.name)
    if match:
        file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        csv_files.append((file_date, file))
        dates.append(file_date)

# Ensure files exist
if not csv_files:
    raise ValueError("No matching CSV files found.")

# Get earliest and latest available dates
start_date = min(dates)
end_date = max(dates)

print(f"Earliest available date: {start_date}")
print(f"Most recent available date: {end_date}")

# Sort files by date
csv_files = sorted(csv_files, key=lambda x: x[0])

# Read and combine all CSVs
df_list = [pd.read_csv(file_path) for _, file_path in csv_files]

df = pd.concat(df_list, ignore_index=True)

print(f"\nRows before duplicate removal: {len(df)}")

# Remove duplicates based on cingo_username and study_day
df = df.drop_duplicates(
    subset=["cingo_username", "study_day", "time_submit_utc"],
    keep="first"
)

print(f"Rows after duplicate removal: {len(df)}")
print(f"Duplicates removed: {len(df_list) and (sum(len(d) for d in df_list) - len(df))}")

print(f"\nFinal dataframe shape: {df.shape}")

def fill_with_mode(s):
    if s.isna().all():
        return s  # leave as NaN if everything is missing
    mode_val = s.mode().iloc[0]  # take the first mode if multiple
    return s.fillna(mode_val)


# Step 1: Fill with mean for each (beiwe_id, studyday)
df[['magnitude_min', 'magnitude_max', 'magnitude_mean', 'magnitude_median', 'magnitude_std', 'magnitude_min_working_day', 'magnitude_max_working_day', 'magnitude_mean_working_day', 'magnitude_median_working_day', 
    'magnitude_std_working_day', 'magnitude_min_last_night', 'magnitude_max_last_night', 'magnitude_mean_last_night', 'magnitude_median_last_night', 'magnitude_std_last_night', 'app_usage_total_minutes_per_hour', 
    'app_usage_category_books_minutes_per_hour', 'app_usage_category_education_minutes_per_hour', 'app_usage_category_entertainment_minutes_per_hour', 'app_usage_category_games_minutes_per_hour', 
    'app_usage_category_health_and_fitness_minutes_per_hour', 'app_usage_category_lifestyle_minutes_per_hour', 'app_usage_category_music_minutes_per_hour', 'app_usage_category_navigation_minutes_per_hour', 
    'app_usage_category_photo_and_video_minutes_per_hour', 'app_usage_category_productivity_minutes_per_hour', 'app_usage_category_shopping_minutes_per_hour', 'app_usage_category_social_networking_minutes_per_hour', 
    'app_usage_category_sports_minutes_per_hour', 'app_usage_category_weather_minutes_per_hour', 'app_usage_total_minutes_daily', 'app_usage_category_books_minutes_daily', 'app_usage_category_education_minutes_daily', 
    'app_usage_category_entertainment_minutes_daily', 'app_usage_category_games_minutes_daily', 'app_usage_category_health_and_fitness_minutes_daily', 'app_usage_category_lifestyle_minutes_daily', 
    'app_usage_category_music_minutes_daily', 'app_usage_category_navigation_minutes_daily', 'app_usage_category_photo_and_video_minutes_daily', 'app_usage_category_productivity_minutes_daily', 
    'app_usage_category_shopping_minutes_daily', 'app_usage_category_social_networking_minutes_daily', 'app_usage_category_sports_minutes_daily', 'app_usage_category_weather_minutes_daily', 'distance_per_hour', 'span_per_hour', 
    'convex_area', 'buffer_space', 'min_lux_mean', 'max_lux_mean', 'mean_lux_mean', 'std_lux_mean', 'num_incoming_messages_per_hour', 'num_outgoing_messages_per_hour', 'num_unique_contacts_per_hour', 
    'num_total_incoming_and_outgoing_messages_daily', 'num_incoming_calls_daily', 'num_outgoing_calls_daily', 'phone_call_duration_minutes_daily', 'num_unique_contacts_daily', 'screen_wakes_per_hour', 'unlocks_per_hour', 
    'unlocked_minutes_per_hour', 'screen_wakes_working_day', 'unlocks_working_day', 'unlocked_minutes_working_day', 'screen_wakes_last_night','unlocks_last_night', 'unlocked_minutes_last_night', 
    'app_usage_category_utilities_minutes_per_hour', 'app_usage_category_utilities_minutes_daily', 'app_usage_category_business_minutes_per_hour', 'app_usage_category_business_minutes_daily', 'app_usage_category_news_minutes_per_hour', 
    'app_usage_category_news_minutes_daily', 'app_usage_category_graphics_and_design_minutes_per_hour', 'app_usage_category_graphics_and_design_minutes_daily', 'app_usage_category_travel_minutes_per_hour', 
    'app_usage_category_travel_minutes_daily', 'app_usage_category_finance_minutes_per_hour', 'app_usage_category_finance_minutes_daily', 
    'battery_level_min', 'battery_level_max', 'battery_level_mean', 'battery_level_median', 'time_spent_at_home_hrs_yesterday', 'time_spent_at_home_hrs_two_days_ago', 
    'time_spent_at_home_avg_hrs_past_week', 'keyboard_total_taps_per_hour', 'keyboard_total_words_per_hour', 'keyboard_average_daily_taps_per_hour', 
    'keyboard_average_daily_words_per_hour', 'estimated_sleep_duration_hours']] = (df.groupby(["cingo_username", "study_day"])[[
    'magnitude_min', 'magnitude_max', 'magnitude_mean', 'magnitude_median', 'magnitude_std', 'magnitude_min_working_day', 'magnitude_max_working_day', 'magnitude_mean_working_day', 'magnitude_median_working_day', 
    'magnitude_std_working_day', 'magnitude_min_last_night', 'magnitude_max_last_night', 'magnitude_mean_last_night', 'magnitude_median_last_night', 'magnitude_std_last_night', 'app_usage_total_minutes_per_hour', 
    'app_usage_category_books_minutes_per_hour', 'app_usage_category_education_minutes_per_hour', 'app_usage_category_entertainment_minutes_per_hour', 'app_usage_category_games_minutes_per_hour', 
    'app_usage_category_health_and_fitness_minutes_per_hour', 'app_usage_category_lifestyle_minutes_per_hour', 'app_usage_category_music_minutes_per_hour', 'app_usage_category_navigation_minutes_per_hour',
    'app_usage_category_photo_and_video_minutes_per_hour', 'app_usage_category_productivity_minutes_per_hour', 'app_usage_category_shopping_minutes_per_hour', 'app_usage_category_social_networking_minutes_per_hour', 
    'app_usage_category_sports_minutes_per_hour', 'app_usage_category_weather_minutes_per_hour', 'app_usage_total_minutes_daily', 'app_usage_category_books_minutes_daily', 'app_usage_category_education_minutes_daily', 
    'app_usage_category_entertainment_minutes_daily', 'app_usage_category_games_minutes_daily', 'app_usage_category_health_and_fitness_minutes_daily', 'app_usage_category_lifestyle_minutes_daily', 
    'app_usage_category_music_minutes_daily', 'app_usage_category_navigation_minutes_daily', 'app_usage_category_photo_and_video_minutes_daily', 'app_usage_category_productivity_minutes_daily', 
    'app_usage_category_shopping_minutes_daily', 'app_usage_category_social_networking_minutes_daily', 'app_usage_category_sports_minutes_daily', 'app_usage_category_weather_minutes_daily', 'distance_per_hour', 'span_per_hour', 
    'convex_area', 'buffer_space', 'min_lux_mean', 'max_lux_mean', 'mean_lux_mean', 'std_lux_mean', 'num_incoming_messages_per_hour', 'num_outgoing_messages_per_hour', 'num_unique_contacts_per_hour', 
    'num_total_incoming_and_outgoing_messages_daily', 'num_incoming_calls_daily', 'num_outgoing_calls_daily', 'phone_call_duration_minutes_daily', 'num_unique_contacts_daily', 'screen_wakes_per_hour', 'unlocks_per_hour', 
    'unlocked_minutes_per_hour', 'screen_wakes_working_day', 'unlocks_working_day', 'unlocked_minutes_working_day', 'screen_wakes_last_night','unlocks_last_night', 'unlocked_minutes_last_night', 
    'app_usage_category_utilities_minutes_per_hour', 'app_usage_category_utilities_minutes_daily', 'app_usage_category_business_minutes_per_hour', 'app_usage_category_business_minutes_daily', 'app_usage_category_news_minutes_per_hour', 
    'app_usage_category_news_minutes_daily', 'app_usage_category_graphics_and_design_minutes_per_hour', 'app_usage_category_graphics_and_design_minutes_daily', 'app_usage_category_travel_minutes_per_hour', 
    'app_usage_category_travel_minutes_daily', 'app_usage_category_finance_minutes_per_hour', 'app_usage_category_finance_minutes_daily', 
    'battery_level_min', 'battery_level_max', 'battery_level_mean', 'battery_level_median', 'time_spent_at_home_hrs_yesterday', 'time_spent_at_home_hrs_two_days_ago', 
    'time_spent_at_home_avg_hrs_past_week', 'keyboard_total_taps_per_hour', 'keyboard_total_words_per_hour', 'keyboard_average_daily_taps_per_hour', 
    'keyboard_average_daily_words_per_hour', 'estimated_sleep_duration_hours']].transform(lambda x: x.fillna(x.mean())))

# Step 2: For leftovers, fill with mean per beiwe_id
df[['magnitude_min', 'magnitude_max', 'magnitude_mean', 'magnitude_median', 'magnitude_std', 'magnitude_min_working_day', 'magnitude_max_working_day', 'magnitude_mean_working_day', 'magnitude_median_working_day', 
    'magnitude_std_working_day', 'magnitude_min_last_night', 'magnitude_max_last_night', 'magnitude_mean_last_night', 'magnitude_median_last_night', 'magnitude_std_last_night', 'app_usage_total_minutes_per_hour', 
    'app_usage_category_books_minutes_per_hour', 'app_usage_category_education_minutes_per_hour', 'app_usage_category_entertainment_minutes_per_hour', 'app_usage_category_games_minutes_per_hour', 
    'app_usage_category_health_and_fitness_minutes_per_hour', 'app_usage_category_lifestyle_minutes_per_hour', 'app_usage_category_music_minutes_per_hour', 'app_usage_category_navigation_minutes_per_hour', 
    'app_usage_category_photo_and_video_minutes_per_hour', 'app_usage_category_productivity_minutes_per_hour', 'app_usage_category_shopping_minutes_per_hour', 'app_usage_category_social_networking_minutes_per_hour', 
    'app_usage_category_sports_minutes_per_hour', 'app_usage_category_weather_minutes_per_hour', 'app_usage_total_minutes_daily', 'app_usage_category_books_minutes_daily', 'app_usage_category_education_minutes_daily', 
    'app_usage_category_entertainment_minutes_daily', 'app_usage_category_games_minutes_daily', 'app_usage_category_health_and_fitness_minutes_daily', 'app_usage_category_lifestyle_minutes_daily', 
    'app_usage_category_music_minutes_daily', 'app_usage_category_navigation_minutes_daily', 'app_usage_category_photo_and_video_minutes_daily', 'app_usage_category_productivity_minutes_daily', 
    'app_usage_category_shopping_minutes_daily', 'app_usage_category_social_networking_minutes_daily', 'app_usage_category_sports_minutes_daily', 'app_usage_category_weather_minutes_daily', 'distance_per_hour', 'span_per_hour', 
    'convex_area', 'buffer_space', 'min_lux_mean', 'max_lux_mean', 'mean_lux_mean', 'std_lux_mean', 'num_incoming_messages_per_hour', 'num_outgoing_messages_per_hour', 'num_unique_contacts_per_hour', 
    'num_total_incoming_and_outgoing_messages_daily', 'num_incoming_calls_daily', 'num_outgoing_calls_daily', 'phone_call_duration_minutes_daily', 'num_unique_contacts_daily', 'screen_wakes_per_hour', 'unlocks_per_hour', 
    'unlocked_minutes_per_hour', 'screen_wakes_working_day', 'unlocks_working_day', 'unlocked_minutes_working_day', 'screen_wakes_last_night','unlocks_last_night', 'unlocked_minutes_last_night', 
    'app_usage_category_utilities_minutes_per_hour', 'app_usage_category_utilities_minutes_daily', 'app_usage_category_business_minutes_per_hour', 'app_usage_category_business_minutes_daily', 'app_usage_category_news_minutes_per_hour', 
    'app_usage_category_news_minutes_daily', 'app_usage_category_graphics_and_design_minutes_per_hour', 'app_usage_category_graphics_and_design_minutes_daily', 'app_usage_category_travel_minutes_per_hour', 
    'app_usage_category_travel_minutes_daily', 'app_usage_category_finance_minutes_per_hour', 'app_usage_category_finance_minutes_daily', 
    'battery_level_min', 'battery_level_max', 'battery_level_mean', 'battery_level_median', 'time_spent_at_home_hrs_yesterday', 'time_spent_at_home_hrs_two_days_ago', 
    'time_spent_at_home_avg_hrs_past_week', 'keyboard_total_taps_per_hour', 'keyboard_total_words_per_hour', 'keyboard_average_daily_taps_per_hour', 
    'keyboard_average_daily_words_per_hour', 'estimated_sleep_duration_hours']] = (df.groupby(["cingo_username"])[[
    'magnitude_min', 'magnitude_max', 'magnitude_mean', 'magnitude_median', 'magnitude_std', 'magnitude_min_working_day', 'magnitude_max_working_day', 'magnitude_mean_working_day', 'magnitude_median_working_day', 
    'magnitude_std_working_day', 'magnitude_min_last_night', 'magnitude_max_last_night', 'magnitude_mean_last_night', 'magnitude_median_last_night', 'magnitude_std_last_night', 'app_usage_total_minutes_per_hour', 
    'app_usage_category_books_minutes_per_hour', 'app_usage_category_education_minutes_per_hour', 'app_usage_category_entertainment_minutes_per_hour', 'app_usage_category_games_minutes_per_hour', 
    'app_usage_category_health_and_fitness_minutes_per_hour', 'app_usage_category_lifestyle_minutes_per_hour', 'app_usage_category_music_minutes_per_hour', 'app_usage_category_navigation_minutes_per_hour',
    'app_usage_category_photo_and_video_minutes_per_hour', 'app_usage_category_productivity_minutes_per_hour', 'app_usage_category_shopping_minutes_per_hour', 'app_usage_category_social_networking_minutes_per_hour', 
    'app_usage_category_sports_minutes_per_hour', 'app_usage_category_weather_minutes_per_hour', 'app_usage_total_minutes_daily', 'app_usage_category_books_minutes_daily', 'app_usage_category_education_minutes_daily', 
    'app_usage_category_entertainment_minutes_daily', 'app_usage_category_games_minutes_daily', 'app_usage_category_health_and_fitness_minutes_daily', 'app_usage_category_lifestyle_minutes_daily', 
    'app_usage_category_music_minutes_daily', 'app_usage_category_navigation_minutes_daily', 'app_usage_category_photo_and_video_minutes_daily', 'app_usage_category_productivity_minutes_daily', 
    'app_usage_category_shopping_minutes_daily', 'app_usage_category_social_networking_minutes_daily', 'app_usage_category_sports_minutes_daily', 'app_usage_category_weather_minutes_daily', 'distance_per_hour', 'span_per_hour', 
    'convex_area', 'buffer_space', 'min_lux_mean', 'max_lux_mean', 'mean_lux_mean', 'std_lux_mean', 'num_incoming_messages_per_hour', 'num_outgoing_messages_per_hour', 'num_unique_contacts_per_hour', 
    'num_total_incoming_and_outgoing_messages_daily', 'num_incoming_calls_daily', 'num_outgoing_calls_daily', 'phone_call_duration_minutes_daily', 'num_unique_contacts_daily', 'screen_wakes_per_hour', 'unlocks_per_hour', 
    'unlocked_minutes_per_hour', 'screen_wakes_working_day', 'unlocks_working_day', 'unlocked_minutes_working_day', 'screen_wakes_last_night','unlocks_last_night', 'unlocked_minutes_last_night', 
    'app_usage_category_utilities_minutes_per_hour', 'app_usage_category_utilities_minutes_daily', 'app_usage_category_business_minutes_per_hour', 'app_usage_category_business_minutes_daily', 'app_usage_category_news_minutes_per_hour', 
    'app_usage_category_news_minutes_daily', 'app_usage_category_graphics_and_design_minutes_per_hour', 'app_usage_category_graphics_and_design_minutes_daily', 'app_usage_category_travel_minutes_per_hour', 
    'app_usage_category_travel_minutes_daily', 'app_usage_category_finance_minutes_per_hour', 'app_usage_category_finance_minutes_daily', 
    'battery_level_min', 'battery_level_max', 'battery_level_mean', 'battery_level_median', 'time_spent_at_home_hrs_yesterday', 'time_spent_at_home_hrs_two_days_ago', 
    'time_spent_at_home_avg_hrs_past_week', 'keyboard_total_taps_per_hour', 'keyboard_total_words_per_hour', 'keyboard_average_daily_taps_per_hour', 
    'keyboard_average_daily_words_per_hour', 'estimated_sleep_duration_hours']].transform(lambda x: x.fillna(x.mean())))

df[['magnitude_min', 'magnitude_max', 'magnitude_mean', 'magnitude_median', 'magnitude_std', 'magnitude_min_working_day', 'magnitude_max_working_day', 'magnitude_mean_working_day', 'magnitude_median_working_day', 
    'magnitude_std_working_day', 'magnitude_min_last_night', 'magnitude_max_last_night', 'magnitude_mean_last_night', 'magnitude_median_last_night', 'magnitude_std_last_night', 'app_usage_total_minutes_per_hour', 
    'app_usage_category_books_minutes_per_hour', 'app_usage_category_education_minutes_per_hour', 'app_usage_category_entertainment_minutes_per_hour', 'app_usage_category_games_minutes_per_hour', 
    'app_usage_category_health_and_fitness_minutes_per_hour', 'app_usage_category_lifestyle_minutes_per_hour', 'app_usage_category_music_minutes_per_hour', 'app_usage_category_navigation_minutes_per_hour', 
    'app_usage_category_photo_and_video_minutes_per_hour', 'app_usage_category_productivity_minutes_per_hour', 'app_usage_category_shopping_minutes_per_hour', 'app_usage_category_social_networking_minutes_per_hour', 
    'app_usage_category_sports_minutes_per_hour', 'app_usage_category_weather_minutes_per_hour', 'app_usage_total_minutes_daily', 'app_usage_category_books_minutes_daily', 'app_usage_category_education_minutes_daily', 
    'app_usage_category_entertainment_minutes_daily', 'app_usage_category_games_minutes_daily', 'app_usage_category_health_and_fitness_minutes_daily', 'app_usage_category_lifestyle_minutes_daily', 
    'app_usage_category_music_minutes_daily', 'app_usage_category_navigation_minutes_daily', 'app_usage_category_photo_and_video_minutes_daily', 'app_usage_category_productivity_minutes_daily', 
    'app_usage_category_shopping_minutes_daily', 'app_usage_category_social_networking_minutes_daily', 'app_usage_category_sports_minutes_daily', 'app_usage_category_weather_minutes_daily', 'distance_per_hour', 'span_per_hour', 
    'convex_area', 'buffer_space', 'min_lux_mean', 'max_lux_mean', 'mean_lux_mean', 'std_lux_mean', 'num_incoming_messages_per_hour', 'num_outgoing_messages_per_hour', 'num_unique_contacts_per_hour', 
    'num_total_incoming_and_outgoing_messages_daily', 'num_incoming_calls_daily', 'num_outgoing_calls_daily', 'phone_call_duration_minutes_daily', 'num_unique_contacts_daily', 'screen_wakes_per_hour', 'unlocks_per_hour', 
    'unlocked_minutes_per_hour', 'screen_wakes_working_day', 'unlocks_working_day', 'unlocked_minutes_working_day', 'screen_wakes_last_night','unlocks_last_night', 'unlocked_minutes_last_night', 
    'app_usage_category_utilities_minutes_per_hour', 'app_usage_category_utilities_minutes_daily', 'app_usage_category_business_minutes_per_hour', 'app_usage_category_business_minutes_daily', 'app_usage_category_news_minutes_per_hour', 
    'app_usage_category_news_minutes_daily', 'app_usage_category_graphics_and_design_minutes_per_hour', 'app_usage_category_graphics_and_design_minutes_daily', 'app_usage_category_travel_minutes_per_hour', 
    'app_usage_category_travel_minutes_daily', 'app_usage_category_finance_minutes_per_hour', 'app_usage_category_finance_minutes_daily', 
    'battery_level_min', 'battery_level_max', 'battery_level_mean', 'battery_level_median', 'time_spent_at_home_hrs_yesterday', 'time_spent_at_home_hrs_two_days_ago', 
    'time_spent_at_home_avg_hrs_past_week', 'keyboard_total_taps_per_hour', 'keyboard_total_words_per_hour', 'keyboard_average_daily_taps_per_hour', 
    'keyboard_average_daily_words_per_hour', 'estimated_sleep_duration_hours']] = (df[[
    'magnitude_min', 'magnitude_max', 'magnitude_mean', 'magnitude_median', 'magnitude_std', 'magnitude_min_working_day', 'magnitude_max_working_day', 'magnitude_mean_working_day', 'magnitude_median_working_day', 
    'magnitude_std_working_day', 'magnitude_min_last_night', 'magnitude_max_last_night', 'magnitude_mean_last_night', 'magnitude_median_last_night', 'magnitude_std_last_night', 'app_usage_total_minutes_per_hour', 
    'app_usage_category_books_minutes_per_hour', 'app_usage_category_education_minutes_per_hour', 'app_usage_category_entertainment_minutes_per_hour', 'app_usage_category_games_minutes_per_hour', 
    'app_usage_category_health_and_fitness_minutes_per_hour', 'app_usage_category_lifestyle_minutes_per_hour', 'app_usage_category_music_minutes_per_hour', 'app_usage_category_navigation_minutes_per_hour',
    'app_usage_category_photo_and_video_minutes_per_hour', 'app_usage_category_productivity_minutes_per_hour', 'app_usage_category_shopping_minutes_per_hour', 'app_usage_category_social_networking_minutes_per_hour', 
    'app_usage_category_sports_minutes_per_hour', 'app_usage_category_weather_minutes_per_hour', 'app_usage_total_minutes_daily', 'app_usage_category_books_minutes_daily', 'app_usage_category_education_minutes_daily', 
    'app_usage_category_entertainment_minutes_daily', 'app_usage_category_games_minutes_daily', 'app_usage_category_health_and_fitness_minutes_daily', 'app_usage_category_lifestyle_minutes_daily', 
    'app_usage_category_music_minutes_daily', 'app_usage_category_navigation_minutes_daily', 'app_usage_category_photo_and_video_minutes_daily', 'app_usage_category_productivity_minutes_daily', 
    'app_usage_category_shopping_minutes_daily', 'app_usage_category_social_networking_minutes_daily', 'app_usage_category_sports_minutes_daily', 'app_usage_category_weather_minutes_daily', 'distance_per_hour', 'span_per_hour', 
    'convex_area', 'buffer_space', 'min_lux_mean', 'max_lux_mean', 'mean_lux_mean', 'std_lux_mean', 'num_incoming_messages_per_hour', 'num_outgoing_messages_per_hour', 'num_unique_contacts_per_hour', 
    'num_total_incoming_and_outgoing_messages_daily', 'num_incoming_calls_daily', 'num_outgoing_calls_daily', 'phone_call_duration_minutes_daily', 'num_unique_contacts_daily', 'screen_wakes_per_hour', 'unlocks_per_hour', 
    'unlocked_minutes_per_hour', 'screen_wakes_working_day', 'unlocks_working_day', 'unlocked_minutes_working_day', 'screen_wakes_last_night','unlocks_last_night', 'unlocked_minutes_last_night', 
    'app_usage_category_utilities_minutes_per_hour', 'app_usage_category_utilities_minutes_daily', 'app_usage_category_business_minutes_per_hour', 'app_usage_category_business_minutes_daily', 'app_usage_category_news_minutes_per_hour', 
    'app_usage_category_news_minutes_daily', 'app_usage_category_graphics_and_design_minutes_per_hour', 'app_usage_category_graphics_and_design_minutes_daily', 'app_usage_category_travel_minutes_per_hour', 
    'app_usage_category_travel_minutes_daily', 'app_usage_category_finance_minutes_per_hour', 'app_usage_category_finance_minutes_daily', 
    'battery_level_min', 'battery_level_max', 'battery_level_mean', 'battery_level_median', 'time_spent_at_home_hrs_yesterday', 'time_spent_at_home_hrs_two_days_ago', 
    'time_spent_at_home_avg_hrs_past_week', 'keyboard_total_taps_per_hour', 'keyboard_total_words_per_hour', 'keyboard_average_daily_taps_per_hour', 
    'keyboard_average_daily_words_per_hour', 'estimated_sleep_duration_hours']].transform(lambda x: x.fillna(x.mean())))

feeling_cols = ['happy_intensity', 'sad_intensity', 'social_intensity', 'calm_intensity', 'motivated_intensity', 'bored_intensity', 
                'anxious_intensity', 'irritated_intensity']

features = [
    'magnitude_min', 'magnitude_max', 'magnitude_mean', 'magnitude_median', 'magnitude_std', 'magnitude_min_working_day', 'magnitude_max_working_day', 'magnitude_mean_working_day', 'magnitude_median_working_day', 
    'magnitude_std_working_day', 'magnitude_min_last_night', 'magnitude_max_last_night', 'magnitude_mean_last_night', 'magnitude_median_last_night', 'magnitude_std_last_night', 'app_usage_total_minutes_per_hour', 
    'app_usage_category_books_minutes_per_hour', 'app_usage_category_education_minutes_per_hour', 'app_usage_category_entertainment_minutes_per_hour', 'app_usage_category_games_minutes_per_hour', 
    'app_usage_category_health_and_fitness_minutes_per_hour', 'app_usage_category_lifestyle_minutes_per_hour', 'app_usage_category_music_minutes_per_hour', 'app_usage_category_navigation_minutes_per_hour', 
    'app_usage_category_photo_and_video_minutes_per_hour', 'app_usage_category_productivity_minutes_per_hour', 'app_usage_category_shopping_minutes_per_hour', 'app_usage_category_social_networking_minutes_per_hour', 
    'app_usage_category_sports_minutes_per_hour', 'app_usage_category_weather_minutes_per_hour', 'app_usage_total_minutes_daily', 'app_usage_category_books_minutes_daily', 'app_usage_category_education_minutes_daily', 
    'app_usage_category_entertainment_minutes_daily', 'app_usage_category_games_minutes_daily', 'app_usage_category_health_and_fitness_minutes_daily', 'app_usage_category_lifestyle_minutes_daily', 
    'app_usage_category_music_minutes_daily', 'app_usage_category_navigation_minutes_daily', 'app_usage_category_photo_and_video_minutes_daily', 'app_usage_category_productivity_minutes_daily', 
    'app_usage_category_shopping_minutes_daily', 'app_usage_category_social_networking_minutes_daily', 'app_usage_category_sports_minutes_daily', 'app_usage_category_weather_minutes_daily', 'distance_per_hour', 'span_per_hour', 
    'convex_area', 'buffer_space', 'min_lux_mean', 'max_lux_mean', 'mean_lux_mean', 'std_lux_mean', 'num_incoming_messages_per_hour', 'num_outgoing_messages_per_hour', 'num_unique_contacts_per_hour', 
    'num_total_incoming_and_outgoing_messages_daily', 'num_incoming_calls_daily', 'num_outgoing_calls_daily', 'phone_call_duration_minutes_daily', 'num_unique_contacts_daily', 'screen_wakes_per_hour', 'unlocks_per_hour', 
    'unlocked_minutes_per_hour', 'screen_wakes_working_day', 'unlocks_working_day', 'unlocked_minutes_working_day', 'screen_wakes_last_night','unlocks_last_night', 'unlocked_minutes_last_night', 
    'app_usage_category_utilities_minutes_per_hour', 'app_usage_category_utilities_minutes_daily', 'app_usage_category_business_minutes_per_hour','app_usage_category_business_minutes_daily', 'app_usage_category_news_minutes_per_hour', 
    'app_usage_category_news_minutes_daily', 'app_usage_category_graphics_and_design_minutes_per_hour', 'app_usage_category_graphics_and_design_minutes_daily', 'app_usage_category_travel_minutes_per_hour', 
    'app_usage_category_travel_minutes_daily', 'app_usage_category_finance_minutes_per_hour', 'app_usage_category_finance_minutes_daily', 
    'battery_level_min', 'battery_level_max', 'battery_level_mean', 'battery_level_median', 'time_spent_at_home_hrs_yesterday', 'time_spent_at_home_hrs_two_days_ago', 
    'time_spent_at_home_avg_hrs_past_week', 'keyboard_total_taps_per_hour', 'keyboard_total_words_per_hour', 'keyboard_average_daily_taps_per_hour', 
    'keyboard_average_daily_words_per_hour', 'estimated_sleep_duration_hours'
]

# Find participants that have at least one non-null value in the feature columns
active_participants = df.dropna(subset=features, how="all")["cingo_username"].unique()
# Keep only those participants in the DataFrame
df = df[df["cingo_username"].isin(active_participants)]

target_lst = feeling_cols
df = df.dropna(subset = features).reset_index(drop = True)
df = df[~(df['cingo_username'] == 'appstoretester1')]
X = df[features]
y = df[target_lst] if len(target_lst) > 1 else df[target_lst[0]]
participants = df['cingo_username'].unique()

print('Unique participants:', participants)

y_df = y if isinstance(y, pd.DataFrame) else pd.DataFrame({(targets if isinstance(targets, str) else targets[0]): y})
targets_list = list(y_df.columns)

# p1 = df[df['cingo_username'] == 'SBP001']
# print('Shape of p1:', p1.shape)

# p2 = df[df['cingo_username'] == 'SBP002']
# print('Shape of p2:', p2.shape)

# # Returns a boolean Series (True for rows where ALL features are NaN)
# all_empty = p2[features].isna().all(axis=1)

# # Count how many such rows exist
# print("Number of completely empty rows:", all_empty.sum())

# # Display those rows
# print(p2[all_empty])

# print(p2)

############################################################################################## HOME-TIME FEATURE ABLATION ##############################################################################################
#
# Design: forward single-feature isolation vs. a no-home baseline.
#   - baseline_no_home : all features EXCEPT the 3 home-time features
#   - +<home feature>  : baseline + exactly ONE home-time feature (x3)
#   - +all_home        : baseline + all 3 home-time features
#
# Every condition uses identical rows (df already drops NaNs across ALL features), identical
# LOGO-by-study_day splits, and is repeated over SEEDS. Per-row absolute errors are averaged
# across seeds, then compared between baseline and each condition with a paired Wilcoxon test.
# ==============================================================================================

from scipy.stats import wilcoxon, binomtest

home_features = [
    "time_spent_at_home_hrs_yesterday",
    "time_spent_at_home_hrs_two_days_ago",
    "time_spent_at_home_avg_hrs_past_week",
]
base_features_no_home = [f for f in features if f not in home_features]

BASELINE = "baseline_no_home"
ALL_HOME = "+all_home"
conditions = {BASELINE: base_features_no_home}
for hf in home_features:
    conditions[f"+{hf}"] = base_features_no_home + [hf]
conditions[ALL_HOME] = base_features_no_home + home_features
single_feature_conditions = [f"+{hf}" for hf in home_features]

SEEDS = [22]
MIN_STUDY_DAYS = 14

participants = sorted(df["cingo_username"].unique())


def make_model(seed):
    return XGBRegressor(
        n_estimators=50,
        learning_rate=0.01,
        max_depth=None,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


def safe_corr(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def run_logo(df_t, target, feature_subset, seeds, pbar=None):
    """LOGO over study_day, repeated for each seed. Advances `pbar` by 1 per model fit.

    Returns a dict with per-row arrays (aligned to df_t row order):
      y_true, y_pred (seed-averaged), abs_err (seed-averaged per-row |error|),
    and per-seed MAE / correlation lists.
    """
    X_all = df_t[feature_subset].to_numpy()
    y_all = df_t[target].to_numpy()
    splits = list(LeaveOneGroupOut().split(X_all, groups=df_t["study_day"]))

    preds = np.zeros((len(seeds), len(y_all)))
    for s_i, seed in enumerate(seeds):
        for train_idx, test_idx in splits:
            model = make_model(seed)
            model.fit(X_all[train_idx], y_all[train_idx])
            preds[s_i, test_idx] = model.predict(X_all[test_idx])
            if pbar is not None:
                pbar.update(1)

    abs_err_per_seed = np.abs(preds - y_all[None, :])
    return {
        "y_true": y_all,
        "y_pred": preds.mean(axis=0),
        "abs_err": abs_err_per_seed.mean(axis=0),
        "mae_per_seed": abs_err_per_seed.mean(axis=1),
        "corr_per_seed": np.array([safe_corr(y_all, preds[s]) for s in range(len(seeds))]),
    }


def paired_wilcoxon(base_err, cond_err):
    diff = base_err - cond_err
    if len(diff) < 2 or np.allclose(diff, 0):
        return np.nan
    try:
        return float(wilcoxon(base_err, cond_err, zero_method="wilcox").pvalue)
    except ValueError:
        return np.nan


# ============================================================================================
# RUN ALL CONDITIONS
# ============================================================================================
emotion_records = []
per_row_records = []

for pid in tqdm(participants, desc="Participants", position=0):
    df_p = df[df["cingo_username"] == pid].reset_index(drop=True)

    if df_p["study_day"].nunique() < MIN_STUDY_DAYS:
        tqdm.write(f"[skip] {pid}: only {df_p['study_day'].nunique()} study days")
        continue

    # Total model fits for this participant = sum over targets of (folds x conditions x seeds)
    valid_targets = []
    total_fits = 0
    for target in targets_list:
        n_folds = df_p.loc[df_p[target].notna(), "study_day"].nunique()
        if n_folds >= 2:
            valid_targets.append(target)
            total_fits += n_folds * len(conditions) * len(SEEDS)

    fit_bar = tqdm(total=total_fits, desc=f"  {pid} fits", position=1, leave=False, unit="fit")

    for t_i, target in enumerate(valid_targets, 1):
        df_t = df_p[df_p[target].notna()].reset_index(drop=True)

        cond_results = {}
        for c_i, (cond, feats) in enumerate(conditions.items(), 1):
            fit_bar.set_postfix_str(
                f"emotion {t_i}/{len(valid_targets)} {target} | cond {c_i}/{len(conditions)} {cond}"
            )
            cond_results[cond] = run_logo(df_t, target, feats, SEEDS, pbar=fit_bar)
        base = cond_results[BASELINE]
        base_mae = float(base["mae_per_seed"].mean())
        base_corr = float(np.nanmean(base["corr_per_seed"])) if not np.all(np.isnan(base["corr_per_seed"])) else np.nan

        for cond, res in cond_results.items():
            for row_i in range(len(res["y_true"])):
                per_row_records.append({
                    "Participant_ID": pid,
                    "Target_Emotion": target,
                    "Condition": cond,
                    "Row_Index": row_i,
                    "Study_Day": df_t["study_day"].iloc[row_i],
                    "Y_True": res["y_true"][row_i],
                    "Y_Pred_SeedAvg": res["y_pred"][row_i],
                    "Abs_Error_SeedAvg": res["abs_err"][row_i],
                })

            if cond == BASELINE:
                continue

            cond_mae = float(res["mae_per_seed"].mean())
            cond_corr = float(np.nanmean(res["corr_per_seed"])) if not np.all(np.isnan(res["corr_per_seed"])) else np.nan
            mae_gain_per_seed = base["mae_per_seed"] - res["mae_per_seed"]

            emotion_records.append({
                "Participant_ID": pid,
                "Target_Emotion": target,
                "Condition": cond,
                "N_Rows": len(df_t),
                "N_Days": df_t["study_day"].nunique(),
                "Baseline_MAE": base_mae,
                "Condition_MAE": cond_mae,
                "MAE_Improvement": base_mae - cond_mae,            # + = lower error than baseline
                "MAE_Improvement_Pct": 100 * (base_mae - cond_mae) / base_mae if base_mae > 0 else np.nan,
                "MAE_Improvement_SeedStd": float(np.std(mae_gain_per_seed)),
                "Baseline_Corr": base_corr,
                "Condition_Corr": cond_corr,
                "Corr_Improvement": cond_corr - base_corr,         # + = higher correlation than baseline
                "Wilcoxon_p": paired_wilcoxon(base["abs_err"], res["abs_err"]),
            })

    fit_bar.close()
    tqdm.write(
        f"[done] {pid}: {total_fits} fits in {fit_bar.format_dict['elapsed']:.1f}s "
        f"({len(valid_targets)} emotions)"
    )

# ============================================================================================
# PER PARTICIPANT x EMOTION
# ============================================================================================
df_emotion = pd.DataFrame(emotion_records)
df_emotion["Significant_p05"] = df_emotion["Wilcoxon_p"] < 0.05

# Rank the 3 single home features within each participant x emotion (all_home not ranked)
is_single = df_emotion["Condition"].isin(single_feature_conditions)
df_emotion["Rank_By_MAE_Gain"] = np.nan
df_emotion.loc[is_single, "Rank_By_MAE_Gain"] = (
    df_emotion[is_single]
    .groupby(["Participant_ID", "Target_Emotion"])["MAE_Improvement"]
    .rank(ascending=False, method="min")
)
df_emotion = df_emotion.sort_values(
    by=["Participant_ID", "Target_Emotion", "MAE_Improvement"],
    ascending=[True, True, False],
).reset_index(drop=True)
df_emotion.to_csv("home_ablation_per_participant_emotion.csv", index=False)

# ============================================================================================
# PER PARTICIPANT (averaged across emotions)
# ============================================================================================
df_participant = (
    df_emotion.groupby(["Participant_ID", "Condition"], as_index=False)
    .agg(
        N_Emotions=("Target_Emotion", "nunique"),
        Avg_Baseline_MAE=("Baseline_MAE", "mean"),
        Avg_Condition_MAE=("Condition_MAE", "mean"),
        Avg_MAE_Improvement=("MAE_Improvement", "mean"),
        Avg_MAE_Improvement_Pct=("MAE_Improvement_Pct", "mean"),
        Avg_Baseline_Corr=("Baseline_Corr", "mean"),
        Avg_Condition_Corr=("Condition_Corr", "mean"),
        Avg_Corr_Improvement=("Corr_Improvement", "mean"),
        Emotions_Improved_MAE=("MAE_Improvement", lambda x: int((x > 0).sum())),
        Emotions_Significant_p05=("Significant_p05", "sum"),
        Times_Ranked_1st=("Rank_By_MAE_Gain", lambda x: int((x == 1).sum())),
    )
    .sort_values(by=["Participant_ID", "Avg_MAE_Improvement"], ascending=[True, False])
    .reset_index(drop=True)
)
df_participant.to_csv("home_ablation_per_participant.csv", index=False)

# ============================================================================================
# GROUP LEVEL: PER EMOTION (across participants) + OVERALL
# ============================================================================================
def group_summary(g):
    gains = g["MAE_Improvement"].to_numpy()
    n_pos = int((gains > 0).sum())
    n_nonzero = int((gains != 0).sum())
    try:
        wil_p = float(wilcoxon(g["Baseline_MAE"], g["Condition_MAE"]).pvalue) if n_nonzero >= 2 else np.nan
    except ValueError:
        wil_p = np.nan
    return pd.Series({
        "N_Participants": g["Participant_ID"].nunique(),
        "Mean_Baseline_MAE": g["Baseline_MAE"].mean(),
        "Mean_Condition_MAE": g["Condition_MAE"].mean(),
        "Mean_MAE_Improvement": gains.mean(),
        "Median_MAE_Improvement": np.median(gains),
        "Median_MAE_Improvement_Pct": g["MAE_Improvement_Pct"].median(),
        "Mean_Corr_Improvement": g["Corr_Improvement"].mean(),
        "N_Improved": n_pos,
        "N_Worse": int((gains < 0).sum()),
        "Sign_Test_p": float(binomtest(n_pos, n_nonzero, 0.5).pvalue) if n_nonzero > 0 else np.nan,
        "Wilcoxon_Across_Participants_p": wil_p,
        "N_Individually_Significant_p05": int(g["Significant_p05"].sum()),
    })


summary_cols = ["Participant_ID", "Baseline_MAE", "Condition_MAE", "MAE_Improvement",
                "MAE_Improvement_Pct", "Corr_Improvement", "Significant_p05"]
df_group_emotion = (
    df_emotion.groupby(["Target_Emotion", "Condition"])[summary_cols]
    .apply(group_summary)
    .reset_index()
)
df_group_overall = (
    df_emotion.groupby("Condition")[summary_cols]
    .apply(group_summary)
    .reset_index()
    .assign(Target_Emotion="ALL")
)
df_group = (
    pd.concat([df_group_emotion, df_group_overall], ignore_index=True)
    .sort_values(by=["Target_Emotion", "Median_MAE_Improvement"], ascending=[True, False])
    .reset_index(drop=True)
)
df_group.to_csv("home_ablation_per_emotion.csv", index=False)

# ============================================================================================
# RAW PER-ROW ERRORS (for re-analysis)
# ============================================================================================
pd.DataFrame(per_row_records).to_csv("home_ablation_per_day_errors.csv", index=False)

print("\n[Success] Saved home-time ablation CSVs:")
print("  home_ablation_per_participant_emotion.csv")
print("  home_ablation_per_participant.csv")
print("  home_ablation_per_emotion.csv")
print("  home_ablation_per_day_errors.csv")
print("\nOverall (all emotions pooled):")
print(df_group_overall[["Condition", "N_Participants", "Median_MAE_Improvement", "N_Improved", "N_Worse", "Sign_Test_p"]].to_string(index=False))
