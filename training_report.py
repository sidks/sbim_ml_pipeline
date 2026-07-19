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
import shap
from datetime import datetime
import random
import math
import pickle
from textwrap import wrap
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfReader, PdfWriter

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

target_lst = feeling_cols
df = df.dropna(subset = features).reset_index(drop = True)
df = df[~(df['cingo_username'] == 'appstoretester1')]
X = df[features]
y = df[target_lst] if len(target_lst) > 1 else df[target_lst[0]]
participants = df['cingo_username'].unique()

y_df = y if isinstance(y, pd.DataFrame) else pd.DataFrame({(targets if isinstance(targets, str) else targets[0]): y})
targets_list = list(y_df.columns)

############################################################################################## TRAINING OF THE MODEL ##############################################################################################

results_all = {}
participants = sorted(df["cingo_username"].unique())

# New data storage to collect feature importance metrics
feature_importance_records = []

# ============================
# tqdm ON THE FIRST LOOP HERE
# ============================
for pid in tqdm(participants, desc="Participants"):

    df_p = df[df["cingo_username"] == pid].reset_index(drop=True)
    results_all[pid] = {}

    # ==========================================
    # LOOP OVER TARGETS
    # ==========================================
    for target in targets_list:

        # --------------------------------------
        # Keep only rows where THIS target exists
        # --------------------------------------
        df_t = df_p[df_p[target].notna()].reset_index(drop=True)

        if len(df_t) < 2:
            continue

        # LOGO over study_day
        logo = LeaveOneGroupOut()
        groups = df_t["study_day"]
        n_days = groups.nunique()

        mae_list = []
        corr_list = []
        held_out_days = []
        shap_values_all = []
        shap_base_values_all = []
        shap_data_all = []
        all_y_true, all_y_pred = [], []
        X_test_list = []
        
        # Temp list to track feature importance across folds for this current target
        fold_importance_list = []

        for fold, (train_idx, test_idx) in enumerate(
            logo.split(df_t, groups=groups), 1):

            train_df = df_t.iloc[train_idx]
            test_df  = df_t.iloc[test_idx]

            held_day = test_df["study_day"].iloc[0]
            held_out_days.append(int(held_day))

            X_train = train_df[features].to_numpy()
            y_train = train_df[target].to_numpy()

            X_test  = test_df[features].to_numpy()
            y_test  = test_df[target].to_numpy()

            # reshape target for XGB
            y_train = y_train.reshape(-1, 1)
            y_test  = y_test.reshape(-1, 1)

            # MODEL
            model = XGBRegressor(
                n_estimators=50,
                learning_rate=0.01,
                max_depth=None,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1,
                reg_alpha=0.0,
                random_state=22,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0
            )

            # SHAP sampling
            X100 = shap.utils.sample(X_train, min(100, len(X_train)))

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test).reshape(-1, 1)

            # SHAP
            expl = shap.Explainer(model, X100)
            shap_vals = expl(X_test)

            shap_values_all.append(shap_vals.values)
            shap_base_values_all.append(shap_vals.base_values)
            shap_data_all.append(shap_vals.data)

            # --- EXTRACT SHAP FEATURE IMPORTANCE FOR FOLD ---
            # Take the mean absolute value of SHAP values across test samples for this fold
            fold_mean_abs_shap = np.mean(np.abs(shap_vals.values), axis=0)
            fold_importance_list.append(fold_mean_abs_shap)

            # METRICS
            fold_mae = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
            try:
                fold_corr = np.corrcoef(y_test[:, 0], y_pred[:, 0])[0, 1]
            except Exception:
                fold_corr = np.nan

            mae_list.append(fold_mae)
            corr_list.append(fold_corr)

            all_y_true.append(y_test)
            all_y_pred.append(y_pred)
            X_test_list.append(X_test)

        # Average feature importance across all folds for this specific target & participant
        if fold_importance_list:
            avg_target_importance = np.mean(np.vstack(fold_importance_list), axis=0)
            
            # Map features to their averaged calculation score
            for feat_idx, feat_name in enumerate(features):
                feature_importance_records.append({
                    "Participant_ID": pid,
                    "Target_Emotion": target,
                    "Feature_Name": feat_name,
                    "Mean_Absolute_SHAP_Importance": avg_target_importance[feat_idx]
                })

        # save results for this target
        results_all[pid][target] = {
            "mae_list": mae_list,
            "corr_list": corr_list,
            "held_out_days": held_out_days,
            "shap_values_all": shap_values_all,
            "shap_base_values_all": shap_base_values_all,
            "shap_data_all": shap_data_all,
            "y_true": all_y_true,
            "y_pred": all_y_pred,
            "X_test_list": X_test_list,
        }

# =============================================
# EXPORT FEATURE IMPORTANCES TO CSV
# =============================================
df_importance_output = pd.DataFrame(feature_importance_records)
# Sort by impact level to maximize file clarity
df_importance_output = df_importance_output.sort_values(
    by=["Participant_ID", "Target_Emotion", "Mean_Absolute_SHAP_Importance"], 
    ascending=[True, True, False]
).reset_index(drop=True)

# Save file directly
df_importance_output.to_csv("shap_feature_importances.csv", index=False)
print("\n[Success] Feature importance metrics saved to 'shap_feature_importances.csv'.")

# =============================================
# PARTICIPANT + TARGET METRICS
# =============================================

participant_metrics = {}
all_y_true_global = {t: [] for t in targets_list}
all_y_pred_global = {t: [] for t in targets_list}

for pid in results_all:

    participant_metrics[pid] = {}

    for target in results_all[pid]:

        res = results_all[pid][target]

        y_true_p = np.vstack(res["y_true"])
        y_pred_p = np.vstack(res["y_pred"])

        mae_val = mean_absolute_error(y_true_p[:, 0], y_pred_p[:, 0])
        corr_val = np.corrcoef(y_true_p[:, 0], y_pred_p[:, 0])[0, 1]

        participant_metrics[pid][target] = {
            "mae": mae_val,
            "corr": corr_val
        }

        all_y_true_global[target].append(y_true_p)
        all_y_pred_global[target].append(y_pred_p)


# =============================================
# GLOBAL METRICS
# =============================================

global_mae = {}
global_corr = {}

for target in targets_list:

    if len(all_y_true_global[target]) == 0:
        continue

    y_true = np.vstack(all_y_true_global[target])
    y_pred = np.vstack(all_y_pred_global[target])

    global_mae[target] = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    global_corr[target] = np.corrcoef(y_true[:, 0], y_pred[:, 0])[0, 1]


############################################################################################## REPORT GENERATION ##############################################################################################

# ============================================================
# LOAD CSV FILES
# ============================================================

INTEGER_CSV = "integerfeatures.csv"
GENERAL_CSV = "generalfeatures.csv"
ACCEL_CSV = "accelerationfeatures.csv"
SD_CSV = "sdfeatures.csv"
BATTERY_CSV = "batteryfeatures.csv"

integer_df = pd.read_csv(INTEGER_CSV)
general_df = pd.read_csv(GENERAL_CSV)
accel_df = pd.read_csv(ACCEL_CSV)
sd_df = pd.read_csv(SD_CSV)
battery_df = pd.read_csv(BATTERY_CSV)

# ============================================================
# CONFIG
# ============================================================

TEMPLATE_PATH = "report_template.pdf"

# ============================================================
# OUTPUT DIRECTORY SETUP
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--report_name",
    type=str,
    required=True,
    help="Base report folder name, e.g. sbim-reports-05-18-2026"
)

args = parser.parse_args()

BASE_OUTPUT_DIR = "output_reports"

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

base_report_name = args.report_name

# Main report directory
OUTPUT_DIR = os.path.join(
    BASE_OUTPUT_DIR,
    base_report_name
)

# PNG subfolder only
PNG_DIR = os.path.join(
    OUTPUT_DIR,
    "pngs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

print(f"\nSaving reports to: {OUTPUT_DIR}\n")



TOP_K = 4
FIG_W, FIG_H = 6, 4
SINGLE_COLOR = "#1f77b4"

feature_names = features
targets = targets_list

TARGET_NAME_MAP = {

    "happy_intensity": "Happiness",
    "sad_intensity": "Sadness",
    "calm_intensity": "Calm",
    "anxious_intensity": "Anxious",
    "social_intensity": "Social",
    "irritated_intensity": "Irritated",
    "bored_intensity": "Bored",
    "motivated_intensity": "Motivated"
    
}

def display_target_name(t):

    return TARGET_NAME_MAP.get(t, t)

# ============================================================
# FEATURE MAPS
# ============================================================

integer_feature_map = {}
for _, row in integer_df.iterrows():
    integer_feature_map[row["Feature Name"]] = {
        "translation": str(row.get("Translation", "")).strip(),
        "units": str(row.get("Units", "")).strip(),
        "unit_single": str(row.get("Unit_singular", "")).strip(),
        "timing": str(row.get("Timing", "")).strip(),
        "general1": str(row.get("General", "")).strip(),
        "general2": str(row.get("General2", "")).strip()
    }

general_feature_map = {}
for _, row in general_df.iterrows():
    general_feature_map[row["Feature Name"]] = {
        "translation": str(row.get("Translation", "")).strip(),
        "feature_type": str(row.get("feature_type", "")).strip(),
        # Uses 'General' as general1 in the new general features CSV
        "general1": str(row.get("General", "")).strip(),
        "general2": str(row.get("General2", "")).strip()
    }

accel_feature_map = {}
for _, row in accel_df.iterrows():
    accel_feature_map[row["Feature Name"]] = {
        "translation": str(row.get("Translation", "")).strip(),
        "timing": str(row.get("timing", "")).strip(),
        "general1": str(row.get("General", "")).strip(),
        "general2": str(row.get("General2", "")).strip()
    }

sd_feature_map = {}
for _, row in sd_df.iterrows():
    sd_feature_map[row["Feature Name"]] = {
        "translation": str(row.get("Translation", "")).strip(),
        "unit": str(row.get("Unit", "")).strip(),
        "timing": str(row.get("timing", "")).strip(),
        "general1": str(row.get("General", "")).strip(),
        "general2": str(row.get("General2", "")).strip()
    }

battery_feature_map = {}
for _, row in battery_df.iterrows():
    battery_feature_map[row["Feature Name"]] = {
        "translation": str(row.get("Translation", "")).strip(),
        "unit": str(row.get("Units", "")).strip(),
        "general1": str(row.get("General", "")).strip()
    }

# ============================================================
# HARDCODED LUX THRESHOLDS
# ============================================================

LUX_THRESHOLDS = [

    (0.002, "a moonless clear night sky"),
    (0.01, "a quarter moon"),
    (0.05, "a full moon"),
    (5, "twilight"),
    (50, "a living room"),
    (100, "a dark overcast day"),
    (320, "office lighting"),
    (400, "a sunset"),
    (1000, "an overcast day"),
    (10000, "daylight"),
    (32000, "direct sunlight")
]

# ============================================================
# HARDCODED AREA THRESHOLDS
# ============================================================

AREA_THRESHOLDS = [

    (60, "a one bedroom apartment"),
    (90, "a two bedroom apartment"),
    (250, "a house"),
    (400, "a basketball court"),
    (500, "a large movie auditorium"),
    (1500, "an ice-skating rink"),
    (3000, "a supermarket"),
    (5000, "a football field"),
    (15000, "a running track"),
    (30000, "the Colorado State Capitol building"),
    (62000, "Ball Arena"),
    (140000, "a shopping mall"),
    (320000, "the Denver Zoo"),
    (500000, "the campus of University of Denver"),
    (2500000, "the campus of the University of Colorado, Boulder"),
    (3500000, "Central Park in New York City"),
    (5000000, "the Cheyenne Mountain Zoo"),
    (13000000, "Dillon Resevoir"),
    (65000000, "the city of Boulder"),
    (140000000, "Denver International Airport"),
    (400000000, "the city of Denver")
]

# ============================================================
# HARDCODED ACCELERATION THRESHOLDS
# ============================================================

ACCELERATION_THRESHOLDS = [

    (
        (0.0001, 1.5),
        "walking"
    ),

    (
        (1.5001, 3.0),
        "running"
    ),

    (
        (3.00001, float("inf")),
        "driving"
    )
]

# ============================================================
# STANDARD DEVIATION BENCHMARKS
# ============================================================

SD_THRESHOLDS = {

    "magnitude_std": 0.13,
    "magnitude_std_working_day": 0.16,
    "magnitude_std_last_night": 0.06,
    "std_lux_mean": 1491
}

# ============================================================
# HELPERS
# ============================================================

def clean_text(x):

    if str(x).lower() == "nan":
        return ""

    return str(x).strip()

# ============================================================
# LOOKUP HELPERS
# ============================================================

def lookup_lux_parameter(value):

    for threshold, label in LUX_THRESHOLDS:

        if value <= threshold:
            return label

    return LUX_THRESHOLDS[-1][1]

def lookup_area_parameter(value):

    for threshold, label in AREA_THRESHOLDS:

        if value <= threshold:
            return label

    return AREA_THRESHOLDS[-1][1]

def lookup_acceleration_parameter(value):

    if value == 0:

        return (
            "you were not moving"
        )

    for (low, high), label in ACCELERATION_THRESHOLDS:

        if low <= value <= high:
            return label

# ============================================================
# SHAP THRESHOLD
# ============================================================

def shap_threshold(
    feature_vals,
    shap_vals,
    n_bins=5,
    min_count=2,
    k=1
):

    bins = np.linspace(
        feature_vals.min(),
        feature_vals.max(),
        n_bins + 1
    )

    bin_ids = np.digitize(
        feature_vals,
        bins
    )

    bin_centers = []
    bin_means = []

    for b in range(1, n_bins + 1):

        mask = bin_ids == b

        if mask.sum() < min_count:
            continue

        bin_centers.append(
            feature_vals[mask].mean()
        )

        bin_means.append(
            shap_vals[mask].mean()
        )

    if len(bin_centers) == 0:
        return None, None, None

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)

    thresh = None

    for i in range(len(bin_means) - k):

        if (
            bin_means[i] < 0
            and np.all(
                bin_means[i+1:i+1+k] > 0
            )
        ):

            thresh = bin_centers[i+1]
            break

        if (
            bin_means[i] > 0
            and np.all(
                bin_means[i+1:i+1+k] < 0
            )
        ):

            thresh = bin_centers[i+1]
            break

    if thresh is None:
        return None, None, None

    return (
        round(float(thresh), 2),
        bin_centers,
        bin_means
    )

# ============================================================
# RELATIONSHIP INTERPRETATION
# ============================================================

def interpret_feature_relationship(
    feature_values,
    shap_values,
    threshold
):

    before_mask = feature_values < threshold
    after_mask = feature_values >= threshold

    n_before = before_mask.sum()
    n_after = after_mask.sum()

    try:

        corr = np.corrcoef(
            feature_values,
            shap_values
        )[0, 1]

    except:

        corr = 0

    if np.isnan(corr):
        corr = 0

    positive_corr = corr >= 0

    # ========================================================
    # MORE POINTS AFTER THRESHOLD
    # ========================================================

    if n_after >= n_before:

        threshold_direction = "more than"

        if positive_corr:

            emotion_direction = "more"
            summary_direction = "More"

        else:

            emotion_direction = "less"
            summary_direction = "More"

    # ========================================================
    # MORE POINTS BEFORE THRESHOLD
    # ========================================================

    else:

        threshold_direction = "less than"

        if positive_corr:

            emotion_direction = "less"
            summary_direction = "Less"

        else:

            emotion_direction = "more"
            summary_direction = "Less"

    return {

        "threshold_direction": threshold_direction,
        "emotion_direction": emotion_direction,
        "summary_direction": summary_direction,
        "correlation": corr
    }

# ============================================================
# EMOTION WORD MAP
# ============================================================

EMOTION_WORD_MAP = {

    "happiness": "happy",
    "sadness": "sad",
    "calm": "calm",
    "anxious": "anxious",
    "social": "social",
    "irritated": "irritated",
    "bored": "bored",
    "motivated": "motivated"
    
}

def emotion_to_word(emotion):

    return EMOTION_WORD_MAP.get(
        emotion.lower(),
        emotion
    )

# ============================================================
# COUNTABLE FEATURES FOR "FEWER THAN"
# ============================================================

FEWER_THAN_FEATURES = {
    "num_incoming_messages_per_hour",
    "num_outgoing_messages_per_hour",
    "num_unique_contacts_per_hour",
    "num_total_incoming_and_outgoing_messages_daily",
    "num_incoming_calls_daily",
    "num_outgoing_calls_daily",
    "num_unique_contacts_daily",
    "keyboard_total_taps_per_hour",
    "keyboard_total_words_per_hour",
    "keyboard_input_modes",
    "keyboard_average_daily_taps_per_hour",
    "keyboard_average_daily_words_per_hour",
    "keyboard_input_modes_daily"
}

LIGHT_FEATURES = {
    "min_lux_mean",
    "max_lux_mean",
    "mean_lux_mean"
}

# ============================================================
# BUILD THRESHOLD SENTENCE
# ============================================================

def build_threshold_sentence(
    feature_name,
    thresh,
    threshold_direction,
    emotion_direction,
    emotion
):

    # Adjust grammar for countable/discrete features
    if threshold_direction == "less than" and feature_name in FEWER_THAN_FEATURES:
        threshold_direction = "fewer than"

    if feature_name in LIGHT_FEATURES and threshold_direction == "less than":
        threshold_direction = "dimmer than"
    elif feature_name in LIGHT_FEATURES and threshold_direction == "more than":
        threshold_direction = "brighter than"

    # ========================================================
    # INTEGER FEATURES
    # ========================================================

    if feature_name in integer_feature_map:

        info = integer_feature_map[
            feature_name
        ]

        translation = clean_text(
            info["translation"]
        )

        if int(round(thresh)) <= 1:
            units = clean_text(
                info["unit_single"]
            )
        else:
            units = clean_text(
                info["units"]
            )

        timing = clean_text(
            info["timing"]
        )

        emotion_word = emotion_to_word(emotion)

        sentence = (
            f"When {translation} "
            f"{threshold_direction} "
            f"{int(round(thresh))} "
            f"{units}"
            f"{' ' + timing if timing else ''}, "
            f"you were "
            f"{emotion_direction} "
            f"{emotion_word}."
        )

        return " ".join(sentence.split())

    # ========================================================
    # GENERAL FEATURES
    # ========================================================

    if feature_name in general_feature_map:

        info = general_feature_map[
            feature_name
        ]

        translation = clean_text(
            info["translation"]
        )

        feature_type = clean_text(
            info["feature_type"]
        ).upper()

        if feature_type == "LUX":

            parameter = lookup_lux_parameter(
                int(thresh)
            )

        else:

            parameter = lookup_area_parameter(
                int(thresh)
            )

        emotion_word = emotion_to_word(emotion)

        sentence = (

            f"When {translation} "
            f"{threshold_direction} "
            f"{parameter}, "
            f"you were "
            f"{emotion_direction} "
            f"{emotion_word}."
        )

        return " ".join(sentence.split())

    # ========================================================
    # SD FEATURES
    # ========================================================

    if feature_name in sd_feature_map:

        info = sd_feature_map[
            feature_name
        ]

        benchmark = SD_THRESHOLDS[
            feature_name
        ]

        translation = clean_text(
            info["translation"]
        )

        unit = clean_text(
            info["unit"]
        )

        timing = clean_text(
            info["timing"]
        )

        size_word = (
            "more"
            if thresh > benchmark
            else "less"
        )

        emotion_word = emotion_to_word(emotion)

        if feature_name == 'std_lux_mean':
            sentence = (
            f"When {translation} "
            f"{size_word}, "
            f"you were "
            f"{emotion_direction} "
            f"{emotion_word}."
        )
        else:
            sentence = (
                f"When {translation} "
                f"{size_word} "
                f"{unit}"
                f"{' ' + timing if timing else ''}, "
                f"you were "
                f"{emotion_direction} "
                f"{emotion_word}."
            )

        return " ".join(sentence.split())

    # ========================================================
    # ACCELERATION FEATURES
    # ========================================================

    if feature_name in accel_feature_map:

        info = accel_feature_map[
            feature_name
        ]

        translation = clean_text(
            info["translation"]
        )

        timing = clean_text(
            info["timing"]
        )

        parameter = lookup_acceleration_parameter(
            float(thresh)
        )

        relation_word = (
            "more"
            if threshold_direction == "more than"
            else "less"
        )

        emotion_word = emotion_to_word(emotion)

        
        if float(thresh) > 0.00:
            sentence = (
                f"When {translation} "
                f"{relation_word} than "
                f"{parameter}"
                f"{' ' + timing if timing else ''}, " 
                f"you were "
                f"{emotion_direction} "
                f"{emotion_word}."
            )
        else:
            sentence = (
                f"When "
                f"{parameter}"
                f"{' ' + timing if timing else ''}, " 
                f"you were "
                f"{emotion_direction} "
                f"{emotion_word}."
            )

        return " ".join(sentence.split())


    # ========================================================
    # BATTERY FEATURES
    # ========================================================

    if feature_name in battery_feature_map:

        info = battery_feature_map[
            feature_name
        ]

        translation = clean_text(
            info["translation"]
        )

        unit = clean_text(
            info["unit"]
        )

        emotion_word = emotion_to_word(emotion)
        
        sentence = (
            f"When {translation} "
            f"{threshold_direction} "
            f"{int(round(thresh * 100))}"
            f"{unit}, " 
            f"you were "
            f"{emotion_direction} "
            f"{emotion_word}."
        )

        return " ".join(sentence.split())

    # ========================================================
    # FALLBACK
    # ========================================================
    
    return (

        f"When {feature_name} was "
        f"{threshold_direction} "
        f"{int(round(thresh))}, "
        f"you were "
        f"{emotion_direction} "
        f"{emotion}."
    )

# ============================================================
# GENERAL SUMMARY
# ============================================================

def build_general_summary(
    feature_name,
    emotion,
    emotion_direction,
    summary_direction
):
    # Retrieve info block based on feature map placement
    info = None
    if feature_name in integer_feature_map:
        info = integer_feature_map[feature_name]
    elif feature_name in general_feature_map:
        info = general_feature_map[feature_name]
    elif feature_name in accel_feature_map:
        info = accel_feature_map[feature_name]
    elif feature_name in sd_feature_map:
        info = sd_feature_map[feature_name]
    elif feature_name in battery_feature_map:
        info = battery_feature_map[feature_name]

    if info and feature_name in battery_feature_map:
        g1 = clean_text(info["general1"])
        g2 = False
    elif info:
        g1 = clean_text(info["general1"])
        g2 = clean_text(info["general2"])
    else:
        # Fallback if feature metadata isn't found
        g1 = feature_name
        g2 = ""

    # Clean directions to lowercase for sentence flow
    e_dir = emotion_direction.lower()      # "more" or "less"
    s_dir = summary_direction.lower()      # "more" or "less"

    if s_dir == "less" and feature_name in FEWER_THAN_FEATURES:
        s_dir = "fewer"

    if s_dir == "less" and feature_name in LIGHT_FEATURES:
        s_dir = "dimmer"
    elif s_dir == "more" and feature_name in LIGHT_FEATURES:
        s_dir = "brighter"
    elif s_dir == "less" and feature_name in battery_feature_map:
        s_dir = "low"
    elif s_dir == "more" and feature_name in battery_feature_map:
        s_dir = "high"

    emotion_word = emotion_to_word(emotion)

    # Construct the base clause
    # sentence = f"When you were {e_dir} {emotion_word}, {g1}"
    # sentence = f"When {g1} {s_dir}"
    
    # Append summary direction and general2 if general2 exists
    if g2:
        sentence = f"When {g1} {s_dir} {g2}, you were {e_dir} {emotion_word}." 
        # sentence += f" {s_dir} {g2}"
    else:
        # If there's no General2 context, fall back cleanly
        sentence = f"When {g1} {s_dir}, you were {e_dir} {emotion_word}."
        # sentence += f" {s_dir}"

    # Clean double spaces and ensure grammatical standard formatting
    return " ".join(sentence.split()).strip()

# ============================================================
# BAR PLOT
# ============================================================

EMOTION_COLORS = {

    "happy_intensity": "#FFD700",      # yellow
    "sad_intensity": "#1f77b4",        # blue
    "calm_intensity": "#800080",       # purple
    "anxious_intensity": "#FFA500",    # orange
    "social_intensity": "#40E0D0",     # teal
    "irritated_intensity": "#d62728",   # red
    "bored_intensity": "#808080",      # grey
    "motivated_intensity": "#2ca02c"  # green
        
}

def save_bar_plot(df_pid, path):

    means = [
        df_pid[t].dropna().mean()
        for t in targets
    ]

    labels = [
        display_target_name(t)
        for t in targets
    ]

    colors = [
        EMOTION_COLORS.get(t, "#1f77b4")
        for t in targets
    ]

    plt.figure(figsize=(8, 4))

    bars = plt.bar(
        labels,
        means,
        color=colors
    )

    for bar, val in zip(bars, means):

        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom"
        )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

# ============================================================
# DEPENDENCE PLOT
# ============================================================

def save_dependence_plot(
    i,
    shap_vals,
    shap_data,
    thresh,
    path
):

    plt.figure(figsize=(FIG_W, FIG_H))

    shap.dependence_plot(
        i,
        shap_vals,
        shap_data,
        feature_names=feature_names,
        interaction_index=None,
        show=False
    )

    plt.ylabel("Impact on emotion")

    plt.axhline(
        0,
        linestyle="--",
        alpha=0.4
    )

    if thresh is not None:

        plt.axvline(
            thresh,
            color="red",
            linestyle="--",
            alpha=0.8
        )

        y_top = plt.ylim()[1]

        plt.text(
            thresh,
            y_top * 1.05,
            f"{int(round(thresh))}",
            color="red",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

# ============================================================
# PDF OVERLAY HELPER
# ============================================================

def create_overlay(draw_fn):

    packet = BytesIO()

    c = canvas.Canvas(
        packet,
        pagesize=letter
    )

    draw_fn(c)

    c.save()

    packet.seek(0)

    return PdfReader(packet)

# ============================================================
# HEADER PHRASE MAP
# ============================================================

HEADER_PHRASE_MAP = {

    "happiness": {
        "more": "more happiness",
        "less": "less happiness"
    },

    "sadness": {
        "more": "more sadness",
        "less": "less sadness"
    },

    "social": {
        "more": "being more social",
        "less": "being less social"
    },

    "calm": {
        "more": "feeling calmer",
        "less": "feeling less calm"
    },

    "motivated": {
        "more": "feeling more motivated",
        "less": "feeling less motivated"
    },

    "bored": {
        "more": "feeling more bored",
        "less": "feeling less bored"
    },

    "anxious": {
        "more": "feeling more anxious",
        "less": "feeling less anxious"
    },

    "irritated": {
        "more": "feeling more irritated",
        "less": "feeling less irritated"
    }
}

# ============================================================
# MAIN REPORT GENERATION
# ============================================================

def generate_overlay_reports():

    template_reader = PdfReader(
        TEMPLATE_PATH
    )

    for pid in results_all.keys():

        print(
            f"Generating overlay for {pid}"
        )

        writer = PdfWriter()

        df_pid = df[
            df["cingo_username"] == pid
        ]

        # ====================================================
        # PAGE 1
        # ====================================================

        bar_path = (
            f"{PNG_DIR}/{pid}_bar.png"
        )

        save_bar_plot(
            df_pid,
            bar_path
        )

        def draw_page1(c):

            c.drawImage(
                bar_path,
                70,
                130,
                width=470,
                height=300
            )

        overlay = create_overlay(
            draw_page1
        )

        base_page = copy.copy(
            template_reader.pages[0]
        )

        base_page.merge_page(
            overlay.pages[0]
        )

        writer.add_page(base_page)

        # ====================================================
        # OTHER PAGES
        # ====================================================

        EMOTION_ORDER = [
            "happy_intensity",
            "sad_intensity",
            "calm_intensity",
            "anxious_intensity",
            "social_intensity",
            "irritated_intensity",
            "bored_intensity",
            "motivated_intensity"
        ]
        
        # reorder targets based on desired order (keep only valid ones)
        ordered_targets = [t for t in EMOTION_ORDER if t in targets]

        emotion_chunks = [

            ordered_targets[i:i+4]

            for i in range(
                0,
                len(ordered_targets),
                4
            )
        ]

        for chunk_idx, chunk in enumerate(
            emotion_chunks
        ):

            def draw_page(c):

                x_positions = [40, 330]
                y_positions = [720, 380]

                for idx, t in enumerate(chunk):

                    col = idx % 2
                    row = idx // 2

                    x_base = x_positions[col]
                    y_base = y_positions[row]

                    if row == 1:
                        y_base -= 30

                    if t not in results_all[pid]:
                        continue

                    # ====================================================
                    # POSITION CUSTOMIZATIONS (COORDINATE SHIFTS)
                    # ====================================================
                    
                    # Page 2 (chunk_idx == 0) adjustments:
                    if chunk_idx == 0:
                        # Move Emotion 1 (col 0, row 0) and Emotion 3 (col 0, row 1) to the left
                        if col == 0:
                            x_base -= 15  # Shift 15 points to the left

                    # Page 3 (chunk_idx == 1) adjustments:
                    elif chunk_idx == 1:
                        # Move Emotion 7 (col 0, row 1) and Emotion 8 (col 1, row 1) up
                        if row == 1:
                            y_base += 15  # Shift 20 points up

                    res = results_all[pid][t]

                    shap_vals = np.vstack(
                        res["shap_values_all"]
                    )

                    shap_data = np.vstack(
                        res["shap_data_all"]
                    )

                    mean_abs_shap = np.mean(
                        np.abs(shap_vals),
                        axis=0
                    )

                    top_idx = np.argsort(
                        mean_abs_shap
                    )[::-1][:TOP_K]

                    valid_features = []
                    threshold_cache = {}

                    for i_feat in top_idx:

                        thresh, bx, by = shap_threshold(
                            shap_data[:, i_feat],
                            shap_vals[:, i_feat]
                        )

                        if thresh is not None:

                            valid_features.append(
                                i_feat
                            )

                            threshold_cache[i_feat] = (
                                thresh,
                                bx,
                                by
                            )

                        if len(valid_features) == 3:
                            break

                    c.setFont(
                        "Helvetica-Bold",
                        14
                    )

                    c.drawString(
                        x_base,
                        y_base,
                        display_target_name(t)
                    )

                    if len(valid_features) == 0:

                        c.setFont(
                            "Helvetica-Oblique",
                            18
                        )

                        c.drawString(
                            x_base,
                            y_base - 80,
                            "Insufficient data collected."
                        )

                        continue

                    # ====================================================
                    # PRIMARY FEATURE
                    # ====================================================

                    plot_idx = valid_features[0]

                    thresh, bx, by = threshold_cache[
                        plot_idx
                    ]

                    relationship = interpret_feature_relationship(
                        shap_data[:, plot_idx],
                        shap_vals[:, plot_idx],
                        thresh
                    )

                    threshold_direction = relationship[
                        "threshold_direction"
                    ]

                    emotion_direction = relationship[
                        "emotion_direction"
                    ]

                    summary_direction = relationship[
                        "summary_direction"
                    ]

                    emotion = display_target_name(
                        t
                    ).lower()

                    feature_name = feature_names[
                        plot_idx
                    ]

                    sentence = build_threshold_sentence(
                        feature_name,
                        thresh,
                        threshold_direction,
                        emotion_direction,
                        emotion
                    )

                    # ====================================================
                    # TEXT
                    # ====================================================

                    c.setFont(
                        "Helvetica",
                        8
                    )

                    wrapped = wrap(
                        sentence,
                        width=60
                    )

                    y_text = y_base - 15

                    for line in wrapped:

                        c.drawString(
                            x_base,
                            y_text,
                            line
                        )

                        y_text -= 10

                    # ====================================================
                    # GRAPH
                    # ====================================================

                    img_path = (
                        f"{PNG_DIR}/"
                        f"{pid}_{t}.png"
                    )

                    save_dependence_plot(
                        plot_idx,
                        shap_vals,
                        shap_data,
                        thresh,
                        img_path
                    )

                    graph_top = y_text - 23
                    graph_bottom = graph_top - 150

                    c.drawImage(
                        img_path,
                        x_base,
                        graph_bottom,
                        width=230,
                        height=180
                    )

                    # ====================================================
                    # SUMMARY HEADER
                    # ====================================================

                    y_sum = max(
                        graph_bottom - 15,
                        50
                    )

                    emotion_phrase = HEADER_PHRASE_MAP.get(
                        emotion,
                        {}
                    ).get(
                        emotion_direction,
                        f"{emotion_direction} {emotion}"
                    )
                    
                    header = (
                        f"Also..."
                    )

                    wrapped_header = wrap(
                        header,
                        width=60
                    )

                    for line in wrapped_header:

                        c.drawString(
                            x_base,
                            y_sum,
                            line
                        )

                        y_sum -= 10

                    y_sum -= 5

                    # ====================================================
                    # SUMMARY BULLETS
                    # ====================================================

                    for rank, i_feat in enumerate(
                        valid_features[1:],
                        start=1
                    ):

                        thresh, bx, by = threshold_cache[
                            i_feat
                        ]

                        relationship = interpret_feature_relationship(
                            shap_data[:, i_feat],
                            shap_vals[:, i_feat],
                            thresh
                        )

                        summary_direction = relationship[
                            "summary_direction"
                        ]

                        feature_name = feature_names[
                            i_feat
                        ]

                        # --- UPDATED CALL ---
                        summary_text = (
                            f"{rank}. "
                            f"{build_general_summary(feature_name, emotion, emotion_direction, summary_direction)}"
                        )

                        wrapped_summary = wrap(
                            summary_text,
                            width=60
                        )

                        for line in wrapped_summary:

                            c.drawString(
                                x_base,
                                y_sum,
                                line
                            )

                            y_sum -= 10
                            
            overlay = create_overlay(
                draw_page
            )

            base_page = copy.copy(
                template_reader.pages[
                    min(
                        chunk_idx + 1,
                        len(template_reader.pages)-1
                    )
                ]
            )

            base_page.merge_page(
                overlay.pages[0]
            )

            writer.add_page(base_page)

        # ====================================================
        # SAVE FINAL PDF
        # ====================================================

        output_path = (
            f"{OUTPUT_DIR}/"
            f"{pid}_report.pdf"
        )

        with open(output_path, "wb") as f:

            writer.write(f)

# ============================================================
# RUN
# ============================================================

generate_overlay_reports()
