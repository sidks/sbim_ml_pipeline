import pandas as pd
import matplotlib.pyplot as plt

accel_df = pd.read_csv(
    "/srv/repos/raddlab_datascience/cingo-db-data-extractor/output/test7_acceleration_2026-03-27_to_2026-06-15.csv"
)

accel_df["captured_at"] = pd.to_datetime(
    accel_df["captured_at"],
    format="mixed",
    utc=True,
)

# convert to participant timezone
accel_df["local_time"] = (
    accel_df["captured_at"]
    .dt.tz_convert("America/Denver")
)

accel_df["date"] = accel_df["local_time"].dt.date

accel_df["hour"] = (
    accel_df["local_time"].dt.hour
    + accel_df["local_time"].dt.minute / 60
)

plt.figure(figsize=(14, 8))

plt.scatter(
    accel_df["date"],
    accel_df["hour"],
    s=5,
)

dates = sorted(accel_df["date"].unique())

for d in dates:

    # 18:00 -> 24:00
    plt.fill_between(
        [d, d],
        18,
        24,
        alpha=0.15,
    )

    # 00:00 -> 14:00
    plt.fill_between(
        [d, d],
        0,
        14,
        alpha=0.15,
    )

plt.ylabel("Hour of Day")
plt.xlabel("Date")
plt.title("Acceleration Sample Coverage")

plt.ylim(0, 24)

plt.show()
