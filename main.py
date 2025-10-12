import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

_spike_times = []

def open_file():
    try:
        # Open the JSON file in read mode ('r')
        with open('gg2013.json', 'r') as file:
            data = json.load(file)
            return data

    except FileNotFoundError:
        print("Error: The file 'data.json' was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file. The file might be malformed.")

def timestamp_spikes():
    df = pd.DataFrame(open_file())
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")

    df["minute"] = df["timestamp"].dt.floor("min")

    df.to_csv("tweets_processed.csv")

    tweet_volume = df.groupby("minute").size().reset_index(name="tweet_count")

    peaks, properties = find_peaks(tweet_volume["tweet_count"], prominence=5, distance=3)

    # Extract the times of detected spikes
    spike_times = tweet_volume.loc[peaks, "minute"]
    spike_values = tweet_volume.loc[peaks, "tweet_count"]

    _spike_times = spike_times

    # --- Step 2: Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(tweet_volume["minute"], tweet_volume["tweet_count"], marker="o", linestyle="-", label="Tweet Volume")

    # Highlight spikes
    plt.scatter(spike_times, spike_values, color="red", s=80, label="Spikes")

    # Optional: annotate spike times
    for t, v in zip(spike_times, spike_values):
        plt.text(t, v + 1, t.strftime("%H:%M"), rotation=45, ha="right", va="bottom", fontsize=9)

    plt.title("Tweet Activity Spikes Over Time")
    plt.xlabel("Time (minute)")
    plt.ylabel("Number of Tweets")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

timestamp_spikes()