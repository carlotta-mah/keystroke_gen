#!/usr/bin/env python3
# Author: Carlotta Mahncke
import csv
import sys
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import numpy as np


# Step 1: Load the data
def read_keystroke_data(pattern, limit=None):
    if pattern is None:
        file = '*_keystrokes.txt'

    # columns that we want to process
    columns_to_keep = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']

    # Find files matching the specified pattern
    file_list = glob.glob(pattern)
    file_list = file_list[:limit or len(file_list)]
    print(file_list)

    # Read the data from each file and concatenate into a single DataFrame
    keystroke_df = pd.concat([pd.read_csv(file, sep='\t',
                                          encoding='latin-1',
                                          usecols=columns_to_keep,
                                          quoting=csv.QUOTE_NONE,
                                          parse_dates={'SEQUENCE_ID': ['PARTICIPANT_ID', 'TEST_SECTION_ID']},
                                          keep_date_col=True)
                              for file in file_list], ignore_index=True)
    return keystroke_df


# Step 2: Data preprocessing
# todo: encoding in sequences
def preprocess_data(keystroke_df):
    # Convert timestamps to milliseconds
    keystroke_df['PRESS_TIME'] = pd.to_numeric(keystroke_df['PRESS_TIME'], errors='coerce')
    keystroke_df['RELEASE_TIME'] = pd.to_numeric(keystroke_df['RELEASE_TIME'], errors='coerce')
    # Convert keycodes to integers
    keystroke_df['KEYCODE'] = keystroke_df['KEYCODE'].astype(int)

    # Drop rows with missing values
    keystroke_df = keystroke_df.dropna()


    # Feature scaling
    keystroke_df[['PRESS_TIME', 'RELEASE_TIME']] = scale_features(
        keystroke_df[['PRESS_TIME', 'RELEASE_TIME']])

    # Calculate sequence start time for each group
    keystroke_df['SEQUENCE_START_TIME'] = keystroke_df.groupby(['SEQUENCE_ID'])['PRESS_TIME'].transform('min')

    # Calculate press time relative to sequence start time and press duration
    keystroke_df['PRESS_TIME_RELATIVE'] = keystroke_df['PRESS_TIME'] - keystroke_df['SEQUENCE_START_TIME']
    keystroke_df['PRESS_DURATION'] = keystroke_df['RELEASE_TIME'] - keystroke_df['PRESS_TIME']
    keystroke_series = keystroke_df.groupby(['SEQUENCE_ID']).apply(
        lambda x: x.sort_values('PRESS_TIME')).reset_index(drop=True)

    return extract_features(keystroke_series)


def scale_features(keystroke_df):
    # Calculate mean and standard deviation for each feature
    mean_values = keystroke_df.mean()
    std_values = keystroke_df.std()

    # Scale each feature
    scaled_features = (keystroke_df - mean_values) / std_values

    return scaled_features


def encode_participant_ids(y: np.ndarray) -> np.ndarray:
    # Extract unique participant IDs from the second column of y
    participant_ids = np.unique(y)

    # Create an empty array to store the encoded labels
    encoded_labels = np.zeros((len(y), len(participant_ids)), dtype=int)

    # Populate the array with one-hot encoded labels
    for i, participant_id in enumerate(y):
        encoded_labels[i, int(participant_id)] = 1

    return encoded_labels
# Function to extract timing and keycode-based features from a series of keystrokes
def extract_features(keystrokes_series):
    # Group keystrokes by sequence ID
    grouped_df = keystrokes_series.groupby('SEQUENCE_ID')

    extracted_features = []

    for sequence_id, group in grouped_df:
        # Extract press times, durations, and keycodes for the current sequence ID
        press_times = group['PRESS_TIME'].values
        press_durations = group['PRESS_DURATION'].values
        keycodes = group['KEYCODE'].values

        # Calculate release times from press times and durations
        release_times = press_times + press_durations

        # Calculate duration for each keystroke
        durations = release_times - press_times

        # Calculate inter-key intervals (time between consecutive key presses)
        inter_key_intervals = np.diff(press_times)

        # Compute statistical characteristics of durations and inter-key intervals
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)

        mean_inter_key_interval = np.mean(inter_key_intervals)
        std_inter_key_interval = np.std(inter_key_intervals)
        min_inter_key_interval = np.min(inter_key_intervals)
        max_inter_key_interval = np.max(inter_key_intervals)

        # Extract information about keycodes
        unique_keycodes, counts = np.unique(keycodes, return_counts=True)
        most_common_keycode = unique_keycodes[np.argmax(counts)]
        num_unique_keycodes = len(unique_keycodes)

        # Additional features can be derived from keycodes, such as frequency, entropy, etc.

        # Store features in a dictionary
        features = {
            'SEQUENCE_ID': sequence_id,
            'PARTICIPANT_ID': group['PARTICIPANT_ID'].values[0],
            'mean_duration': mean_duration,
            'std_duration': std_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'mean_inter_key_interval': mean_inter_key_interval,
            'std_inter_key_interval': std_inter_key_interval,
            'min_inter_key_interval': min_inter_key_interval,
            'max_inter_key_interval': max_inter_key_interval,
            'most_common_keycode': most_common_keycode,
            'num_unique_keycodes': num_unique_keycodes
        }

        extracted_features.append(features)

    # Convert the list of dictionaries to a pandas DataFrame
    extracted_features_df = pd.DataFrame(extracted_features)

    return extracted_features_df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename> or <number of files>")
        sys.exit(1)

    filespec = sys.argv[1]
    df = None

    # if filename is number process number files
    if filespec.isdigit():
        file_pattern = '../data/Keystrokes/files/????_keystrokes.txt'
        df = read_keystroke_data(file_pattern, filespec)
    else:
        try:
            df = read_keystroke_data("../data/Keystrokes/files/" + filespec)  # todo: add path before block
        except FileNotFoundError:
            print("File not found:", filespec)
        except Exception as e:
            print("An error occurred:", e)

    if df is not None:
        df_preprocessed = preprocess_data(df)

        # Step 5: Split the data into training and testing sets
        X = df_preprocessed.drop('USER_INPUT', axis=1)
        y = df_preprocessed['USER_INPUT']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Number of samples:", len(df))
        print("Sample format:", df.iloc[0])
