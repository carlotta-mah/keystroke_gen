#!/usr/bin/env python3
# Author: Carlotta Mahncke
import csv
import sys
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
import torch

from src.ks_dataset import KeystrokeDataset


def get_train_data():
    return torch.load('../data/keystroke_train_dataset.pt')


def get_classify_data():
    return torch.load('../data/keystroke_classify_dataset.pt')


def save_keystroke_dataset(df: pd.DataFrame, train_split: int = 10, filename: str = 'keystroke_dataset.pt'):
    sentence_ks_data = preprocess_data(df)
    # split series into 10 train sentences and the other sentences
    grouped_ks_data = sentence_ks_data.groupby('PARTICIPANT_ID')

    # for each series in groupby, take the first 10 sentences
    train_sentence_ks_data, classify_ks_data = grouped_ks_data.head(10).reset_index(drop=True), grouped_ks_data.tail(
        -10).reset_index(drop=True)
    train_dataset, classify_dataset = (
    KeystrokeDataset(train_sentence_ks_data.drop(columns=['PARTICIPANT_ID'], axis=1).values,
                     train_sentence_ks_data['PARTICIPANT_ID'].values),
    KeystrokeDataset(classify_ks_data.drop(columns=['PARTICIPANT_ID'], axis=1).values,
                     classify_ks_data['PARTICIPANT_ID'].values))
    torch.save(train_dataset, '../data/keystroke_train_dataset.pt')
    torch.save(classify_dataset, '../data/keystroke_classify_dataset.pt')


def read_keystroke_data(pattern, limit=None):
    '''Read keystroke data from files matching the specified pattern and return as a DataFrame.
    '''

    # If no pattern is specified, use the default pattern
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
                                          )
                              for file in file_list], ignore_index=True)

    return keystroke_df


# Step 2: Data preprocessing
# todo: encoding in sequences
def preprocess_data(keystroke_df, train_split=10):
    '''Preprocess keystroke data.
    Features are scaled and additional features are extracted.
    :param train_split: '''

    # Create a unique sequence ID for each group
    keystroke_df['SEQUENCE_ID'] = keystroke_df['PARTICIPANT_ID'] + keystroke_df['TEST_SECTION_ID']
    keystroke_df = keystroke_df.drop(columns=['TEST_SECTION_ID'])

    # Convert timestamps to milliseconds
    keystroke_df['PRESS_TIME'] = pd.to_numeric(keystroke_df['PRESS_TIME'], errors='coerce')
    keystroke_df['RELEASE_TIME'] = pd.to_numeric(keystroke_df['RELEASE_TIME'], errors='coerce')
    # Convert keycodes to integers
    keystroke_df['KEYCODE'] = pd.to_numeric(keystroke_df['KEYCODE'], errors='coerce')

    # Drop rows with missing values
    keystroke_df = keystroke_df.dropna()

    # Feature scaling
    keystroke_df.loc[:, ['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']] = scale_features(
        keystroke_df[['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']])

    # Calculate sequence start time for each group

    keystroke_df['SEQUENCE_START_TIME'] = keystroke_df.groupby(['SEQUENCE_ID'])['PRESS_TIME'].transform('min')

    # Calculate press time relative to sequence start time and press duration
    keystroke_df['PRESS_TIME_RELATIVE'] = keystroke_df['PRESS_TIME'] - keystroke_df['SEQUENCE_START_TIME']
    keystroke_df['PRESS_DURATION'] = keystroke_df['RELEASE_TIME'] - keystroke_df['PRESS_TIME']
    keystroke_series = keystroke_df.groupby(['SEQUENCE_ID']).apply(
        lambda x: x.sort_values('PRESS_TIME')).reset_index(drop=True)
    data = extract_features(keystroke_series)
    # split data in train sentences
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return data


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
    '''Extract keystroke features from a series of keystrokes.'''

    # Group keystrokes by sequence ID
    grouped_df = keystrokes_series.groupby('SEQUENCE_ID')

    extracted_features = []
    for sequence_id, group in grouped_df:
        # Extract press times, durations, and keycodes for the current sequence ID
        press_times = group['PRESS_TIME'].values
        press_durations = group['PRESS_DURATION'].values
        keycodes = group['KEYCODE'].values

        # Calculate duration for each keystroke
        # Calculate inter-key intervals (time between consecutive key presses)
        # if there is only one key press, the inter-key interval is 0
        inter_key_intervals = np.diff(press_times)

        # Compute statistical characteristics of durations and inter-key intervals
        mean_duration = np.mean(press_durations)
        std_duration = np.std(press_durations)
        min_duration = np.min(press_durations)
        max_duration = np.max(press_durations)

        mean_inter_key_interval = np.mean(inter_key_intervals)
        std_inter_key_interval = np.std(inter_key_intervals)
        min_inter_key_interval = np.min(inter_key_intervals)
        max_inter_key_interval = np.max(inter_key_intervals)

        # Extract information about keycodes
        unique_keycodes, counts = np.unique(keycodes, return_counts=True)
        most_common_keycode = unique_keycodes[np.argmax(counts)]
        num_unique_keycodes = len(unique_keycodes)

        # Store features in a dictionary
        # todo: select important features
        features = {
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
    # example usage
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
