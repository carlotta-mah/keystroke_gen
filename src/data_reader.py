#!/usr/bin/env python3
# Author: Carlotta Mahncke
import sys
import pandas as pd
import glob
from sklearn.model_selection import train_test_split


# Step 1: Load the data
def read_keystroke_data(pattern, limit=None):
    if pattern is None:
        file = '*_keystrokes.txt'

    # columns that we want to process
    columns_to_keep = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME', 'KEYCODE', 'USER_INPUT']

    # Find files matching the specified pattern
    file_list = glob.glob(pattern)
    file_list = file_list[:limit or len(file_list)]

    keystroke_df = [pd.read_csv(file, sep='\t', usecols=columns_to_keep, parse_dates={'SEQUENCE_ID': ['PARTICIPANT_ID', 'TEST_SECTION_ID']})
                              for file in file_list]
    print(keystroke_df)
    keystroke_df = pd.concat([pd.read_csv(file, sep='\t', usecols=columns_to_keep, parse_dates={'SEQUENCE_ID': ['PARTICIPANT_ID', 'TEST_SECTION_ID']})
                              for file in file_list], ignore_index=True)
    return keystroke_df


# Step 2: Data preprocessing
# todo: encoding in sequences
def preprocess_data(keystroke_df):
    # Convert timestamps to milliseconds
    keystroke_df['PRESS_TIME'] = pd.to_numeric(keystroke_df['PRESS_TIME'], errors='coerce')
    keystroke_df['RELEASE_TIME'] = pd.to_numeric(keystroke_df['RELEASE_TIME'], errors='coerce')

    # Drop rows with missing values
    keystroke_df = keystroke_df.dropna()

    # Feature scaling
    keystroke_df[['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']] = scale_features(keystroke_df[['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']])

    return keystroke_df


# Custom feature scaling function
def scale_features(keystroke_df):
    # Calculate mean and standard deviation for each feature
    mean_values = keystroke_df.mean()
    std_values = keystroke_df.std()

    # Scale each feature
    scaled_features = (keystroke_df - mean_values) / std_values

    return scaled_features


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename> or <number of files>")
        sys.exit(1)

    filespec = sys.argv[1]
    df = None

    # if filename is number process number files
    if filespec.isdigit():
        file_pattern = '"../data/Keystrokes/files/????_keystrokes.txt'
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
