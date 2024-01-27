#!/usr/bin/env python3
# Author: Carlotta Mahncke
import sys
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

# Step 1: Load the data
def read_keystroke_data(file_pattern, limit=None):
    # columns that we want to process
    columns_to_keep = ['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE', 'USER_INPUT']

    # Find files matching the specified pattern
    file_list = glob.glob("../data/Keystrokes/files/" + file_pattern)
    file_list = file_list[:limit or len(file_list)]

    # Initialize an empty list to store DataFrames
    df_list = []

    # Read data from each file and append to the list
    for file in file_list:
        df = pd.read_csv(file, sep='\t', usecols=columns_to_keep)
        df_list.append(df)

    return  pd.concat(df_list, ignore_index=True)


# Step 2: Data preprocessing
def preprocess_data(df):
    # Convert timestamps to milliseconds
    df['PRESS_TIME'] = pd.to_numeric(df['PRESS_TIME'], errors='coerce')
    df['RELEASE_TIME'] = pd.to_numeric(df['RELEASE_TIME'], errors='coerce')

    # Drop rows with missing values
    df = df.dropna()

    # Feature scaling
    df[['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']] = scale_features(df[['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']])

    return df

# Custom feature scaling function
def scale_features(df):
    # Calculate mean and standard deviation for each feature
    mean_values = df.mean()
    std_values = df.std()

    # Scale each feature
    scaled_features = (df - mean_values) / std_values

    return scaled_features

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename> or <number of files>")
        sys.exit(1)

    filespec = sys.argv[1]
    df = None

    #if filename is number process number files
    if filespec.isdigit():
        file_pattern = '????_keystrokes.txt'
        df = read_keystroke_data(file_pattern, filespec)
    else:
        try:
            df = read_keystroke_data(filespec)
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