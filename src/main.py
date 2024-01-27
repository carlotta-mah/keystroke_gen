#!/usr/bin/env python3
# Author: Carlotta Mahncke
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load the data
def read_keystroke_data(filename):
    columns_to_keep = ['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE', 'USER_INPUT']
    df = pd.read_csv(filename, sep='\t', usecols=columns_to_keep)
    return df

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
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        df = read_keystroke_data(filename)
        df_preprocessed = preprocess_data(df)

        # Step 5: Split the data into training and testing sets
        X = df_preprocessed.drop('USER_INPUT', axis=1)
        y = df_preprocessed['USER_INPUT']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Number of samples:", len(df))
        print("Sample format:", df.iloc[0])
    except FileNotFoundError:
        print("File not found:", filename)
    except Exception as e:
        print("An error occurred:", e)