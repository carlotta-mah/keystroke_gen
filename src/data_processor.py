#!/usr/bin/env python3
# Author: Carlotta Mahncke
import csv
import os
import sys
import pandas as pd
import glob
import numpy as np
import torch

from src.ks_dataset import KeystrokeDataset


class KeystrokeDataReader:
    def __init__(self):
        pass

    # Function to get share of previously saved data
    def get_train_data(self, filename='keystroke_dataset.pt', dir: str = '../data'):
        return torch.load(dir + '/train_' + filename)

    # Function to get share of previously saved data
    def get_classify_data(self, filename='keystroke_dataset.pt', dir: str = '../data'):
        return torch.load(dir + '/classify_' + filename)

    # saves preprocessed data in easlily readable format (torch dataset)
    def save_keystroke_dataset(self, df: pd.DataFrame, train_split: int = 10, filename: str = 'keystroke_dataset.pt',
                               dir: str = '../data'):
        sentence_ks_data = self.preprocess_data(df)
        # split series into 10 train sentences and the other sentences
        grouped_ks_data = sentence_ks_data.groupby('PARTICIPANT_ID')

        # for each series in groupby, take the first 10 sentences
        train_sentence_ks_data, classify_ks_data = grouped_ks_data.head(train_split).reset_index(
            drop=True), grouped_ks_data.tail(
            -train_split).reset_index(drop=True)
        train_dataset, classify_dataset = (
            KeystrokeDataset(train_sentence_ks_data.drop(columns=['PARTICIPANT_ID'], axis=1).values,
                             train_sentence_ks_data['PARTICIPANT_ID'].values),
            KeystrokeDataset(classify_ks_data.drop(columns=['PARTICIPANT_ID'], axis=1).values,
                             classify_ks_data['PARTICIPANT_ID'].values))
        torch.save(train_dataset, dir + '/train_' + filename)
        torch.save(classify_dataset, dir + '/classify_' + filename)

    # reads keystroke data from file
    def read_keystroke_data(self, pattern='*_keystrokes.txt', limit=None):
        '''Read keystroke data from files matching the specified pattern and return as a DataFrame.
        '''

        # columns that we want to process
        columns_to_keep = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']

        print('Reading keystroke data from files matching the pattern:', pattern)
        print("current working directory: ", os.getcwd())
        # Find files matching the specified pattern
        file_list = glob.glob(pattern)
        file_list = file_list[:limit or len(file_list)]

        if len(file_list) == 0:
            sys.exit('No files found matching the specified pattern.')
        if len(file_list) == 1:
            keystroke_df = pd.read_csv(file_list[0], sep='\t',
                                       encoding='latin-1',
                                       usecols=columns_to_keep,
                                       quoting=csv.QUOTE_NONE,
                                       )
        else:
            # Read the data from each file and concatenate into a single DataFrame
            keystroke_df = pd.concat([pd.read_csv(file, sep='\t',
                                                  encoding='latin-1',
                                                  usecols=columns_to_keep,
                                                  quoting=csv.QUOTE_NONE,
                                                  )
                                      for file in file_list], ignore_index=True)

        return keystroke_df

    # Function that provides preprocessed data to outside
    def preprocess_data(self, keystroke_df):
        '''Preprocess keystroke data.
        Features are scaled and additional features are extracted.
        '''

        # Create a unique sequence ID for each group
        keystroke_df['SEQUENCE_ID'] = keystroke_df['PARTICIPANT_ID'].astype(str) + keystroke_df[
            'TEST_SECTION_ID'].astype(str)
        keystroke_df = keystroke_df.drop(columns=['TEST_SECTION_ID'])

        # Convert timestamps to milliseconds
        keystroke_df['PRESS_TIME'] = pd.to_numeric(keystroke_df['PRESS_TIME'], errors='coerce')
        keystroke_df['RELEASE_TIME'] = pd.to_numeric(keystroke_df['RELEASE_TIME'], errors='coerce')
        # Convert keycodes to integers
        keystroke_df['KEYCODE'] = pd.to_numeric(keystroke_df['KEYCODE'], errors='coerce')

        # Drop rows with missing values
        keystroke_df = keystroke_df.dropna()

        # Feature scaling
        keystroke_df.loc[:, ['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']] = self._scale_features(
            keystroke_df[['PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']])

        # Calculate sequence start time for each group

        keystroke_df['SEQUENCE_START_TIME'] = keystroke_df.groupby(['SEQUENCE_ID'])['PRESS_TIME'].transform('min')

        # Calculate press time relative to sequence start time and press duration
        keystroke_df['PRESS_TIME_RELATIVE'] = keystroke_df['PRESS_TIME'] - keystroke_df['SEQUENCE_START_TIME']
        keystroke_df['PRESS_DURATION'] = keystroke_df['RELEASE_TIME'] - keystroke_df['PRESS_TIME']
        keystroke_series = keystroke_df.groupby(['SEQUENCE_ID']).apply(
            lambda x: x.sort_values('PRESS_TIME')).reset_index(drop=True)
        return self._extract_features(keystroke_series)

    # helper function to scale features around zero
    def _scale_features(self, keystroke_df):
        # Calculate mean and standard deviation for each feature
        mean_values = keystroke_df.mean()
        std_values = keystroke_df.std()

        # Scale each feature
        scaled_features = (keystroke_df - mean_values) / std_values

        return scaled_features

    # Function to extract timing and keycode-based features from a series of keystrokes
    def _extract_features(self, keystrokes_series):
        '''Extract keystroke features from a series of keystrokes.'''

        # Group keystrokes by sequence ID
        grouped_df = keystrokes_series.groupby('SEQUENCE_ID')
        extracted_features = []
        for sequence_id, group in grouped_df:
            if len(group) < 2:
                continue
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
            most_common_keycode = unique_keycodes[np.argmax(counts)] if len(counts) > 1 else counts[0]
            num_unique_keycodes = len(unique_keycodes)

            # Store features in a dictionary
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
