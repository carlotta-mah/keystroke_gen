import os

import pytest
import pandas as pd
import numpy as np
import torch

from src.data_processor import KeystrokeDataReader
from src.ks_dataset import KeystrokeDataset


def test_read_keystroke_data():
    data_reader = KeystrokeDataReader()
    filename = 'src/test/data/test_data.txt'
    df = data_reader.read_keystroke_data(filename)
    assert len(df) == 25
    assert len(df.iloc[0]) == 5
    assert df.iloc[0]['PRESS_TIME'] == 1476488368032


def test_preprocess_data():
    data_reader = KeystrokeDataReader()
    # Sample DataFrame with test data
    data = {
        'PARTICIPANT_ID': [1, 1, 2, 2, 1, 1, 2, 2],
        'TEST_SECTION_ID': [1, 1, 1, 1, 2, 2, 2, 2],
        'PRESS_TIME': [1000, 1500, 1799, 2000, 2200, 2500, 2800, None],
        'RELEASE_TIME': [1500, 2000, 2500, 3000, 3500, None, 4500, 5000],
        'KEYCODE': [72, 69, 76, 77, 79, 82, 65, 66],
        'USER_INPUT': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    }
    df = pd.DataFrame(data)

    # Call the preprocess_data function
    df_preprocessed = data_reader.preprocess_data(df)

    # Check if DataFrame matches the expected shape
    assert df_preprocessed.shape == (2, 11)  # Expected shape after dropping rows with missing values
    assert df_preprocessed['num_unique_keycodes'][0] == 2


def test_save_keystroke_dataset():
    data_reader = KeystrokeDataReader()
    # if file exsists already, delete it
    if os.path.exists('data/train_test_dataset.pt'):
        os.remove('data/train_test_dataset.pt')
    if os.path.exists('data/classify_test_dataset.pt'):
        os.remove('data/classify_test_dataset.pt')
    data = {
        'PARTICIPANT_ID': [1, 1, 2, 2, 1, 1, 2, 2],
        'TEST_SECTION_ID': [1, 1, 1, 1, 2, 2, 2, 2],
        'PRESS_TIME': [20, 20, 10, 10, 20, 20, 10, None],
        'RELEASE_TIME': [20, 20, 10, 10, 20, None, 10, 10],
        'KEYCODE': [1, 1, 3, 4, 1, 1, 3, 4],
        'USER_INPUT': ['A', 'A', 'C', 'D', 'A', 'A', 'C', 'D'],
    }
    df = pd.DataFrame(data)
    data_reader.save_keystroke_dataset(df, filename='test_dataset.pt', dir='data')
    # see if the file is created
    assert os.path.exists('data/train_test_dataset.pt')
    assert os.path.exists('data/classify_test_dataset.pt')

    cd = torch.load('data/classify_test_dataset.pt')
    td = torch.load('data/train_test_dataset.pt')
    assert td.data.shape == (2, 10)  # 2 complete sequences, 10 features
    assert td.data[0][9] == 1
    assert td.data[1][9] == 2


def test_get_classify_data():
    data_reader = KeystrokeDataReader()
    # if file exsists already, delete it
    if os.path.exists('data/classify_test_dataset.pt'):
        os.remove('data/classify_test_dataset.pt')

    data_c = [0, 0, 0, 0, 0, 0]
    label_c = [0, 0, 0, 1, 1, 1]
    ks = KeystrokeDataset(data_c, label_c)
    torch.save(ks, 'data/classify_test_dataset.pt')

    cd = data_reader.get_classify_data(filename='test_dataset.pt', dir='data')

    assert cd.data == data_c
    assert cd.labels == label_c


def test_get_train_data():
    data_reader = KeystrokeDataReader()
    # if file exsists already, delete it
    if os.path.exists('data/train_test_dataset.pt'):
        os.remove('data/train_test_dataset.pt')

    data_t = [1, 1, 1, 0, 0, 0]
    label_t = [0, 0, 0, 1, 1, 1]
    ks = KeystrokeDataset(data_t, label_t)
    torch.save(ks, 'data/train_test_dataset.pt')

    td = data_reader.get_train_data(filename='test_dataset.pt', dir='data')

    assert td.data == data_t
    assert td.labels == label_t


def test_scale_features():
    data_reader = KeystrokeDataReader()
    # Sample DataFrame with test data
    data = {
        'PRESS_TIME': [1000, 1500, 1799, 2000, 2200, 2500, 2800, 2200],
        'RELEASE_TIME': [1500, 2000, 2500, 3000, 3500, 2500, 4500, 5000],
        'KEYCODE': [72, 69, 76, 77, 79, 82, 65, 66],
    }
    df = pd.DataFrame(data)

    # Call the preprocess_data function
    df_preprocessed = data_reader._scale_features(df)

    # Check if DataFrame matches the expected shape
    assert np.isclose(df_preprocessed['PRESS_TIME'].mean(), 0.0)
    assert np.isclose(df_preprocessed['PRESS_TIME'].std(), 1.0)
    assert np.isclose(df_preprocessed['RELEASE_TIME'].mean(), 0.0)
    assert np.isclose(df_preprocessed['RELEASE_TIME'].std(), 1.0)
    assert np.isclose(df_preprocessed['KEYCODE'].mean(), 0.0)
    assert np.isclose(df_preprocessed['KEYCODE'].std(), 1.0)


def test_extract_features():
    data_reader = KeystrokeDataReader()
    # Sample DataFrame with test data
    data = {
        'PARTICIPANT_ID': [1, 1, 2, 2, 1, 1, 2],
        'TEST_SECTION_ID': [1, 1, 1, 1, 2, 2, 2],
        'PRESS_DURATION': [1000, 1500, 1799, 2000, 2200, 2500, 2800],
        'PRESS_TIME': [1500, 2000, 2500, 3000, 3500, 3000, 4500],
        'KEYCODE': [72, 72, 76, 77, 79, 82, 65],
        'USER_INPUT': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'SEQUENCE_ID': [1, 1, 2, 2, 3, 3, 4],
    }
    df = pd.DataFrame(data)

    # Call the preprocess_data function
    extracted_features_df = data_reader._extract_features(df)

    # Extract features
    assert extracted_features_df.shape == (3, 11)  # Expected shape after dropping rows with missing values
    assert extracted_features_df['mean_duration'][0] == np.mean([1000, 1500])
    assert extracted_features_df['std_duration'][0] == np.std([1000, 1500])
    assert extracted_features_df['min_duration'][0] == np.min([1000, 1500])
    assert extracted_features_df['max_duration'][0] == np.max([1000, 1500])

    inter_key_intervals = np.diff([1500, 2000])
    assert extracted_features_df['mean_inter_key_interval'][0] == np.mean(inter_key_intervals)
    assert extracted_features_df['std_inter_key_interval'][0] == np.std(inter_key_intervals)
    assert extracted_features_df['min_inter_key_interval'][0] == np.min(inter_key_intervals)
    assert extracted_features_df['max_inter_key_interval'][0] == np.max(inter_key_intervals)
    assert extracted_features_df['most_common_keycode'][0] == 2
    assert extracted_features_df['num_unique_keycodes'][0] == 1
    assert extracted_features_df['PARTICIPANT_ID'][0] == 1
