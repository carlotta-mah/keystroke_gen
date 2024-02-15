import pytest
import pandas as pd
# tests for all the functions in the program

# imports for testing module in main
from src.data_processor import read_keystroke_data
from src.data_processor import preprocess_data


def test_read_keystroke_data():
    filename = 'test/test_data.txt'
    df = read_keystroke_data(filename)
    assert len(df) == 25
    assert len(df.iloc[0]) == 6
    assert df.iloc[0]['PRESS_TIME'] == 1476488368032


def test_preprocess_data():
    # Sample DataFrame with test data
    data = {
        'PRESS_TIME': [1000, 1500, None, 2000],  # Include some missing values
        'RELEASE_TIME': [1500, 2000, 2500, None],  # Include some missing values
        'KEYCODE': [72, 69, 76, 77],
        'USER_INPUT': ['A', 'B', 'C', 'D'],
        'SEQUENCE_ID': [1, 2, 1, 2],
    }
    df = pd.DataFrame(data)

    # Call the preprocess_data function
    df_preprocessed = preprocess_data(df)

    # Check if DataFrame matches the expected shape
    assert df_preprocessed.shape == (2, 11)  # Expected shape after dropping rows with missing values
