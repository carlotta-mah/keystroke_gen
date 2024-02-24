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
    assert len(df.iloc[0]) == 5
    assert df.iloc[0]['PRESS_TIME'] == 1476488368032


def test_preprocess_data():
    # Sample DataFrame with test data
    data = {
        'PARTICIPANT_ID': [1, 2, 1, 2, 1, 2, 1, 2],
        'TEST_SECTION_ID': [1, 1, 2, 2, 3, 3, 4, 4],
        'PRESS_TIME': [1000, 1500, 1799, 2000, 2200, 2500, 2800, None],
        'RELEASE_TIME': [1500, 2000, 2500, 3000, 3500, None, 4500, 5000],
        'KEYCODE': [72, 69, 76, 77, 79, 82, 65, 66],
        'USER_INPUT': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    }
    df = pd.DataFrame(data)

    # Call the preprocess_data function
    df_preprocessed = preprocess_data(df)

    # Check if DataFrame matches the expected shape
    assert df_preprocessed.shape == (2, 11)  # Expected shape after dropping rows with missing values
