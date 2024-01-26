
import pytest
import sys
# tests for all the functions in the program

# test for read_keystroke_data in main.py
from  src.main import read_keystroke_data

def test_read_keystroke_data():
    filename = 'test_data.txt'
    data = read_keystroke_data(filename)
    assert len(data) == 25
    assert len(data[0]) == 9