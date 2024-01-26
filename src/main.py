#!/usr/bin/env python3
# Author: Carlotta Mahncke
import sys
def read_keystroke_data(filename):
    keystroke_data = []
    with open(filename, 'r') as file:
        next(file)  # Skip the header row
        for line in file:
            # Assuming tab-separated values
            values = line.strip().split('\t')
            # Assuming the columns are in the given order
            participant_id = values[0]
            test_section_id = values[1]
            sentence = values[2]
            user_input = values[3]
            keystroke_id = values[4]
            press_time = int(values[5])  # Convert to integer
            release_time = int(values[6])  # Convert to integer
            letter = values[7]
            keycode = int(values[8])  # Convert to integer
            keystroke_data.append((participant_id, test_section_id, sentence, user_input, keystroke_id, press_time, release_time, letter, keycode))
    return keystroke_data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        data = read_keystroke_data(filename)
        print("Number of samples:", len(data))
        print("Sample format:", data[0])  # Displaying the first sample
    except FileNotFoundError:
        print("File not found:", filename)
    except Exception as e:
        print("An error occurred:", e)