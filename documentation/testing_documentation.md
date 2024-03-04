![GHA workflow badge](https://github.com/carlotta-mah/keystroke_gen/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/carlotta-mah/keystroke_gen/graph/badge.svg?token=GUKTSYYD5L)](https://codecov.io/gh/carlotta-mah/keystroke_gen)

## Testing
To run the tests, you may run the following command in the terminal: 

```coverage run -m pytest```

Alternatively, you can run the tests via github actions on the main page of the repository.

If you wish to run the tests in the pycharm IDE, you may need change the path to the data in the `test_data` folder in the `test_data_reader.py` file to the relative path.