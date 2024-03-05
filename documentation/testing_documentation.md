![GHA workflow badge](https://github.com/carlotta-mah/keystroke_gen/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/carlotta-mah/keystroke_gen/graph/badge.svg?token=GUKTSYYD5L)](https://codecov.io/gh/carlotta-mah/keystroke_gen)

## Testing
To run the tests, you may run the following command in the terminal: 

```coverage run -m pytest```

Alternatively, you can run the tests via github actions on the main page of the repository.

If you wish to run the tests in the pycharm IDE, you may need change the path to the data in the `test_data` folder in the `test_data_reader.py` file to the relative path.

## Test Coverage
With the command above, or in this document, you can also see the current test coverage. For the `data_reader_test` the test coverage is not high. 
However, every function is tested. 
The tests do not cover a wide range of different inputs, as they are only called from functions within the same file, which guarantee the absence of None values for example.
This should be improved in the future.
