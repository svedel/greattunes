# Contributing

The repo on `GitHub` is open and contributions are welcomed. The backlog can be found via the 
[`PyPI` page](https://pypi.org/project/greattunes/) or accessed directly at 
[open issues](https://github.com/svedel/greattunes/issues).

## Tech stack

This library is built using the following
* `torch`
* `GPyTorch`
* `BoTorch`
* `numpy`
* `pandas`
* `matplotlib`

## Development and deployment cycles

## Testing strategy

Regular unit, integration and functional testing is performed during each run of the CI pipeline. In addition, linting, 
code format and import style guide as well as execution of all unit tests have been added.

In addition, a set of sample problems are also executed. The purpose of these special integration tests is to verify
that the optimization performance of the framework remains consistent. These sample problems is a series of pre-defined
applications of the framework with known results.

Finally, the set of end-to-end example problems in notebook format available under `/examples` is executed as part of
CI pipeline. These notebooks will install the latest available version of the `greattunes` library from `PyPI` before 
running.

### Test tooling

The following stack is used for testing, see [`requirements-dev.txt`](requirements-dev.txt) for details of versions etc:
* **Unit, functional and integration tests**: `pytest` with settings kept in `tests/pytest.ini` and making use of `fixtures` 
  (collected in `conftest.py`)
* **Code coverage**: `pytest-cov`
* **Linting and code style checks**: `flake8` and `black` for static checks, with settings (also aligning the two tools) 
  in  `setup.cfg`
* **Import style checks**: `isort` is used to ensure consistent library import style

### Running tests locally

All tests are residing in [`creative_project\tests`](tests). To execute all tests, run the following from the terminal 
from the project root folder (`creative_project`)
```python
~/creative_project$ python -m pytest tests/
```

The tests are available in 
* Unit tests: [`creative_project\tests\unit`](tests/unit)
* Integration and functional tests: [`creative_project\tests\integration`](tests/integration)
* Sample problems: [`creative_project\tests\sample_problems`](tests/sample_problems)

All fixtures are collected in `tests\conftest.py` and config is specified in `tests/pytest.ini`.

### Tests during CI
All tests are run by CI pipeline when committing to any branch. In addition, linting, style format and library import
style checks are also executed.

`sample_problems` tests are allowed to fail for regular commits but must pass for merge commits.

### Tests during development
During development, unit and integration tests are typically sufficient to check ongoing developments. These tests can
be executed by the command
```python
~/creative_project$ python -m pytest tests/unit tests/integration
```
Before committing it is good practise to also run sample problems, which can be done by either `python -m pytest tests/`
(running all tests including sample problems), or `python -m pytest tests/sample_problems`.

### Additional code checks 
In addition, linting, code style checking and sorting of imports is also executed by the CI pipeline. The library is
currently using `flake8` for linting, `black` for code format checking and `isort` to manage library import style.

The following commands, executed in the terminal from the project root folder (`greattunes`), will run the checks 
and perform style corrections if needed
```python
# === linting ===
~/greattunes$ flake8 creative_project/

# === code style ===
~/greattunes$ black creative_project --check # checks for fixes needed
~/greattunes$ black creative_project --diff # shows suggested edits
~/greattunes$ black creative_project # makes the edits (only command needed to update the code)

# === sort imports ===
~/greattunes$ /bin/sh -c "isort greattunes/**/*.py --check-only" # checks for sorting opportunities
~/greattunes$ /bin/sh -c "isort greattunes/**/*.py --diff" # shows changes that could be done
~/greattunes$ /bin/sh -c "isort greattunes/**/*.py" # makes the changes (only command needed to update the code)
```

### Pre-commit hooks are not used
This package has been developed using `PyCharm` which does not have good support for `pre-commit` and consequently these 
features have not been used.

For more on pre-commit hooks check out this introduction 
[Getting started with Python Pre-commit hooks](https://towardsdatascience.com/getting-started-with-python-pre-commit-hooks-28be2b2d09d5) 
and the `python` project [pre-commit](https://pre-commit.com/#intro).