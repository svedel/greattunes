# Contributing

## Tech stack

This library is built using the following
* `torch`
* `GPyTorch`
* `BOTorch`
* `numpy`
* `pandas`
* `matplotlib`

## Access to backlog etc

To be detailed later

## Testing strategy

Regular unit and integration testing is performed during each run of the CI pipeline. In addition, pre-commit hooks
for lint check, code format and import style guide as well as execution of all unit tests have been added.

In addition, a set of sample problems are also executed. The purpose of these special integration tests is to verify
that the optimization performance of the framework remains consistent. These sample problems is a series of pre-defined
applications of the framework with known results.

### Test tooling

The `pytest` framework is used for this library, with all tests residing in 
[`creative_project\tests`](tests). To execute all tests, run the following from the terminal from the 
project root folder (`creative_project`)
```python
~/creative_project$ python -m pytest tests/
```

The tests are available in 
* Unit tests: [`creative_project\tests\unit`](tests/unit)
* Integration and functional tests: [`creative_project\tests\integration`](tests/integration)
* Sample problems: [`creative_project\tests\sample_problems`](tests/sample_problems)

All fixtures are collected in `tests\conftest.py` and config is specified in `tests\pytest.ini`.

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

The following commands, executed in the terminal from the project root folder (`creative_project`), will run the checks 
and perform style corrections if needed
```python
# === linting ===
~/creative_project$ flake8 creative_project/

# === code style ===
~/creative_project$ black creative_project --check # checks for fixes needed
~/creative_project$ black creative_project --diff # shows suggested edits
~/creative_project$ black creative_project # makes the edits (only command needed to update the code)

# === sort imports ===
~/creative_project$ /bin/sh -c "isort creative_project/**/*.py --check-only" # checks for sorting opportunities
~/creative_project$ /bin/sh -c "isort creative_project/**/*.py --diff" # shows changes that could be done
~/creative_project$ /bin/sh -c "isort creative_project/**/*.py" # makes the changes (only command needed to update the code)
```

### Pre-commit hooks
Code checks are run before each commit via pre-commit git hooks. These include linting (`flake8`), code checks 
(`black`), import style checks (`isort`) and unit tests (all tests in `tests/unit`).

To execute pre-commit hooks, the package `pre-commit` has been added to the development requirements [`requirements-dev.txt`](requirements-dev.txt).
This runs all hooks configured in [`.pre-commit-config.yaml`](.pre-commit-config.yaml). Beware that for any changes to 
take effect, the following command must be run
```commandline
(<virtual_env>) >>> pre-commit install
```
where `(<virtual_env>)` indicates that the virtual environment is initiated.

The hooks don't show graphically when committing through PyCharm, but do show when committing from the command line.

For more details on using pre-commit hooks check ["Getting started with Python pre-commit hooks"](https://towardsdatascience.com/getting-started-with-python-pre-commit-hooks-28be2b2d09d5), 
["Automating Python workflows with pre-commit hooks"](https://towardsdatascience.com/automating-python-workflows-with-pre-commit-hooks-e5ef8e8d50bb) 
and the official page for the `pre-commit` package [https://pre-commit.com/](https://pre-commit.com/).