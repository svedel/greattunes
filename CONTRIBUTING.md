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

Versions used of each library are specified in [requirements.txt`](requirements.txt).

## Development and release cycles

### Branches and workflows

#### Basics
New work must pass all tests before it is merged to a release. Testing is done automatically at each push to `GitHub` 
via a CI workflow ([workflow file](#.github/workflows/testing.yml)). Additionally, testing should also be run 
locally before committing to remote to ensure that the code works. Details on testing strategy and tooling is provided 
[further below](#testing-strategy).

CI/CD workflow scripts are available in [`.github/workflows`](#.github/workflows). Access tokens for pushing to 
`test-PyPI` and `PyPI` proper are stored as secrets in `GitHub`.

#### Git workflow: Getting features ready for a release 

##### Development
Any development should be done on `development` branches and merged into the master development branch upon completion 
(incl. all tests passing). Follow good practise and create a dev branch for each feature/bug.

##### Staging
When stuff is ready for a release, merge all relevant code changes from `development` into the `staging` branch. 
As part of testing and verification, the latest version of the `greattunes` is built as part of the CI/CD and published 
to `test-PyPI` ([link](https://test.pypi.org/project/greattunes/)). The CD-part is handled by the 
[staging workflow](#.github/workflows/staging.workflow.yml). **Remember** to update the version number in 
[`greattunes/_version.py`](#greattunes/_version.py) before merging into `staging` for two reasons: version numbers 
should be consistent across `staging` and deployed versions to `PyPI` (deployed from `main`), and secondly, `test-PyPI`
and `PyPI` proper does not accept more than one push with the same version number.

### Release a new version 

Merge changes from `staging` into `main` and create a release in `GitHub`. For transparency, **make sure to use the same
version number** as set in the library itself (in [`greattunes/_version.py`](#greattunes/_version.py)) when defining the 
release on `GitHub`. Furthermore, it is **important that the release has a tag**.

Once a release is create with a tag, the code is built from the `main` branch and pushed to `PyPI` for finish the 
release using a CD [workflow script](#.github/workflows/prod.workflow.yml).

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

### Pre-commit hooks are not used due to incompatibility with `PyCharm`
This package has been developed using `PyCharm` which does not have good support for `pre-commit` and consequently these 
features have not been used.

For more on pre-commit hooks check out this introduction 
[Getting started with Python Pre-commit hooks](https://towardsdatascience.com/getting-started-with-python-pre-commit-hooks-28be2b2d09d5) 
and the `python` project [pre-commit](https://pre-commit.com/#intro).