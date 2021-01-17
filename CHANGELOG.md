# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Version number for this release: 0.0.2

### Added
* In `.tell`-method:
    * Optional functionality to provide observations of covariates and response programmatically (provide as input
      parameters `covars` and `response`)
* In `.auto`-method: 
    * Optional functionality to stop based on relative improvement of best response detected by the algorithm. Users can 
      stop the algorithm as soon as the relative improvement of the best response drops below a user-specified limit in 
      order to improve the speed of reaching an answer. See Example 5 in `examples` for illustration.
* Examples of end-to-end workflows of using the library as Jupyter notebooks are added in `examples` folder with descriptions.


### Changed
* Extended the README.md of the repo to describe usage, design decisions, repo content and how to contribute

### Deprecated
None

### Removed
None

### Fixed
None

## [0.0.1] - December 30, 2020

Version number for first release: 0.0.1

### Added
* `setup.py` to wrap repo as a package.
* `CHANGELOG.md` to keep track of changes going forward.
* `__str__` and `__repr__` methods to core user-facing method `creative_project.CreativeProject` for improved 
developer and user experience.

### Changed
* Added conditional to build pipeline so `sample_problems` only need to pass when merging pull requests in order
for code to be considered passing. 

### Deprecated
None

### Removed
None

### Fixed
None