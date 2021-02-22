# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Version number for this release: 0.0.3

### Added
* Added new random sampling functionality with two purposes. Firstly, during initialization it is known to be good to start with random sampling if no data is available. Secondly, and also to ensure speedier optimization convergence, a single randomly sampled point every now and then in between Bayesian points is known to increase convergence. Random sampling is now available for both `auto` approach and `ask`-`tell` approach with the following features
    * During class initialization, using random initialization or not is controlled by `random_start` (default: `True`)
    * Additional parameters during initialization 
        * `num_initial_random`: number of initial random; no default, if not specified will be set to $\sqrt{# dimensions}$
        * `andom_sampling_method`: sampling method with options `random` (completely random) and `latin_hcs` (latin hypercube sampling); defaults to `latin_hcs` 
        * `random_step_cadence`: the cadence of random sampling after initialization (default: 10)
         

### Changed

### Deprecated

### Removed
Removed the attribute `start_from_random` as part of adding more elaborate random sampling functionality.

### Fixed


## [0.0.2] - January 17, 2021

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