# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased] - 

Version for this release: 0.0.5

### Added
### Changed
### Deprecated
### Removed
### Fixed

## [0.0.4] - July 14, 2021

### Added
* Functionality to use integer and categorical covariates as input to the function under optimization, using the method
  of Garrido-Merchán and Hernandéz-Lobato ([journal link](https://www.sciencedirect.com/science/article/abs/pii/S0925231219315619), 
  [ArXiv preprint](https://arxiv.org/pdf/1805.03463.pdf)). This significantly extends the applicability of the 
  framework.
* Named covariates
* Pretty data format for covariates (`x_data`) and response (`y_data`) which keeps track of observations in their
natural data types (`float` for doubles, `int` for integers and `str` for categorical variables). These are in `pandas`
  format
  
* Two new end-to-end [examples](#Examples) to illustrate a simple use case of integer covariates (Example 6) and a more elaborate combining continuous, integer and categorical (Example 7).
  

### Changed
* Extended how the package determines covariates enabled via a wider range of options for the parameter `covars` provided 
  during class initialization. There are now two methods, see [README.md](README.md/#Covariates:-the-free-parameters-which-are-adjusted-by-the-framework-during-optimization): 
    1) A simple in which requires a list of tuples, with each tuple giving the guess, the minimum and the maximum of the covariate. Data types are inferred and covariate names are assigned.
    2) An elaborate that allows more control over data type and covariate naming.
* In `_best_response.current_best`: switched to storing in pretty user-facing format (`pandas` df), updated output 
slightly
* Extended `creative_project.transformed_kernel_models.transformation.GP_kernel_transformation` to support high-rank
tensors. This allows using `botorch`'s `optimize_acqf` method to determine best next covariate datapoint from 
  acquisition function.

### Deprecated
None

### Removed
In `ask`-`tell`-approach: reporting observations via `covars` and `response` entries to `tell`-method cannot be
done via the backend data format (`torch` tensor of same format as `train_X` and `train_Y`). Instead, use the same 
user-facing format (in `pandas` df) to report all entries, including integer and categorical variables in their natural
data types (`int` and `str`).

### Fixed
None

## [0.0.3] - February 25, 2021

Version number for this release: 0.0.3

### Added
* Added new random sampling functionality with two purposes. Firstly, during initialization it is known to be good to 
  start with random sampling if no data is available. Secondly, and also to ensure speedier optimization convergence, a 
  single randomly sampled point every now and then in between Bayesian points is known to increase convergence. Random 
  sampling is now available for both `auto` approach and `ask`-`tell` approach with the following features
  * During class initialization, using random initialization or not is controlled by `random_start` (default: `True`)
  * Additional parameters during initialization
    * `num_initial_random`: number of initial random; no default, if not specified will be set to $\sqrt{# dimensions}$
    * `random_sampling_method`: sampling method with options `random` (completely random) and `latin_hcs` (latin hypercube sampling); defaults to `latin_hcs`
    * `random_step_cadence`: the cadence of random sampling after initialization (default: 10)

### Changed
In `CreativeProject` class initialization:
* If historical data is added via `train_X`, `train_Y`
  * `proposed_X` has been changed to be a zero tensor of the same size as `train_X`. This replaces an empty tensor for 
    `proposed_X`, which confusingly could take any values.
  * optimization cycle counters (iteration counters) `model["covars_proposed_iter"]`, `model["covars_sampled_iter"]` 
    and `model["response_sampled_iter"]` are set so the first iterations are taken as those from the historical data. 
    That is, if `train_X`, `train_Y` is provided with two observations during initialization, then the counters are set 
    as `model["covars_proposed_iter"]=2`, `model["covars_sampled_iter"]=2` and `model["response_sampled_iter"]=2`.

### Deprecated
None

### Removed
Removed the attribute `start_from_random` as part of adding more elaborate random sampling functionality.

### Fixed
None

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