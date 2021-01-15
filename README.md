# Creative project: user-friendly Bayesian optimization library 

![CI/CD pipeline status](https://github.com/svedel/creative-brain/workflows/CI%20CD%20workflow/badge.svg)

Easy-to-use Bayesian Optimization library made available for either closed-loop or user-driven (manual) optimization of either 
known or unknown objective functions. Drawing on `PyTorch` (`GPyTorch`), `BOTorch` and with proprietary extensions.

## Features

* Optimization of either *known* or *unknown* functions. The allows for optimization of e.g. real-world experiments 
  without specifically requiring a model of the system be defined a priori.
* Focus on ease of use: only few lines of code required for full Bayesian optimization.
* Simple interface.
* Erroneous observations of either covariates or response can be overridden during optimization. 
* Well-documented code with detailed end-to-end examples of use, see [examples](#examples).
* Optimization can start from scratch or repurpose existing data.


### Design decisions

* **Multivariate covariates, univariate system response:** It is assumed that input covariates (the independent 
  variables) can be either multivariate or univariate, while the system response (the dependent variable) is only 
  univariate.
* **System-generated or manual input:** Observations of covariates and responses during optimization can be provided 
  both programmatically or manually via prompt input.
* **Optimizes known and unknown response functions:** Both cases where the response function can be formulated 
  mathematically and cases where the response can only be measured (e.g. a real-life experiment) can be 
  optimized.  
* **Observed covariates can vary from the proposed covariates:** The optimization routine at each iteration proposes 
  new covariate data points to investigate, but there is no requirement that this is also the observed data point.
  At each iteration step, proposed covariates, observed covariates and observed response are 3 separate entities.
* **Data stored in class instance:** Data for *proposed covariate data points*, *observed covariates* and *observed 
  responses* is stored in the instantiated class object.
* **Data format and type validation:** Input data is validated at all iterations.
* **Observations of covariates and response can be overridden during execution:** If an observation of either covariates 
  or response seems incorrect, the framework allows overriding the previous observation.  
* **Consistency in number of covariates and observations:** It is assumed that there is consistency in the number of 
  observations of covariates and responses: at each step a new covariate data point is proposed, before observations
  of covariates and response *for this iteration* are reported (specifically the number of proposed data points cannot 
  exceed the number of observed covariates by more than 1, and the number of observed covariates also cannot exceed the
  number of observed responses by more than 1). If additional data is provided for either observed covariates or 
  observed response, this will override the last provided data. 


## Installation

### First install `torch` dependencies

**Installing `torch` dependencies is a requirement.** Unfortunately `torch`-libraries have to be installed outside 
normal bulk `pip install -r requirements.txt`.

To find the right installation command for `torch`, use [this link](https://pytorch.org/get-started/locally/)
to determine the details and add as a separate command in the `github` actions yaml. As an example, the following is the 
install command on my local system (an `Ubuntu`-based system with `pip` and without `CUDA` access)
```python
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Install library

Currently the code is not available on any repo servers except the private GitHub account. The best way to install the
code (after adding `torch` and `torchvision`) is follow this series of steps.

1. Upgrade local versions of packaging libraries
```python
pip install --upgrade setuptools wheel
```
2. Clone this repo
3. Do local installation
```python
python -m pip install <path_to_repo>/kre8_core/
```

Step 3 will install by running `kre8_core/setup.py` locally and installing. This step can also be broken into two, 
which might improve debugging
```python
python3 <path_to_repo>/kre8_core/ setup.py bdist_wheel
python -m pip install <path_to_repo>/kre8_core/dist/creative_project-<version>-py3-none-any.whl
```
where `<version>` is the latest version in normal `python` format of `MAJOR.MINOR[.MICRO]` 
(check `/dist`-folder to see which one to pick).


## Using the framework

All capabilities of the framework are described below.

For readers wanting to skip directly to working with the framework, a number of examples of how to use the framework end-to-end are included as Jupyter notebooks in [examples](#examples).  

### Solving a problem

Solving an optimization problem consists of two steps in this framework:
1. Define the input variables (covariates), the surrogate model type and the acquisition function. Also define the
   response function if this is known
2. Optimize based on closed-loop or iterative interface

Here's a simple illustration of how to do this for a known function `f`.

#### Step 1: Define the problem

The critical things to define in this step are
* The number of covariates. Upper and lower limits must be provided for each covariate to constraint the search space, 
  and initial guess for each to be provided as well. Works for both univariate and multivariate covariate structures.
* The type of surrogate model. The model will be fitted at each step of the optimization.
* The type of acquisitition function. This will also be fitted at each step of the optimization.

```python
# import library
from creative_project import CreativeProject

# === Step 1: define the input ===

# specify covariate. For each covariate of the model provide best guess of starting point together with upper and lower
# limit 
x_start = 0.5  # initial guess
x_min = 0  # lower limit
x_max = 1  # upper limit
covars = [(x_start, x_min, x_max)]

# initialize the class
cls = CreativeProject(covars=covars, model="SingleTaskGP", acq_func="EI")
```

#### Step 2: Solve the problem

In order to optimize, we must first describe *which* function we want to do this for. The framework works both when this
function can be formulated mathematically and when it can only be sampled (e.g. through examples) but cannot be
formulated. For an illustrate of the latter see Example 2 under [examples](#examples).

Here we will work with a known objective function to optimize
```python
# univariate function to optimize
import torch

def f(x):
    return -(6 * x - 2) ** 2 * torch.sin(12 * x - 4)
```
Beware that the number of covariates (including their range) specified by `covars` under Step 1 must comply with the
functional dependence of the objective function (`x` in the case above).

We are now ready to solve the problem. We will run for `max_iter`=20 iterations.
```python
# run the auto-method
    cc.auto(response_samp_func=f, max_iter=max_iter)
```

Had we worked with an objective function `f` which could not be formulated explicitly, the right entrypoint would have
been to use the `.ask`-`.tell` methods instead of `.auto`.

### Key attributes

The following key attributes are stored for each optimization as part of the instantiated class 

| Attribute | Comments |
| --------- | -------- |
| `train_X` | All *observed* covariates with dimensions `num_observations` $\times$ `num_covariates`. |
| `proposed_X` | All *proposed* covariate datapoints to investigate, with dimensions `num_observations` $\times$ `num_covariates`. |
| `train_Y` | All *observed* responses corresponding to the covariate points in `train_X`. Dimensions `num_observations` $\times$ 1. |
| `best_response_value` | Best *observed* response value during optimization run, including current iteration. Dimensions `num_observations` $\times$ 1. |
| `covars_best_response_value` | *Observed* covariates for best response value during optimization run, i.e. each row in `covars_best_response_value` generated the same row in `best_response_value`. Dimensions $`num_observations` $\times$ `num_covariates`. |    

### Initialization options

#### Multivariate covariates

Multivariate covariates are set via the (mandatory) `covars` parameter during class initialization. Each covariate is 
given as a 3-tuple of parameters (`<initial_guess>`,`<parameter_minimum>`, `<parameter_maximum>`) (the order matters!), with `covars` being a
list of these tuples. As an example, for a cases with 3 covariates, the `covars` parameter would be

```python
covars = [(1, 0, 4.4), (5.2, 1.5, 7.0), (4, 2.2, 5.1)]
```

The order of the covariates matters since framework does not work with named covariates. Hence, the parameter defined 
by the first tuple in `covars` will always have to be reported as the first covariate when iterating during 
optimization, the second covariate will be initialized by the second tuple in `covars` etc.  

Observations of multivariate covariates are specified as columns in the `train_X` attribute (format: `torch.tensor`), 
with observations added as rows. As an example, the initial guess for the three covariates defined by `covars` above
would be
```python
train_X = torch.tensor([[1, 5.2, 4]], dtype=torch.double)
```

#### Starting with some historical data

If historical data for pairs of covariates and response is available for your system, this can be added during
initialization. In this case the optimization framework will have a better starting position and will likely converge
more quickly.

Historical data is added during class initialization. The number of observations (rows) of covariates and response must
match. Historical training data is added during class instantiation via arguments `train_X=<>` and `train_Y=<>` as
illustrated below for the following cases
1. Multiple observations of multivariate system
2. Single observation of univariate system
3. Single observation of multivariate system

```python
# import
import torch
from creative_project import CreativeProject

### ------ Case 1 - multiple observations (multivariate) ------ ###

# set range of data
covars = [(1, 0, 4.4), (5.2, 1.5, 7.0), (4, 2.2, 5.1)]

# define initial data
X = torch.tensor([[1, 2, 3],[3, 4.4, 5]], dtype=torch.double)
Y = torch.tensor([[33],[37.8]], dtype=torch.double)

# initialize class
cls = CreativeProject(covars=covars,train_X=X, train_Y=Y)

### ------ Case 2 - single observation (univariate) ------ ###

# set range of data
covars = [(1, 0, 4.4)]

# define initial data
X = torch.tensor([[1]], dtype=torch.double)
Y = torch.tensor([[33]], dtype=torch.double)

# initialize class
cls = CreativeProject(covars=covars,train_X=X, train_Y=Y)

### ------ Case 3 - single observation (multivariate) ------ ###

# set range of data
covars = [(1, 0, 4.4), (5.2, 1.5, 7.0), (4, 2.2, 5.1)]

# define initial data
X = torch.tensor([[1, 2, 3]], dtype=torch.double)
Y = torch.tensor([[33]], dtype=torch.double)

# initialize class
cls = CreativeProject(covars=covars,train_X=X, train_Y=Y)
```

#### Kernels for Gaussian process surrogate model

The following kernels for Gaussian process surrogate model are implemented. Listed parameters are provided as input to 
class initialization

| Model name | Parameters | Comments |
| ---------- | ---------- | -------- |
| `"SingleTaskGP"` | N/A | A single-task exact kernel for Gaussian process regression. Follow this link for [more details](https://botorch.org/api/models.html#module-botorch.models.gp_regression). |
| `"Custom"` | `nu` | A Matérn kernel with parameter `nu` (a float). For more details on Matérn kernels see [wiki page](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function) |

#### Acquisition functions

These acquisition functions are currently available

| Acquisition function name | Comments |
| ------------------------- | -------- |
| `"EI"` | Expected improvement acquisition function. For more details [see here](https://botorch.org/api/acquisition.html#module-botorch.acquisition.analytic). |

### Closed-loop: the `.auto` method

Closed-loop optimization refers to situations where the function is known and therefore can iterate itself to 
optimality. These are addressed via the `.auto` method, which takes a function handle `response_samp_func` as well as a 
maximum number of iterations `max_iter` as input parameters. See the [example above](#Step-2:-Solve-the-problem) as 
illustration of how to use the method.

#### Stopping based on relative improvement in best observed response: `rel_tol` and `rel_tol_steps`

The optimization can be stopped before `max_iter` steps have been taken by specifying the limit on the relative 
improvement in best observed response value (`best_response_value`). This is invoked by providing the parameter 
`rel_tol` to the `.auto` method. 

In most cases the best results are found by requiring the `rel_tol` limit to be satisfied for multiple consecutive
iterations. This can be achieved by also providing the number of consecutive steps required `rel_tol_steps`. If 
`rel_tol_steps` is not provided, the limit on relative improvement only needs to be reached once for convergence.

Best practises on using `rel_tol` and `rel_tol_steps` are provided in Example 5 in [examples](examples).

### Iterative: the `.ask` and `.tell` methods

Describe how to use + how with unknown function

#### Providing input via prompt









NOTE: the new response data counter ("how many responses do we have") is derived from the number of proposed 
    covariates, not the number of sampled responses. This in order to allow for a covariate to be reported after the
    response. However, only when .ask-method is rerun will can new covariates and responses be added.

## Advanced uses --- perhaps remove this? 

## Examples 

A number of examples showing how to use the framework in `jupyter` notebooks is available in the [examples](examples) 
folder. This includes both closed-loop and iterative usages, as well as a few real-world examples (latter to come!)

## Contributing

### Tech stack

### Access to backlog etc

### Testing strategy

# OLD BELOW HERE

## Start-up notes
Need to install the `torch`-libraries outside normal bulk `pip install`.

To find the right installation command for `torch`, use [this link](https://pytorch.org/get-started/locally/)
to determine the details and add as a separate command in the `github` actions yaml. As an example, the following is the 
install command on my local system (an `Ubuntu`-based system with 
`pip` and without `CUDA` access)
```python
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Installation
Currently the code is not available on any repo servers except the private GitHub account. The best way to install the
code (after adding `torch` and `torchvision`) is follow this series of steps.

1. Upgrade local versions of packaging libraries
```python
pip install --upgrade setuptools wheel
```
2. Clone this repo
3. Do local installation
```python
python -m pip install <path_to_repo>/kre8_core/
```

Step 3 will install by running `kre8_core/setup.py` locally and installing. This step can also be broken into two, 
which might improve debugging
```python
python3 <path_to_repo>/kre8_core/ setup.py bdist_wheel
python -m pip install <path_to_repo>/kre8_core/dist/creative_project-<version>-py3-none-any.whl
```
where `<version>` is the latest version in normal `python` format of `MAJOR.MINOR[.MICRO]` 
(check `/dist`-folder to see which one to pick).

### Uploading build to repo servers (e.g. `PyPI`)

To be investigated. Here's [a link with help](https://docs.github.com/en/free-pro-team@latest/actions/guides/building-and-testing-python) on how to leverage `GitHub actions` for this purpose.

## Testing 
The `pytest` framework is used for this library, with all tests residing in `creative-brain\tests`. Tests consist of 
unit tests, integration tests and sample problems, where the latter is a series of pre-defined applications of the 
framework with known results.

To execute all tests, run the following from the terminal from the project root folder (`creative-brain`)
```python
~/creative-brain$ python -m pytest tests/
```

### Tests during CI
All tests are run by CI pipeline when committing to any branch. 

### Tests during development
During development, unit and integration tests are typically sufficient to check ongoing developments. These tests can
be executed by the command
```python
~/creative-brain$ python -m pytest tests/unit tests/integration
```
Before committing it is good practise to also run sample problems, which can be done by either `python -m pytest tests/`
(running all tests including sample problems), or `python -m pytest tests/sample_problems`.

### Additional code-level checks 
In addition, linting, code style checking and sorting of imports is also covered. The following commands, executed in
the terminal from the project root folder (`creative-brain`), will run the checks and perform style corrections if 
needed
```python
### linting
~/creative-brain$ flake8 creative_project/

### code style
~/creative-brain$ black creative_project --check # checks for fixes needed
~/creative-brain$ black creative_project --diff # shows suggested edits
~/creative-brain$ black creative_project # makes the edits (only command needed to update the code)

### sort imports
~/creative-brain$ /bin/sh -c "isort creative_project/**/*.py --check-only" # checks for sorting opportunities
~/creative-brain$ /bin/sh -c "isort creative_project/**/*.py --diff" # shows changes that could be done
~/creative-brain$ /bin/sh -c "isort creative_project/**/*.py" # makes the changes (only command needed to update the code)
```

## References
A list of Bayesian optimization references for later use
* [borealis.ai](https://www.borealisai.com/en/blog/tutorial-8-bayesian-optimization/)
* [bayesopt, SigOpt page](http://bayesopt.github.io/)
* [Towards Data Science](https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319)
* [Gaussian processes for dummies](https://katbailey.github.io/post/gaussian-processes-for-dummies/)
* [Peter Frazier, Cornell, Bayesian Optimization expert](https://people.orie.cornell.edu/pfrazier/)
* [Tutorial on Bayesian Optimization](https://arxiv.org/pdf/1807.02811.pdf)
* [Bayesian Optimization, Martin Krasser's blog](http://krasserm.github.io/2018/03/21/bayesian-optimization/)
* [Bayesian Optimization with inequality constraints](https://stat.columbia.edu/~cunningham/pdf/GardnerICML2014.pdf)
* [Bayesian deep learning](https://towardsdatascience.com/bayesian-deep-learning-with-fastai-how-not-to-be-uncertain-about-your-uncertainty-6a99d1aa686e)
