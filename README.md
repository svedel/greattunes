# Creative project: user-friendly Bayesian optimization library 

![CI/CD pipeline status](https://github.com/svedel/creative-brain/workflows/CI%20CD%20workflow/badge.svg)

Easy-to-use Bayesian optimization library made available for either closed-loop or user-driven (manual) optimization of either 
known or unknown objective functions. Drawing on `PyTorch` (`GPyTorch`), `BOTorch` and with proprietary extensions.

For a short primer on Bayesian optimization, see [this section](#a-primer-on-bayesian-optimization)  

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

#### Uploading build to repo servers (e.g. `PyPI`)

To be investigated. Here's [a link with help](https://docs.github.com/en/free-pro-team@latest/actions/guides/building-and-testing-python) on how to leverage `GitHub actions` for this purpose.

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
# === Step 2: solve the problem ===

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
| `train_X` | All *observed* covariates with dimensions `num_observations` X `num_covariates`. |
| `proposed_X` | All *proposed* covariate datapoints to investigate, with dimensions `num_observations` X `num_covariates`. |
| `train_Y` | All *observed* responses corresponding to the covariate points in `train_X`. Dimensions `num_observations` X 1. |
| `best_response_value` | Best *observed* response value during optimization run, including current iteration. Dimensions `num_observations` X 1. |
| `covars_best_response_value` | *Observed* covariates for best response value during optimization run, i.e. each row in `covars_best_response_value` generated the same row in `best_response_value`. Dimensions `num_observations` X `num_covariates`. |    

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
| `"Custom"` | `nu` | A custom Matérn kernel with parameter `nu` (a float). For more details on Matérn kernels see [wiki page](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function), and see the source code for the model in [`creative_project\custom_models`](creative_project/custom_models). |

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

```python
# some function to optimize
def f(x):
  ...

# parameters
max_iter = 100
rel_tol = 1e-10

# run the auto-method
cc.auto(response_samp_func=f, max_iter=max_iter, rel_tol=rel_tol)
```

In most cases the best results are found by requiring the `rel_tol` limit to be satisfied for multiple consecutive
iterations. This can be achieved by also providing the number of consecutive steps required `rel_tol_steps`. If 
`rel_tol_steps` is not provided, the limit on relative improvement only needs to be reached once for convergence.

```python
# some function to optimize
def f(x):
  ...

# parameters
max_iter = 100
rel_tol = 1e-10
rel_tol_steps = 5

# run the auto-method
cc.auto(response_samp_func=f, max_iter=max_iter, rel_tol=rel_tol, rel_tol_steps=rel_tol_steps)
```

Best practises on using `rel_tol` and `rel_tol_steps` are provided in Example 5 in [examples](examples).

### Iterative: the `.ask` and `.tell` methods

The true value of Bayesian optimization is its ability to optimize problems which cannot be formulated mathematically.
The mathematical method can work as long as a response can be generated, and in fact makes no assumptions on the 
nature of the problem (except that a maximum is present). Thus, whether the response is generated as a measurement from
an experiment, the feedback from users or the output of a defined mathematical function does not matter; all can be
optimized via the framework.

Optimization of unknown functions is handled by the methods `.ask` and `.tell`.
* `.ask` provides a best guess of the next covariate data point to sample, given the history of previously sampled points for the 
  problem (that is, `.ask` provides the output of the acquisition function)
* `.tell` is the method to report the observed covariate data point and the associated response
One call to `.ask` followed by a call to `.tell` performs one iteration of `.auto` from the point of view of the 
Bayesian optimization; the difference is only in how to interface with it. Examples 2 and 3 in [examples](examples)
shows how to use `.ask`-`.tell` to solve problems end-to-end.  

To solve a problem, apply these problems iteratively: in each iteration start by calling `.ask`, then use the proposed 
new data point to sample the system response and provide both this value and the actually sampled covariate values (can 
be different from proposed values) back via `.tell`.

```python
# in below, "cc" is an instantiated version of CreativeProject class (identical initialization as when using .auto method) 
max_iter = 20

for i in range(max_iter):
  
    # generate candidate
    cc.ask()  # new candidate is last row in cc.proposed_X

    # sample response (beware results must be formulated as torch tensors)
    observed_covars = <from measurement or from cc.proposed_X>
    observed_response = <from measurement or from specified objective function>

    # report response
    cc.tell(covars=observed_covars, response=observed_response)
```

#### Providing input via prompt

Observations of covariates and response can be provided manually to `.tell`. To do so, simply call `.tell` without any 
arguments at each iteration (all book keeping will be handled on backend)
```python
# in below, "cc" is an instantiated version of CreativeProject class (identical initialization as when using .auto method) 
max_iter = 20

for i in range(max_iter):
  
    # generate candidate
    cc.ask()  # new candidate is last row in cc.proposed_X

    # report response
    cc.tell()
```

In this case, the user will be prompted to provide input manually. There will be 3 attempts to provide covariates 
(another 3 for response), and the method will stop if not successful within these attempts. Provided input data will be
validated for number of variables and data type as part of these cycles.

Any of `covars` and `response` not provided as (named) parameter to `.tell` the user will be requested to provide via 
manual input in prompt. It is thus possible to get e.g. covariates automatically but manually read off response values
from an instrument.

#### Overriding reported values of covariates or response 

Observed covariates and observed responses are sometimes off. To override the latest datapoint for either, simply 
provide it again in the same iteration. This will automatically override the latest reported value 
```python
# in below, "cc" is an instantiated version of CreativeProject class (identical initialization as when using .auto method) 
# further assumes that at least on full iteration has been taken

# define a response
def f(x):
  ...

# generate candidate
cc.ask()  # new candidate is last row in cc.proposed_X

# first result
observed_results = torch.tensor([[it.item() for it in cc.proposed_X[-1]]], dtype=torch.double)
observed_response = torch.tensor([[f(cc.proposed_X[-1]).item()]], dtype=torch.double)

# report first response
cc.tell(covars=observed_results, response=observed_response)

# second result
observed_response_second = observed_response + 1

# update response
cc.tell(covars=observed_results, response=observed_response_second)
```


## Examples 

A number of examples showing how to use the framework in `jupyter` notebooks is available in the [examples](examples) 
folder. This includes both closed-loop and iterative usages, as well as a few real-world examples (latter to come!)


## Contributing

### Tech stack

This library is built using the following
* `torch`
* `GPyTorch`
* `BOTorch`
* `numpy`

### Access to backlog etc

To be detailed later

### Testing strategy

Regular unit and integration testing is performed during each run of the CI pipeline.

In addition, a set of sample problems are also executed. The purpose of these special integration tests is to verify
that the optimization performance of the framework remains consistent. These sample problems is a series of pre-defined
applications of the framework with known results.

#### Test tooling

The `pytest` framework is used for this library, with all tests residing in 
[`creative_project\tests`](tests). To execute all tests, run the following from the terminal from the 
project root folder (`creative_project`)
```python
~/creative_project$ python -m pytest tests/
```

The tests are available in 
* Unit tests: [`creative_project\tests\unit`](tests/unit)
* Integration tests: [`creative_project\tests\integration`](tests/integration)
* Sample problems: [`creative_project\tests\sample_problems`](tests/sample_problems)

All fixtures are collected in `tests\conftest.py` and config is specified in `tests\pytest.ini`.

#### Tests during CI
All tests are run by CI pipeline when committing to any branch. In addition, linting, style format and library import
style checks are also executed.

`sample_problems` tests are allowed to fail for regular commits but must pass for merge commits.

#### Tests during development
During development, unit and integration tests are typically sufficient to check ongoing developments. These tests can
be executed by the command
```python
~/creative_project$ python -m pytest tests/unit tests/integration
```
Before committing it is good practise to also run sample problems, which can be done by either `python -m pytest tests/`
(running all tests including sample problems), or `python -m pytest tests/sample_problems`.

#### Additional code checks 
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

## A primer on Bayesian optimization

A number of good resources are available for Bayesian optimization, so below follows only a short primer. Interested
readers are referred to the references listed below for more information.

### Basics of Bayesian optimization

Briefly and heuristically, Bayesian optimization works as follows. 
1. Define a *objective function*. The goal of the optimization is to maximize this function.
2. Define a *surrogate model*. This is an approximation of the actual functional dependencies underlying the objective
function. Because Bayesian optimization builds its own model there is no requirement that the objective function can be
   written as a mathematical expression.
3. Define an *acquisition function*. This function is applied to the surrogate model to identify the next datapoint to
sample (as such, the acquisition function is actually a functional)
4. Iterate:
    * Use the acquisition function to identify the next data point to sample.
    * Observe the response of the objective function at the proposed point  
    * Based on all observed covariates and responses of the objective function, update the surrogate model via Bayes 
      theorem and repeat. 

### Surrogate models

A typical choice of surrogate model class is the [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process), 
but this is not a strict requirement. Examples exist in which both random forest and various types of neural networks 
have been used. 

Formally, Bayesian optimization considers the function to be optimized as unknown and instead places a Bayesian prior
distribution over it. This is the initial surrogate model. Upon observing the response, the prior model is updated to 
obtain the posterior distribution of functions.

The benefit of Gaussian process models is their explicit modeling of the uncertainty and ease of obtaining the posterior.

### Acquisition functions

Acquisition functions (functionals) propose the best point to sample for a particular problem, given the
prior distribution of the surrogate model.

A number of different functions exist, with some typical ones provided in Peter Frazier's 
[Tutorial on Bayesian Optimization](https://arxiv.org/pdf/1807.02811.pdf). They typically balance exploration and 
exploitation in different ways.

## References
A list of Bayesian optimization references for later use
* [Wikipedia entry on Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)
* [borealis.ai](https://www.borealisai.com/en/blog/tutorial-8-bayesian-optimization/)
* [bayesopt, SigOpt page](http://bayesopt.github.io/)
* [Towards Data Science](https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319)
* [Gaussian processes for dummies](https://katbailey.github.io/post/gaussian-processes-for-dummies/)
* [Peter Frazier, Cornell, Bayesian Optimization expert](https://people.orie.cornell.edu/pfrazier/)
* [Tutorial on Bayesian Optimization](https://arxiv.org/pdf/1807.02811.pdf)
* [Bayesian Optimization, Martin Krasser's blog](http://krasserm.github.io/2018/03/21/bayesian-optimization/)
* [Bayesian Optimization with inequality constraints](https://stat.columbia.edu/~cunningham/pdf/GardnerICML2014.pdf)
* [Bayesian deep learning](https://towardsdatascience.com/bayesian-deep-learning-with-fastai-how-not-to-be-uncertain-about-your-uncertainty-6a99d1aa686e)
