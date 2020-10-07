# explore-create
Library with Bayesian Optimization made available for either closed-loop or user-driven (manual) optimization of either known or unknown objective functions. Drawing on PyTorch (GPyTorch), BOTorch and with proprietary extensions.

## Start-up notes
Need to install the `torch`-libraries outside normal bulk `pip install`.

To find the right installation command for `torch`, use [this link](https://pytorch.org/get-started/locally/)
to determine the details and add as a separate command in the `github` actions yaml. As an example, the following is the 
install command on my local system (an Ubuntu-based system with 
pip and without CUDA access)
```python
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Testing 
The `pytest` framework is used for this library, with all tests residing in `creative-brain\tests`. To execute all tests, 
run the following from the terminal from the project root folder (`creative-brain`)
```python
python -m pytest tests/
```

In addition, linting, code style checking and sorting of imports is also covered. The following commands, executed in
the terminal from the project root folder (`creative-brain`), will run the checks and perform style corrections if 
needed
```python
### linting
flake8 src/

### code style
black src --check # checks for fixes needed
black src --diff # shows suggested edits
black src # makes the edits (only command needed to update the code)

### sort imports
/bin/sh -c "isort src/**/*.py --check-only" # checks for sorting opportunities
/bin/sh -c "isort src/**/*.py --diff" # shows changes that could be done
/bin/sh -c "isort src/**/*.py" # makes the changes (only command needed to update the code)
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
