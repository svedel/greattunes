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