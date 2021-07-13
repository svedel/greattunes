"""
NOTE: installing torch and torchvision via dependencies with setuptools is proving difficult. Seems it's unable to
resolve the dependecies of torch distributions (seems to be well-known issue from online investigations).

Currently users will have to first install torch and torchvision before installing this package. Upside is that this
will allow users to install the version of torch that suits their need of hardware and operating system.

More details on installing torch is given in README.md or by following this link:
https://pytorch.org/get-started/locally/
"""

import setuptools

from greattunes._version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]


def req_remove(requirements, remove_str):
    """
    remove any line from requirements containing remove_str
    :param requirements (list of strings): each string a package install requirement (e.g. 'flake8==3.7.9')
    :param remove_str (str): string pattern
    :return: updated list of requirements
    """
    new_req = []
    for req in requirements:
        if remove_str not in req:
            new_req.append(req)
    return new_req


def req_extend(requirements, target_str, extend_str):
    """
    extend any line from requirements containing target_str with extend_str, i.e. new output will be
    <match to target_str><extend_str>
    :param requirements (list of strings): each string a package install requirement (e.g. 'flake8==3.7.9')
    :param target_str (str): string pattern
    :param extend_str (str): string to add
    :return: updated list of requirements
    """
    counter = 0
    new_req = requirements
    for req in requirements:
        if target_str in req:
            new_req[counter] = req + extend_str
        counter += 1
    return new_req


requirements = req_remove(requirements, "--find-link")
requirements = req_remove(requirements, "torch==1.6.0+cpu")
requirements = req_remove(requirements, "torchvision==0.7.0+cpu")

# proposed way of installing torch (which doesn't seem to work)
# requirements = req_extend(requirements, 'torch==1.6.0+cpu', ' @ https://download.pytorch.org/whl/torch_stable.html ')
# requirements = req_extend(requirements, 'torchvision==0.7.0+cpu',
# ' @ https://download.pytorch.org/whl/torch_stable.html ')

setuptools.setup(
    name="greattunes",
    version=__version__,
    author="Søren Vedel",
    description="Toolset for easy execution of Bayesian optimization for either step-by-step or closed-loop needs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # dependency_links=["https://download.pytorch.org/whl/torch_stable.html"],
    install_requires=requirements,
    classifiers=[
        "Framework :: Torch",
        "Framework :: BOTorch",
        "Framework :: GPyTorch",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(
        include=["greattunes", "greattunes.*"]
    ),
    package_data={"greattunes": ["requirements.txt"]},
    python_requires=">=3.7",
    url="https://github.com/svedel/greattunes",
)
