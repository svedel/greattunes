import subprocess

import setuptools

from creative_project._version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

# print("Trying to install pytorch and torchvision!")
# code = 1
# try:
#     code = subprocess.call(
#         [
#             "pip",
#             "install",
#             "torch==1.6.0+cpu",
#             "torchvision==0.7.0+cpu",
#             "-f",
#             "https://download.pytorch.org/whl/torch_stable.html",
#         ]
#     )
#     if code != 0:
#         raise Exception("Torch and torchvision installation failed!")
# except:
#     try:
#         code = subprocess.call(
#             [
#                 "pip3",
#                 "install",
#                 "torch==1.6.0+cpu",
#                 "torchvision==0.7.0+cpu",
#                 "-f",
#                 "https://download.pytorch.org/whl/torch_stable.html",
#             ]
#         )
#         if code != 0:
#             raise Exception("Torch and torchvision installation failed!")
#     except:
#         print(
#             "Failed to install pytorch, please install pytorch and torchvision manually by following the simple instructions over at: https://pytorch.org/get-started/locally/"
#         )
# if code == 0:
#     print(
#         "Successfully installed pytorch and torchvision CPU version! (If you need the GPU version, please install it manually, checkout the mindsdb docs and the pytroch docs if you need help)"
#     )


setuptools.setup(
    name="bayesian_core",
    version=__version__,
    author="Søren Vedel",
    description="Toolset for easy execution of Bayesian optimization for either step-by-step or closed-loop needs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["creative_project", "creative_project.*"]),
    package_data={"kre8_core": ["requirements.txt"]},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    #setup_requires=['pytest-runner'],
    #tests_require=['pytest'],
)
