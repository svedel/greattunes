from creative_project._version import __version__
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

setuptools.setup(
    name="bayesian_core",
    version=__version__,
    author="Søren Vedel",
    description="Toolset for easy execution of Bayesian optimization for either step-by-step or closed-loop needs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={'kre8_core': ['requirements.txt']},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
