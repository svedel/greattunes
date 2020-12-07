import setuptools

from creative_project._version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

setuptools.setup(
    name="creative_project",
    version=__version__,
    author="Søren Vedel",
    description="Toolset for easy execution of Bayesian optimization for either step-by-step or closed-loop needs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["creative_project", "creative_project.*"], exclude=["tests"]),
    package_data={"kre8_core": ["requirements.txt"]},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    #setup_requires=['pytest-runner', 'black==19.10b0', 'flake8===3.7.9', 'isort==4.3.21'],
    #tests_require=['pytest==5.3.5', 'pytest-cov==2.8.1'],
)
