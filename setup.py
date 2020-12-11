import setuptools
import subprocess

from setuptools.command.install import install
from creative_project._version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]


# tmp_requirements = [
#         'black==19.10b0',
#         'botorch==0.2.1',
#         'flake8===3.7.9',
#         'gpytorch==1.1.1',
#         'isort==4.3.21',
#         'matplotlib==3.3.2',
#         'numpy==1.19.1',
#         'pytest==5.3.5',
#         'pytest-cov==2.8.1',
#         'torch==1.6.0+cpu',
#         'torchvision==0.7.0+cpu'
#     ]
#
# def pip_install(package_name):
#     """
#     pip installs packages from 'package_name'
#     :param package_name (str): package name
#     """
#     if 'torch' in package_name:
#         subprocess.call(['python', '-m', 'pip', 'install', package_name, '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
#     else:
#         subprocess.call(['python', '-m', 'pip', 'install', package_name])
#
# class CustomInstall(install):
#     def run(self):
#         print('Inside custom install...')
#         pip_install(tmp_requirements)
#         install.run(self)
#         print('Validating...')


setuptools.setup(
    name="creative_project",
    #version=__version__,
    author="Søren Vedel",
    description="Toolset for easy execution of Bayesian optimization for either step-by-step or closed-loop needs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'black==19.10b0',
        'botorch==0.2.1',
        'flake8===3.7.9',
        'gpytorch==1.1.1',
        'isort==4.3.21',
        'matplotlib==3.3.2',
        'numpy==1.19.1',
        'pytest==5.3.5',
        'pytest-cov==2.8.1',
        #'torch==1.6.0+cpu',
        #'torchvision==0.7.0+cpu'
    ], #  @ https://download.pytorch.org/whl/torch_stable.html
    #dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
    #requirements,
    #cmdclass= {'install': CustomInstall},
    classifiers=[
        "Framework :: Torch :: 1.6.0",
        "Framework :: BOTorch :: 0.2.1",
        "Framework :: GPyTorch :: 1.1.1",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    #packages=setuptools.find_packages(include=["creative_project", "creative_project.*"]),
    #package_data={"kre8_core": ["requirements.txt"]},
    #python_requires=">=3.6",
    #setup_requires=[
    #    'black==19.10b0',
    #    'botorch==0.2.1',
    #    'flake8===3.7.9',
    #    'gpytorch==1.1.1',
    #    'isort==4.3.21',
    #    'matplotlib==3.3.2',
    #    'numpy==1.19.1',
    #    'pytest==5.3.5',
    #    'pytest-cov==2.8.1',
    #    'torch==1.6.0+cpu',
    #    'torchvision==0.7.0+cpu'
    #],
    #tests_require=['pytest==5.3.5', 'pytest-cov==2.8.1'],
)
