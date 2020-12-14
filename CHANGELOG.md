# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Removed

### Fixed