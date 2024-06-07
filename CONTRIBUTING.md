# Introduction

First off, thank you for considering contributing to open-universe.
This project itself was built thanks to generous people like you sharing the
fruit of their labour.

The guidelines in this document should help you through sending your first pull request.
Following the guidelines will ensure a smooth experience for yourself as well
as the project maintainers.

This project has two intendend goals.
1. Provide an open implementation of the [UNIVERSE]() speech enhancement algorithm which was not provided with the original paper.
2. Provide an implementation for the improved version [UNIVERSE++]() that we built on top of the original one.
3. Allow easy use of these models with pre-trained weights on Huggingface.
We are not looking to grow the project beyond this scope.
If you would like to integrate these algorithms into your own framework, or further transform them, please feel free to copy the code (but keep the license) or fork the repo.

On the other hand, we welcome the following contributions.
* bug fixes or improvements to existing functions
* new or improved configurations for training
* training code for new datasets

# Ground Rules

Responsibilities
* Create issues for any major changes and enhancements that you wish to make. Discuss things transparently and get community feedback.
* Keep feature versions as small as possible, preferably one new feature per version.
* Be welcoming to newcomers and encourage diverse new contributors from all backgrounds. See our [Code of Conduct](CODE_OF_CONDUCT.md).

# Getting started

Here is a rough overview of the process to contribute.

1. Create your own fork of the code
2. Do the changes in your fork, preferrably in a new branch, e.g. named with the template `<username>/pr-<keyword>`
3. Please make sure the proposed change runs by training a model or creating a short test script
4. Please do the following before creating a new pull request
    * Python code is formated with [black](https://github.com/psf/black) and import sorted with [isort](https://pycqa.github.io/isort/) (with `--profile=black` option)
    * YAML config files use two spaces indent
    * Send a pull request specifying
        - what the change/addition is
        - why it is useful
        - any other relevant information
4. If you like the change and think the project could use it go ahead and create the pull request

The linting of the code can be done as follows.
```bash
python -m pip install black isort
python -m black .
python -m isort --profile=black .
```

# How to report a bug

Just open an issue, unless it is a security issue.
For a security issue, please drop an email to [dl\_oss\_dev@linecorp.com](dl_oss_dev@linecorp.com) instead.

When you file an issue for a bug, please include the following information.

1. Version of python/torch that you are using.
2. Operating system.
3. If possible, a minimum code example that reproduces the bug. If not possible, at least what command you were running.
4. The error message/problematic behavior encountered.

# How to suggest a feature or enhancement

Before starting to code, it is strongly suggested to open an issue first to
make sure the feature can be smoothly integrated.
