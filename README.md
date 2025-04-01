# Causal Testing Framework
### A Causal Inference-Driven Software Testing Framework


[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![example workflow](https://github.com/CITCOM-project/CausalTestingFramework/actions/workflows/ci-tests.yaml/badge.svg)
[![codecov](https://codecov.io/gh/CITCOM-project/CausalTestingFramework/branch/main/graph/badge.svg?token=04ijFVrb4a)](https://codecov.io/gh/CITCOM-project/CausalTestingFramework)
[![Documentation Status](https://readthedocs.org/projects/causal-testing-framework/badge/?version=latest)](https://causal-testing-framework.readthedocs.io/en/latest/?badge=latest)
![Dynamic TOML Badge](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FCITCOM-project%2FCausalTestingFramework%2Fmain%2Fpyproject.toml&query=%24.project%5B'requires-python'%5D&label=python)
![PyPI - Version](https://img.shields.io/pypi/v/causal-testing-framework)
![GitHub Licens[schematic.tex](images%2Fschematic.tex)e](https://img.shields.io/github/license/CITCOM-project/CausalTestingFramework)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07739/status.svg)](https://doi.org/10.21105/joss.07739)
[![DOI](https://img.shields.io/badge/doi-10.26180/5c6e1160b8d8a-blue.svg?style=flat&labelColor=whitesmoke&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAB8AAAAfCAYAAAAfrhY5AAAJsklEQVR42qWXd1DTaRrHf%2BiB2Hdt5zhrAUKz4IKEYu9IGiGFFJJQ0gkJCAKiWFDWBRdFhCQUF3UVdeVcRQEBxUI3yY9iEnQHb3bdW1fPubnyz%2F11M7lvEHfOQee2ZOYzPyDv%2B3yf9%2Fk95YX4fx%2BltfUt08GcFEuPR4U9hDDZ%2FVngIlhb%2FSiI6InkTgLzgDcgfvtnovhH4BzoVlrbwr55QnhCtBW4QHXnFrZbPBaQoBh4%2FSYH2EnpBEtqcDMVzB93wA%2F8AFwa23XFGcc8CkT3mxz%2BfXWtq9T9IQlLIXYEuHojudb%2BCM7Hgdq8ydi%2FAHiBXyY%2BLjwFlAEnS6Jnar%2FvnQVhvdzasad0eKvWZKe8hvDB2ofLZ%2FZEcWsh%2BhyIuyO5Bxs2iZIE4nRv7NWAb0EO8AC%2FWPxjYAWuOEX2MSXZVgPxzmRL3xKz3ScGpx6p6QnOx4mDIFqO0w6Q4fEhO5IzwxlSwyD2FYHzwAW%2BAZ4fEsf74gCumykwNHskLM7taQxLYjjIyy8MUtraGhTWdkfhkFJqtvuVl%2F9l2ZquDfEyrH8B0W06nnpH3JtIyRGpH1iJ6SfxDIHjRXHJmdQjLpfHeN54gnfFx4W9QRnovx%2FN20aXZeTD2J84hn3%2BqoF2Tqr14VqTPUCIcP%2B5%2Fly4qC%2BUL3sYxSvNj1NwsVYPsWdMUfomsdkYm3Tj0nbV0N1wRKwFe1MgKACDIBdMAhPE%2FwicwNWxll8Ag40w%2BFfhibJkGHmutjYeQ8gVlaN%2BjO51nDysa9TwNUFMqaGbKdRJZFfOJSp6mkRKsv0rRIpEVWjAvyFkxNOEpwvcAVPfEe%2Bl8ojeNTx3nXLBcWRrYGxSRjDEk0VlpxYrbe1ZmaQ5xuT0u3r%2B2qe5j0J5uytiZPGsRL2Jm32AldpxPUNJ3jmmsN4x62z1cXrbedXBQf2yvIFCeZrtyicZZG2U2nrrBJzYorI2EXLrvTfCSB43s41PKEvbZDEfQby6L4JTj%2FfIwam%2B4%2BwucBu%2BDgNK05Nle1rSt9HvR%2FKPC4U6LTfvUIaip1mjIa8fPzykii23h2eanT57zQ7fsyYH5QjywwlooAUcAdOh5QumgTHx6aAO7%2FL52eaQNEShrxfhL6albEDmfhGflrsT4tps8gTHNOJbeDeBlt0WJWDHSgxs6cW6lQqyg1FpD5ZVDfhn1HYFF1y4Eiaqa18pQf3zzYMBhcanlBjYfgWNayAf%2FASOgklu8bmgD7hADrk4cRlOL7NSOewEcbqSmaivT33QuFdHXj5sdvjlN5yMDrAECmdgDWG2L8P%2BAKLs9ZLZ7dJda%2BB4Xl84t7QvnKfvpXJv9obz2KgK8dXyqISyV0sXGZ0U47hOA%2FAiigbEMECJxC9aoKp86re5O5prxOlHkcksutSQJzxZRlPZmrOKhsQBF5zEZKybUC0vVjG8PqOnhOq46qyDTDnj5gZBriWCk4DvXrudQnXQmnXblebhAC2cCB6zIbM4PYgGl0elPSgIf3iFEA21aLdHYLHUQuVkpgi02SxFdrG862Y8ymYGMvXDzUmiX8DS5vKZyZlGmsSgQqfLub5RyLNS4zfDiZc9Edzh%2FtCE%2BX8j9k%2FqWB071rcZyMImne1SLkL4GRw4UPHMV3jjwEYpPG5uW5fAEot0aTSJnsGAwHJi2nvF1Y5OIqWziVCQd5NT7t6Q8guOSpgS%2Fa1dSRn8JGGaCD3BPXDyQRG4Bqhu8XrgAp0yy8DMSvvyVXDgJcJTcr1wQ2BvFKf65jqhvmxXUuDpGBlRvV36XvGjQzLi8KAKT2lYOnmxQPGorURSV0NhyTIuIyqOmKTMhQ%2BieEsgOgpc4KBbfDM4B3SIgFljvfHF6cef7qpyLBXAiQcXvg5l3Iunp%2FWv4dH6qFziO%2BL9PbrimQ9RY6MQphEfGUpOmma7KkGzuS8sPUFnCtIYcKCaI9EXo4HlQLgGrBjbiK5EqMj2AKWt9QWcIFMtnVvQVDQV9lXJJqdPVtUQpbh6gCI2Ov1nvZts7yYdsnvRgxiWFOtNJcOMVLn1vgptVi6qrNiFOfEjHCDB3J%2BHDLqUB77YgQGwX%2Fb1eYna3hGKdlqJKIyiE4nSbV8VFgxmxR4b5mVkkeUhMgs5YTi4ja2XZ009xJRHdkfwMi%2BfocaancuO7h%2FMlcLOa0V%2FSw6Dq47CumRQAKhgbOP8t%2BMTjuxjJGhXCY6XpmDDFqWlVYbQ1aDJ5Cptdw4oLbf3Ck%2BdWkVP0LpH7s9XLPXI%2FQX8ws%2Bj2In63IcRvOOo%2BTTjiN%2BlssfRsanW%2B3REVKoavBOAPTXABW4AL7e4NygHdpAKBscmlDh9Jysp4wxbnUNna3L3xBvyE1jyrGIkUHaqQMuxhHElV6oj1picvgL1QEuS5PyZTEaivqh5vUCKJqOuIgPFGESns8kyFk7%2FDxyima3cYxi%2FYOQCj%2F%2B9Ms2Ll%2Bhn4FmKnl7JkGXQGDKDAz9rUGL1TIlBpuJr9Be2JjK6qPzyDg495UxXYF7JY1qKimw9jWjF0iV6DRIqE%2B%2FeWG0J2ofmZTk0mLYVd4GLiFCOoKR0Cg727tWq981InYynvCuKW43aXgEjofVbxIqrm0VL76zlH3gQzWP3R3Bv9oXxclrlO7VVtgBRpSP4hMFWJ8BrUSBCJXC07l40X4jWuvtc42ofNCxtlX2JH6bdeojXgTh5TxOBKEyY5wvBE%2BACh8BtOPNPkApjoxi5h%2B%2FFMQQNpWvZaMH7MKFu5Ax8HoCQdmGkJrtnOiLHwD3uS5y8%2F2xTSDrE%2F4PT1yqtt6vGe8ldMBVMEPd6KwqiYECHDlfbvzphcWP%2BJiZuL5swoWQYlS%2Br7Yu5mNUiGD2retxBi9fl6RDGn4Ti9B1oyYy%2BMP5G87D%2FCpRlvdnuy0PY6RC8BzTA40NXqckQ9TaOUDywkYsudxJzPgyDoAWn%2BB6nEFbaVxxC6UXjJiuDkW9TWq7uRBOJocky9iMfUhGpv%2FdQuVVIuGjYqACbXf8aa%2BPeYNIHZsM7l4s5gAQuUAzRUoT51hnH3EWofXf2vkD5HJJ33vwE%2FaEWp36GHr6GpMaH4AAPuqM5eabH%2FhfG9zcCz4nN6cPinuAw6IHwtvyB%2FdO1toZciBaPh25U0ducR2PI3Zl7mokyLWKkSnEDOg1x5fCsJE9EKhH7HwFNhWMGMS7%2BqxyYsbHHRUDUH4I%2FAheQY7wujJNnFUH4KdCju83riuQeHU9WEqNzjsJFuF%2FdTDAZ%2FK7%2F1WaAU%2BAWymT59pVMT4g2AxcwNa0XEBDdBDpAPvgDIH73R25teeuAF5ime2Ul0OUIiG4GpSAEJeYW9wDTf43wfwHgHLKJoPznkwAAAABJRU5ErkJggg%3D%3D)](http://doi.org/10.15131/shef.data.24427516.v2)

Causal testing is a causal inference-driven framework for functional black-box testing. This framework utilises
graphical causal inference (CI) techniques for the specification and functional testing of software from a black-box
perspective. In this framework, we use causal directed acyclic graphs (DAGs) to express the anticipated cause-effect
relationships amongst the inputs and outputs of the system-under-test and the supporting mathematical framework to
design statistical procedures capable of making causal inferences. Each causal test case focuses on the causal effect of
an intervention made to the system-under test. That is, a prescribed change to the input configuration of the
system-under-test that is expected to cause a change to some output(s).

![Causal Testing Workflow](images/schematic-dark.png#gh-dark-mode-only)
![Causal Testing Workflow](images/schematic.png#gh-light-mode-only)

## Installation

### Requirements
- Python 3.10, 3.11 and 3.12

- Microsoft Visual C++ 14.0+ (Windows only).

To install the latest stable release of the Causal Testing Framework:

``pip install causal-testing-framework``

or if you want to install with the development packages/tools:

``pip install causal-testing-framework[dev]``

Alternatively, you can install directly via source:

```shell
git clone https://github.com/CITCOM-project/CausalTestingFramework
cd CausalTestingFramework
```
then to install a specific release:

```shell
git fetch --all --tags --prune
git checkout tags/<tag> -b <branch>
pip install . # For core API only
pip install -e . # For editable install, useful for development work
```
For more information on how to use the Causal Testing Framework, please refer to our [documentation](https://causal-testing-framework.readthedocs.io/en/latest/?badge=latest).  

>[!NOTE]
>We recommend you use a 64 bit OS (standard in most modern machines) as we have had reports of the installation crashing on some 32 bit Debian installations.

## Usage
>[!NOTE]
> Example usage can be found in the `examples` directory.

1. To run the causal testing framework, you need some runtime data from your system, some causal test cases, and a causal DAG that specifies the expected causal relationships between the variables in your runtime data (and any other relevant variables that are _not_ recorded in the data but are known to be relevant).

2. If you do not already have causal test cases, you can convert your causal DAG to causal tests by running the following command.

```
python causal_testing/testing/metamorphic_relation.py --dag_path $PATH_TO_DAG --output_path $PATH_TO_TESTS
```

3. You can now execute your tests by running the following command.
```
python -m causal_testing --dag_path $PATH_TO_DAG --data_paths $PATH_TO_DATA --test_config $PATH_TO_TESTS --output $OUTPUT
```
The results will be saved for inspection in a JSON file located at `$OUTPUT`.
In the future, we hope to add a visualisation tool to assist with this.

## How to Cite
If you use our framework in your work, please cite the following:

``This research has used version X.Y.Z (software citation) of the
Causal Testing Framework (paper citation).``

The paper citation should be the Causal Testing Framework [paper](https://dl.acm.org/doi/10.1145/3607184),
and the software citation should contain the specific Figshare [DOI](https://orda.shef.ac.uk/articles/software/CITCOM_Software_Release/24427516) of the version used in your work.



<details>
  <summary><b>BibTeX Citations</b></summary>

  <details>
    <summary>Paper</summary>

    ```
    @ARTICLE{Clark_etal_2023,
    author = {Clark, Andrew G. and Foster, Michael and Prifling, Benedikt and Walkinshaw, Neil and Hierons, Robert M.
    and Schmidt, Volker and Turner, Robert D.},
    title = {Testing Causality in Scientific Modelling Software},
    year = {2023},
    publisher = {Association for Computing Machinery},
    url = {https://doi.org/10.1145/3607184},
    doi = {10.1145/3607184},
    journal = {ACM Trans. Softw. Eng. Methodol.},
    month = {jul},
    keywords = {Software Testing, Causal Testing, Causal Inference}
    }
    ```

  </details>

  <details>
    <summary>Software (example)</summary>

    ```
    @ARTICLE{Wild2023,
    author = {Foster, Michael and Clark, Andrew G. and Somers, Richard and Wild, Christopher and Allian, Farhad and Hierons, Robert M. and Wagg, David and Walkinshaw, Neil},
    title = {CITCOM Software Release},
    year = {2023},
    month = {nov},
    url = {https://orda.shef.ac.uk/articles/software/CITCOM_Software_Release/24427516},
    doi = {10.15131/shef.data.24427516.v1}
    }
    ```
  </details>

</details>

## How to Contribute

To contribute to our work, please ensure the following:

1. [Fork the repository](https://help.github.com/articles/fork-a-repo/) into your own GitHub account, and [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) it to your local machine.
2. [Create a new branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository) in your forked repository. Give this branch an appropriate name, and create commits that describe the changes.
3. [Push your changes](https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository) to your new branch in your remote fork, compare with `CausalTestingFramework/main`, and ensure any conflicts are resolved.
4. Create a draft [pull request](https://docs.github.com/en/get-started/quickstart/hello-world#opening-a-pull-request) from your branch, and ensure you have [linked](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls) it to any relevant issues in your description.

We use the [unittest]() module to develop our tests and the [pytest](https://pytest.org/en/latest/) framework as our test discovery, [pylint](https://pypi.org/project/pylint/) for our code analyser, and [black](https://pypi.org/project/black/) for our code formatting.
To find the other (optional) developer dependencies, please check `pyproject.toml`.



## Acknowledgements

The Causal Testing Framework is supported by the UK's Engineering and Physical Sciences Research Council (EPSRC),
with the project name [CITCOM](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/T030526/1) - "_Causal Inference for Testing of Computational Models_"
under the grant EP/T030526/1.
