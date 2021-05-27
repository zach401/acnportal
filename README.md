[![Build Status](https://travis-ci.org/zach401/acnportal.svg?branch=master)](https://travis-ci.org/zach401/acnportal)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/c357a20f61f941688c157ce21de905b7)](https://www.codacy.com/manual/Caltech_ACN/acnportal?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=zach401/acnportal&amp;utm_campaign=Badge_Grade)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![DOI](https://zenodo.org/badge/134629497.svg)](https://zenodo.org/badge/latestdoi/134629497)

# ACN Portal

The ACN Portal is a suite of research tools developed at Caltech to accelerate the pace of large-scale EV charging research.
Checkout the documentation at <https://acnportal.readthedocs.io/en/latest/>.

For more information about the ACN Portal and EV reasearch at Caltech check out <https://ev.caltech.edu>.

## ACN-Data

The ACN-Data Dataset is a collection of EV charging sessions collected at Caltech and NASA's Jet Propulsion Laboratory (JPL). This basic Python client simplifies the process of pulling data from his dataset via its public API.

## ACN-Sim

ACN-Sim is a simulation environment for large-scale EV charging algorithms. It interfaces with ACN-Data to provide access to realistic test cases based on actual user behavior.

## algorithms

algorithms is a package of common EV charging algorithms which can be used for comparison when evaluating new algorithms.

This package is intended to be populated by the community. If you have a promising EV charging algorithm, please implement it as a subclass of BasicAlgorithm and send a pull request.

## Installation

Download or clone this repository. Navigate to its root directory. Install using pip.

```bash
pip install .
```

## Tutorials

See the `tutorials` directory for jupyter notebooks that you can
run to learn some of the functionality of `acnportal`. These
tutorials are also included on the readthedocs page. Additional
demos and case studies can be found at
<https://github.com/caltech-netlab/acnportal-experiments>
We also have a video series of `acnportal` demos, which can
be found at TODO.

## Running Tests

Tests may be run after installation by executing

```bash
python -m unittest discover -v
```

Remove `-v` after `discover` to suppress verbose output.

## Contributing

If you're submitting a bug report, feature request, question, or
documentation suggestion, please submit the issue through Github and
follow the templates outlined there.

If you are contributing code to the project, please view the 
contributing guidelines [here.](https://github.com/zach401/acnportal/master/CONTRIBUTING.md)

## Questions

Contact the ACN Research Portal team at <mailto:ev-help@caltech.edu> with any
questions, or submit a question through Github issues.
