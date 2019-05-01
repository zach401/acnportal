# ACN Portal

The ACN Portal is a suite research tools developed at Caltech to accelerate the pace of large-scale EV charging research. 

### ACN-Data
The ACN-Data Dataset is a collection of EV charging sessions collected at Caltech and NASA's Jet Propulsion Laboratory (JPL). This basic Python client simplifies the process of pulling data from his dataset via its public API.

### ACN-Sim
ACN-Sim is a simulation environment for large-scale EV charging algorithms. It interfaces with ACN-Data to provide access to realistic test cases based on actual user behavior. 

### algorithms
algorithms is a package of common EV charging algorithms which can be used for comparison when evaluating new algorithms. 

This package is intended to be populated by the community. If you have a promising EV charging algorithm, please implement it as a subclass of BasicAlgorithm and send a pull request. 

## Installation
Download or clone this repository. Navigate to its root directory. Install using pip. 

```bash
pip install .
```
