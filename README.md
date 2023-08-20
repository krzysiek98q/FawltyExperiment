This repository contains a performance check of the FawltyDeps package on known (selected) python packages.

FawltyDeps respository: [GitHub - tweag/FawltyDeps: Python dependency checker](https://github.com/tweag/FawltyDeps)

# About the experiment

For the purpose of conducting the experiment, a class (in the *autoproject.py* file) was created that, based on the package name, is able to retrieve package information from the PyPI API, clone the repository with the source code and run FawltyDeps (with optional settings).

The experiment was conducted on ten selected python libraries. The code of the experiment can be found in the file *experiment_ten_projects.py*. The results of FawltyDeps were saved in the *results_ten_projects* directory.


