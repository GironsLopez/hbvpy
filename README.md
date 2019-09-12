# HBVpy
Python functions to interact with the command line version of HBV-light.

This package provides:

- Bindings to run HBV-light from python scripts.
- Functions to pre-process the necessary input data for HBV-light for Swiss catchments.
- An easy way to generate HBV-light configuration files.
- Functions to load and process results files from HBV-light.

---

## HBV-light

HBV-light is a version of the HBV semi-distributed rainfall-runoff model developed and maintained by the Hydrology and Climate group of the University of Zurich. You can find more details on the project's [website](http://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model.html).

---

## How to...

### ... Install the package

Download the master branch of the [package](https://github.com/GironsLopez/hbvpy/archive/master.zip) to a suitable location of your hard drive an unzip the file. Open a command prompt (if you are using Anaconda Python you should use the Anaconda prompt instead), navigate to the package main folder and run the following command:

```
pip install -e .
```

The package should be installed as well as it's dependencies. Currently, HBVpy depends on the following packages:

```
'lxml'
'netCDF4'
'numpy'
'gdal',
'pandas'
'pyproj'
'scipy'
```

It appears that there is a bug when installing `pyproj` using this method so I would recommend to ensure that the packages are already installed in your Python distribution when attempting to install HBVpy. If you are using Anaconda Python you can run the following command:

```
conda install lxml netCDF4 numpy gdal pandas pyproj scipy
```

### ... Run HBV-light with default settings

If you want to run a simple HBV-light simulation for a single catchment, with a default folder structure, and all the necessary configuration and data files in place (see the help function of HBV-light), you can use the following code:

```python
from hbvpy import HBVsimulation, HBVcatchment

# Create an instance of the HBVsimulation class with the default arguments.
# HBVsimulation provides the location of the HBV-light executable, as well as
# the names of the input data and configuration files to use in the simulation
# and the output folder.
simulation = HBVsimulation()

# Set the path of the catchment directory
catchment_dir = 'path\\to\\catchment\\directory'

# Create an instance of the HBVcatchment class by providing the path to the
# catchment folder and the simulation instance. HBVcatchment locates all the
# files indicated by HBVsimulation for the given catchment and allows to run
# HBV-light.
catchment = HBVcatchment(catchment_dir, simulation)

# Perform a single run of the model.
catchment.run('SingleRun')
```

If you want to inspect the screen-dump messages from HBV-light when running the model (e.g. to track possible errors), you can use the flag `debug_mode=True` in the `catchment.run()` method.

Besides performing a single model run (i.e. `SingleRun`) it is also possible to perfom Monte Carlo, Batch and GAP runs by using `MonteCarloRun`, `BatchRun`, and `GAPRun` respectively.

### ... Performing parallel runs

If you need to run HBV-light for e.g. multiple catchments using the same simulation setup you can use the following code:

```python
import multiprocessing as mp
from hbvpy import HBVsimulation, HBVcatchment


def main(catchment_dir):
  """Function containing all the necessary code to run the model.
  """
  # Create an HBVsimulation instance
  simulation = HBVsimulation()

  # Create an HBVcatchment instance providing the path to the catchment folder
  # and the simulation instance.
  catchment = HBVcatchment(catchment_dir, simulation)

  # Run the model
  catchment.run('SingleRun')


# Create a list of all the catchment directories (just as a simple example;
# it can be more elegant than this!)
catchment_dir_list = [catchment_dir_1, catchment_dir_2, catchment_dir_3]

# Define the number of cores (threads) to use for the model simulations
cores_n = 3

# Set up the parallel simulations
p = mp.Pool(processes=cores_n)
p.starmap(main, iterable=catchment_dir_list)
```

For more complex cases such as when calibrating multiple catchments using different parameter values (but also different objective functions, input data...), we can use the `itertools` package. This will allow us to easily produce all combinations of e.g. catchments and parameter values to feed into the `iterable` option. Building on the previous example:

```python
import itertools
import multiprocessing as mp
from hbvpy import HBVsimulation, HBVcatchment


def main(catchment_dir, parameter_file):
  """Function containing all the necessary code to run the model.
  """
  simulation = HBVsimulation(p=parameter_file)

  catchment = HBVcatchment(catchment_dir, simulation)

  catchment.run('SingleRun')


catchment_dir_list = [catchment_dir_1, catchment_dir_2, catchment_dir_3]

parameter_file_list = [parameter_file_1, parameter_file_2]

combinations = list(itertools.product(catchment_dir_list, parameter_file_list))

cores_n = 3

p = mp.Pool(processes=cores_n)
p.starmap(main, iterable=combinations)
```

### ... Performing other tasks

HBVpy also allows to:
* Generate HBV-light input data files for Switzerland from a range of data products by **MeteoSwiss**, **FOEN**, **swisstopo**, **SLF**, and **MODIS**. Check out the classes and functions in the `hbvpy.data` module for more details on the supported products and further documentation.  

* Create and/or modify HBV-light configuration files from a Python environment. Check out the `HBVconfig` class and it's associated methods for further documentation on specific configuration files.

* Parse HBV-light output files into Python data structures. Check out the classes in the `hbvpy.process` module for further documentation.
