# Unsupervised classification of the spectrogram zeros

Juan M. Miramont, François Auger, Marcelo A. Colominas, Nils Laurent, Sylvain Meignen

## Visualizing python notebooks

In order to run the python code in this folder, please install the dependencies listed in the ```.toml``` file. You can easily do this using [```poetry```](https://python-poetry.org/docs/), a tool for dependency management and packaging in python.

### Installation using ```poetry```

*Remark for conda users:*

*If you have [`Anaconda`](https://www.anaconda.com/) or [`Miniconda`](https://docs.conda.io/en/latest/miniconda.html) installed please disable the auto-activation of the base environment and your conda environment using:*

```bash
conda config --set auto_activate_base false
conda deactivate
```

First install ```poetry``` following the steps described [here](https://python-poetry.org/docs/#installation). Once you're done with this, open a terminal in the directory where you clone this repository (or use the console in your preferred IDE). Then, make ```poetry``` create a virtual environment and install all the current dependencies of the benchmark using:

```bash
poetry install 
```
