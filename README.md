This repository contains the python files xphm.py and xphm_mrb.py, which allow one to perform mode-by-mode relative binning as detailed in https://arxiv.org/abs/2109.09872. An example that uses this code in in the jupyter notebook MRB_Example_GW190814.ipynb.

This code requires the following python packages to be installed (if the latest versions do not work, it was tested with the following versions):
numpy 1.19.1
matplotlib 3.3.1
scipy 1.6.2
gwpy 2.0.2
lal, lalsimulation (see below)

python version 3.7.9

The correct lal and lalsimulation packages (from Geraint Pratten's branch) can be installed with
RUN git clone https://github.com/GeraintPratten/lalsuite_gp && LAL_INSTALL_PREFIX="/opt/conda" && cd lalsuite_gp  && ./00boot && ./configure --prefix=${LAL_INSTALL_PREFIX} --enable-swig-python --disable-lalstochastic --disable-lalframe --disable-lalmetaio --disable-lalburst --disable-lalinspiral --disable-lalxml --disable-lalinference --disable-laldetchar --disable-lalapps CFLAGS="-Wno-error" && make -j10 && make install