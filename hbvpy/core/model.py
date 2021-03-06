#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hbvpy.model
===========

**A package to run the command line version of HBV-light.**

This package is intended to provide bindngs to the command line version of
HBV-light so the model can be run from a python script.

.. author:: Marc Girons Lopez

"""

import os
import subprocess

from hbvpy.ThirdParty import AnimatedProgressBar


__all__ = ['HBVcatchment', 'HBVsimulation']


class HBV(object):
    """
    Set the command line version of HBV-light (HBV-light-CLI.exe).

    Attributes
    ----------
    hbv_path : str, optional
        Non-default HBV-light-CLI.exe path, default is None.

    Raises
    ------
    ValueError
        If the specified path to the HBV-light-CLI.exe file does not exist.

    """
    def __init__(self, hbv_path=None):

        if hbv_path is None:
            self.hbv_path = (
                'C:\\Program Files (x86)\\HBV-light\\HBV-light-CLI.exe')

        else:
            self.hbv_path = hbv_path

        if not os.path.exists(self.hbv_path):
            raise ValueError(
                'The specified HBV-ligh-CLI.exe file does not exist.')


class HBVsimulation(HBV):
    """
    HBV-light simulation setup.

    This class defines the input data and configuration files for setting up
    a particular simulation setup.

    HBV-light will search for the default files in the data directory and use
    them if they are present. If the user decides not to use a specific file
    (that is located in the data directory with a default name), the str
    'dummy.txt' should be passed to the corresponding attribute.

    Attributes
    ----------
    hbv_path : str, optional
        Non-default HBV-light-CLI.exe path, default is None.
    c : str, optional
        File with catchment settings, default is 'Clarea.xml'.
    p : str, optional
        File with parameter settings, default is 'Parameter.xml'.
    s : str, optional
        File with simulation settings, default is 'Simulation.xml'.
    ptq : str, optional
        File with daily precipitation, temperature and discharge values,
        default is 'ptq.txt'.
    evap : str, optional
        File with potential evaporation values, default is 'EVAP.txt'.
    tmean : str, optional
        File with long-term mean temperature values, default is 'T_mean.txt'.
    ptcalt : str, optional
        File with daily temperature and/or precipitation gradients,
        default is 'PTCALT'txt'.
    sc : str, optional
        File describing the spatial relation between different subcatchments,
        default is 'SubCatchment.txt'.
    b : str, optional
        File with parameter sets for batch simulation, default is 'Batch.txt'.
    ps : str, optional
        File with precipitation series, default is 'P_series.txt'.
    ts : str, optional
        File with temperature series, default is 'T_series.txt'.
    es : str, optional
        File with evaporation series, default is 'EVAP_series.txt'.
    bs : str, optional
        File with batch simulation settings, default is 'Batch_Simulation.txt'.
    mcs : str, optional
        File with Monte Carlo simulation settings,
        default is 'MC_Simulation.txt'.
    gaps : str, optional
        File with GAP simulation settings, default is 'GAP_Simulation.txt'.
    results : str, optional
        Results output folder, default is 'Results'.
    summary : str, optional
        Summary output file, default is 'Summary.txt'.
    g : str, optional
        Glacier profile file, default is 'GlacierProfile.txt'.
    swe : str, optional
        File with snow water equivalent data, default is 'ObsSWE.txt'.
    snowcover . str, optional
        File with snow cover data, default is 'SnowCover.txt'.
    python : str, optional
        Python objective function file, default is 'ObjFunc.py'.

    """

    def __init__(
            self, hbv_path=None, c='Clarea.xml', p='Parameter.xml',
            s='Simulation.xml', ptq='ptq.txt', evap='EVAP.txt',
            tmean='T_mean.txt', ptcalt='PTCALT.txt', sc='SubCatchment.txt',
            b='Batch.txt', ps='P_series.txt', ts='T_series.txt',
            es='EVAP_series.txt', bs='Batch_Simulation.txt',
            mcs='MC_Simulation.txt', gaps='GAP_Simulation.txt',
            results='Results', summary='Summary.txt', g='GlacierProfile.txt',
            swe='ObsSWE.txt', snowcover='SnowCover.txt', python='ObjFunc.py'):

        super().__init__(hbv_path)

        self.results_folder = results

        self.files = {
                'c': c, 'p': p, 's': s, 'ptq': ptq, 'evap': evap,
                'tmean': tmean, 'ptcalt': ptcalt, 'sc': sc, 'b': b, 'ps': ps,
                'ts': ts, 'es': es, 'bs': bs, 'mcs': mcs, 'gaps': gaps,
                'summary': summary, 'g': g, 'swe': swe, 'snowcover': snowcover,
                'python': python}


class HBVcatchment(HBVsimulation):
    """
    HBV-light catchment.

    This class defines the catchment folder for HBV-light and provides
    methods to run the model and show the progress.

    Attributes
    ----------
    bsn_dir : str
        Path to the basin folder (containing a 'Data' sub-folder).
    simulation : hbvpy.model.Scenario instance
        Predefined HBV-light simulation setup to run for the chosen catchment.

    """
    def __init__(self, bsn_dir, simulation):
        """
        """
        self.__simulation = simulation

        self.bsn_dir = bsn_dir

        self.basin_name = os.path.relpath(bsn_dir, bsn_dir + '..')

        if not os.path.exists(self.bsn_dir + self.results_folder):
            os.makedirs(self.bsn_dir + self.results_folder)

    def __getattr__(self, attr):
        """
        """
        return getattr(self.__simulation, attr)

    def __setattr__(self, attr, val):
        """
        """
        if attr == '_HBVcatchment__simulation':
            object.__setattr__(self, attr, val)

        return setattr(self.__simulation, attr, val)

    def _parse_files(self, command):
        """
        Parse the necessary files to run HBV-light.

        Parameters
        ----------
        command : list
            List of arguments needed to run HBV-light.

        Returns
        -------
        command : list
            List of arguments and files needed to run HBV-light.

        """
        for name, file in self.files.items():
            if file is None:
                continue

            else:
                command.append('/' + name + ':' + file)

        return command

    def _print_progress(self, sim_type, process, debug_mode=False):
        """
        Print the run progress of HBV-light.

        Parameters
        ----------
        sim_type : str
            Simulation type.
        process : Subprocess.process
            Process to run the HBV model.
        debug_mode : bool, optional
            Choose whether to show the full HBV-light messages on the
            command line, default is False.

        """
        print('\nProcessing: ' + str(sim_type) +
              ' | Catchment: ' + str(self.basin_name))

        if debug_mode is True:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                print(line)

        else:
            p = AnimatedProgressBar(end=100, width=50)
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                p + 1
                p.show_progress()
                print

    def run(self, sim_type, debug_mode=False):
        """
        Run HBV-light.

        NOTE: Each simulation type (sim_type) requires specific configuration
        files. Please refer to the documentation of HBV-light for information
        on the different simulation types and required files.

        Parameters
        ----------
        sim_type : {'SingleRun', 'MonteCarloRun', 'BatchRun', 'GAPRun'}
            Simulation type.
        debug_mode : bool, optional
            If False a progress bar is shown, otherwise the standard
            HBV-light output is shown, default is False.

        """
        command = [
                self.hbv_path, 'Run', self.bsn_dir,
                sim_type, self.results_folder]
        command = self._parse_files(command)

        process = subprocess.Popen(
                command, stdout=subprocess.PIPE, universal_newlines=True)

        self._print_progress(sim_type, process, debug_mode=debug_mode)
