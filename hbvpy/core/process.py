#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hbvpy_dev.process
=================

**A package to process HBV-light simulation results.**

This package is intended to provide functions and methods to parse and process
the output of the different types HBV-light simulations (i.e. SingleRun,
BatchRun, GAPRun).

.. author:: Marc Girons Lopez

"""

import os
import pandas as pd
import datetime as dt


__all__ = ['HBVresults', 'get_gap_results', 'load_metadata']


def load_metadata(catchments_dir):
    """
    Load the metadata of all catchments in a given directory.

    Parameters
    ----------
    catchments_dir : str
        Path of the directory where the catchment folders are stored.

    Returns
    -------
    metadata : Pandas.DataFrame
        Data structure containing the metadata for all catchments in the
        given directory.

    """
    metadata = pd.DataFrame()

    for root, dirs, files in os.walk(catchments_dir):
        for file in files:
            filename = os.path.join(root, file)
            if file == 'metadata.txt':
                meta = pd.read_csv(
                        filename, sep='\t', engine='python', index_col=0)
                metadata = pd.concat([metadata, meta], axis=0)

    return metadata


class HBVresults(object):
    """
    Process results from HBV-light simulations.

    Attributes
    ----------
    bsn_dir : str
        Basin directory.

    """
    def __init__(self, bsn_dir):

        self.bsn_dir = bsn_dir

        self.basin_name = os.path.relpath(bsn_dir, bsn_dir + '..')

    def load_ptq(self, filename='PTQ.txt', no_data=-9999):
        """
        Load the HBV-light PTQ.txt file.

        Parameters
        ----------
        filename : str, optional.
            File name of the ptq file, default: 'PTQ.txt'.
        no_data : int or float, optional.
            Value of NoData values.

        Returns
        -------
        Pandas DataFrame
            Data structure containing the PTQ data.

        """
        filename = self.bsn_dir + '\\Data\\' + filename

        return pd.read_csv(filename, sep='\t', na_values=no_data,
                           index_col=0, parse_dates=True, skiprows=2,
                           header=None, names=['P', 'T', 'Q'])

    def get_ptq_units(self, filename='PTQ.txt'):
        """
        Get the units of the PTQ.txt file to use for labelling figure axes.

        Parameters
        ----------
        filename : str, optional.
            File name of the ptq file, default: 'PTQ.txt'.

        Returns
        -------
        p_units : str
            Precipitation data units.
        t_units : str
            Temperature data units.
        q_units : str
            Runoss data units.

        """
        # Load the ptq.txt file
        ptq = self.load_ptq(filename=filename)
        # Get the temporal resolution
        delta = ptq.index.resolution

        p_units = 'Precipitation [mm ' + delta + '^{-1}]'
        t_units = 'Temperature [$^{\circ}$C]'
        q_units = 'Runoff [mm ' + delta + '^{-1}]'

        return p_units, t_units, q_units

    @staticmethod
    def slice_data(data, start=None, end=None):
        """
        Get a slice of a Pandas.DataFrame.

        Parameters
        ----------
        data : Pandas.DataFrame
            Pandas DataFrame with the relevant data for plotting.
        start : '%Y%m%d', optional
            Start date for the plot, default is None.
        end : '%Y%m%d', optional
            End date for the plot, default is None.

        Return
        ------
        Pandas.DataFrame
            Slice of input Pandas.DataFrame.

        """
        if start is not None and end is not None:
            start = dt.datetime.strptime(start, '%Y%m%d')
            end = dt.datetime.strptime(end, '%Y%m%d')
            return data.loc[start:end]

        if start is not None:
            start = dt.datetime.strptime(start, '%Y%m%d')
            return data.loc[start:]

        elif end is not None:
            end = dt.datetime.strptime(end, '%Y%m%d')
            return data.loc[:end]

        else:
            return data

    def load_gap_results(self, gap_folder):
        """
        Load the results of a HBV-light GAP calibration run.

        Parameters
        ----------
        gap_folder : str
            Name of the GAP results folder.

        Returns
        -------
        Pandas DataFrame
            Data structure containing the GAP results.

        """
        path = self.bsn_dir + '\\' + gap_folder + '\\'
        filename = path + 'GA_best1.txt'

        return pd.read_csv(filename, sep='\t')

    def load_batch_results(self, batch_folder):
        """
        Load the results of a HBV-light Batch Run.

        Parameters
        ----------
        batch_folder : str
            Name of the Batch Run results folder.

        Returns
        -------
        Pandas DataFrame
            Data structure containing the Batch Run results.

        """
        path = self.bsn_dir + '\\' + batch_folder + '\\'
        filename = path + 'BatchRun.txt'

        return pd.read_csv(filename, sep='\t')

    def load_batch_runoff(self, batch_folder):
        """
        Load the runoff time series from a HBV-light Batch Run.

        Parameters
        ----------
        batch_folder : str
            Name of the Batch Run results folder.

        Returns
        -------
        Pandas DataFrame
            Data structure containing the Batch Run runoff time series.

        """
        path = self.bsn_dir + '\\' + batch_folder + '\\'
        filename = path + 'BatchQsimSummary.txt'

        return pd.read_csv(filename, sep='\t', parse_dates=True, index_col=0)

    def load_batch_runoff_comp(self, batch_folder, component='Snow'):
        """
        Load a given runoff component from a HBV-light Batch Run.

        Parameters
        ----------
        batch_folder : str
            Name of the Batch Run results folder.
        component : {'Rain', 'Snow', 'Glacier', 'Q0', 'Q1', 'Q2'}
            Name of the runoff component to load, default 'Snow'.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the Batch Run runoff component
            time series.

        """
        path = self.bsn_dir + '\\' + batch_folder + '\\'
        filename = path + 'BatchQsim_' + component + '.txt.'

        # Parse the index.
        index = pd.read_csv(filename, sep='\t', header=None, nrows=1,
                            index_col=False, parse_dates=True,
                            squeeze=True).transpose()

        # Parse the data.
        data = pd.read_csv(filename, sep='\t', header=None, index_col=False,
                           skiprows=1).transpose()

        # Rename the index and convert it to datetime format.
        index.columns = ['Date']
        index = pd.to_datetime(index['Date'], format='%Y%m%d')

        # Merget the index and data into a single Pandas.DataFrame structure.
        df = pd.concat([data, index], axis=1)

        # Set the index.
        return df.set_index('Date')

    def get_snow_fraction(self, batch_folder):
        """
        Get the median snow fraction.

        Parameters
        ----------
        batch_folder : str
            Name of the Batch Run results folder.

        Returns
        -------
        float
            Snow fraction.

        """
        runoff = self.load_batch_runoff(batch_folder)
        snow = self.load_batch_runoff_comp(batch_folder, component='Snow')

        qmedian = self.mean_yearly_runoff(runoff['Qmedian']).sum().median()
        # q = runoff['Qmedian'].groupby(runoff.index.year).sum().median()
        smedian = snow.quantile(q=0.50, axis=1)
        smedian = snow.groupby(smedian.index.year).sum().median()

        return smedian / qmedian

    def get_batch_runoff_quantile(self, batch_folder, quantile=0.5):
        """
        Get the time series of runoff magnitudes corresponding to a given
        quantile.

        Parameters
        ----------
        batch_folder : str
            Name of the Batch Run results folder.
        quantile : float, optional
            Quantile to get the runoff for, default 0.5:

        Returns
        -------
        Pandas.Series
            Time series of runoff magnitudes corresponding to the given
            quantile.

        """
        path = self.bsn_dir + '\\' + batch_folder + '\\'
        filename = path + 'BatchQsim_(InColumns).txt'

        data = pd.read_csv(filename, sep='\t', parse_dates=True, index_col=0)
        sim = data.drop('Qobs', axis=1)

        return sim.quantile(quantile, axis=1)

    def get_batch_mean_runoff_quantile(self, batch_folder, quantile=0.5):
        """
        Get the median runoff magnitude for a given quantile.

        Parameters
        ----------
        batch_folder : str
            Name of the Batch Run results folder.
        quantile : float, optional
            Quantile to get the runoff for, default 0.5.

        Returns
        -------
        float
            Observed runoff magnitude corresponding to the given quantile.
        float
            Simulated reunoff magnitude corresponding to the given quantile.

        """
        path = self.bsn_dir + '\\' + batch_folder + '\\'
        filename = path + 'BatchQsim_(InColumns).txt'

        sim = pd.read_csv(filename, sep='\t', parse_dates=True, index_col=0)
        obs = sim['Qobs']
        sim = sim.drop('Qobs', axis=1)
        sim = sim.median(axis=1)

        return obs.quantile(quantile, axis=0), sim.quantile(quantile, axis=0)

    def mean_daily_runoff(self, runoff_data):
        """
        Get the average daily runoff.

        Parameters
        ----------
        runoff_data : Pandas.DataFrame or Pandas.Series
            Time series of runoff data.

        Returns
        -------
        Pandas.DataFrame or Pandas.Series
            Average daily runoff.

        """
        runoff_year = self.mean_yearly_runoff(runoff_data)
        runoff_day = runoff_year.cumcount(1) + 1

        return runoff_data.groupby(runoff_day)

    @staticmethod
    def mean_monthly_runoff(runoff_data):
        """
        Get the average monthly runoff.

        Parameters
        ----------
        runoff_data : Pandas.DataFrame or Pandas.Series
            Time series of runoff data.

        Returns
        -------
        Pandas.DataFrame or Pandas.Series
            Average monthly runoff.

        """
        return runoff_data.groupby(runoff_data.index.month)

    @staticmethod
    def mean_yearly_runoff(runoff_data):
        """
        Get the average yearly runoff.

        Parameters
        ----------
        runoff_data : Pandas.DataFrame or Pandas.Series
            Time series of runoff data.

        Returns
        -------
        Pandas.DataFrame or Pandas.Series
            Average yearly runoff.

        """
        return runoff_data.groupby(runoff_data.index.year)


def get_gap_results(
        catchments_dir, catchments, dev_versions, period, obj_fun='SWE_Reff',
        results_type='absolute', reference='Original'):
    """
    Get the median gap results from a number of catchments and development
    versions.

    Parameters
    ----------
    catchments_dir : str
        Path to the directory where the catchment folders are stored.
    catchments : list
        List of folders (catchment names) to get the results for.
    dev_versions : list
        List of development versions names to get the results for.
    period : int
        Simulation period to get the results for.
    obj_fun : str, optional
        Objective function to get the results for, default is 'SWE_Reff'.
    results_type : {'absolute', 'relative', 'rank'}, optional
        Type of results to return, default is 'absolute'.
    reference : str, optional
        Development version to use as reference for the relative results type,
        default is 'Original'.

    Returns
    -------
    Pandas.DataFrame
        Data structure containing the median results for each catchment and
        development version.

    Raises
    ------
    ValueError
        If the provided type of results is not recognised.

    """
    if results_type == 'relative' and reference is None:
        raise ValueError('A reference development version needs to be '
                         'specified to get relative results.')

    board = pd.DataFrame(index=catchments, columns=dev_versions)

    for catchment in catchments:
        catchment_dir = catchments_dir + str(catchment) + '\\'

        for dev_version in dev_versions:
            if results_type == 'relative' and dev_version == reference:
                continue

            gap_dir = dev_version + '_gap_' + str(period)
            results = HBVresults(catchment_dir).load_gap_results(gap_dir)

            if results_type in ['absolute', 'rank']:
                board.loc[catchment, dev_version] = results[obj_fun].median()

            elif results_type == 'relative':
                ref_dir = reference + '_gap_' + str(period)
                ref_res = HBVresults(catchment_dir).load_gap_results(ref_dir)

                delta = results[obj_fun].median() - ref_res[obj_fun].median()
                board.loc[catchment, dev_version] = delta

            else:
                raise ValueError('Selected results type not valid.')

        if results_type == 'rank':
            data = pd.Series(board.loc[catchment])

            rank = 1
            for value in data.sort_values(ascending=False):
                data[data == value] = rank
                rank += 1

            board.loc[catchment] = data

    return board.apply(pd.to_numeric)
