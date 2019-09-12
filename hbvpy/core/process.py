#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hbvpy.process
=============

**A package to process HBV-light simulation results.**

This package is intended to provide functions and methods to parse and process
the output of the different types HBV-light simulations (i.e. SingleRun,
BatchRun, GAPRun).

.. author:: Marc Girons Lopez

"""

import os
import pandas as pd

from . import HBVdata


__all__ = ['BatchRun', 'GAPRun', 'MonteCarloRun', 'SingleRun']


class SingleRun(object):
    """
    Process results from HBV-light single run simulations.

    Attributes
    ----------
    bsn_dir : str
        Basin directory.

    """
    def __init__(self, bsn_dir):

        self.bsn_dir = bsn_dir

    def load_results(self, results_folder='Results', sc=None):
        """
        Load the results from a single HBV-light run.

        Parameters
        ----------
        results_folder : str, optional
            Name of the results folder, default is 'Results'.
        sc : int, optional
            Sub-catchment number, in case there are more than one
            sub-catchments, default is None.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the model results.

        Raises
        ------
        ValueError
            If the specified file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the results filename.
        if sc is not None:
            filepath = path + 'Results_SubCatchment_' + str(sc) + '.txt'
        else:
            filepath = path + 'Results.txt'

        # Check if the results file exists
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Load the results file.
        return pd.read_csv(
                filepath, sep='\t', index_col=0,
                parse_dates=True, infer_datetime_format=True)

    def load_dist_results(self, results_folder='Results', sc=None):
        """
        Load the distributed results from a single HBV-light run.

        Parameters
        ----------
        results_folder : str, optional
            Name of the results folder, default is 'Results'.
        sc : int, optional
            Sub-catchment number, in case there are more than one
            sub-catchments, default is None.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the distributed model results.

        Raises
        ------
        ValueError
            If the specified file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the results filename.
        if sc is not None:
            filepath = path + 'Dis_SubCatchment_' + str(sc) + '.txt'
        else:
            filepath = path + 'Dis.txt'

        # Check if the results file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Load the results file.
        return pd.read_csv(
                filepath, sep='\t', index_col=0,
                parse_dates=True, infer_datetime_format=True)

    def load_summary(self, results_folder='Results'):
        """
        Load the summary of the results from a single HBV-light run.

        Parameters
        ----------
        results_folder : str, optional
            Name of the results folder, default is 'Results'.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the distributed model results.

        Raises
        ------
        ValueError
            If the specified file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the summary filename.
        filepath = path + 'Summary.txt'

        # Check if the summary file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Load the summary file.
        return pd.read_csv(filepath, sep='\t', index_col=0)

    def load_peaks(self, results_folder='Results'):
        """
        Load the list of peak flows from a single HBV-light run.

        Following the documentation of HBV-light, a peak is defined as a data
        point with a Qobs value that is at least three times the average Qobs.
        Only a single peak is allowed in a window of 15 days.

        Parameters
        ----------
        results_folder : str, optional
            Name of the results folder, default is 'Results'.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the peak flow dates and values.

        Raises
        ------
        ValueError
            If the specified file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the peaks filename.
        filepath = path + 'Peaks.txt'

        # Check if the peaks file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Load the peaks file.
        return pd.read_csv(
                filepath, sep='\t', index_col=0, parse_dates=True,
                infer_datetime_format=True, squeeze=True)

    def load_q_peaks(self, results_folder='Results'):
        """
        Load the list of observed runoff and peak flows from a single
        HBV-light run.

        Following the documentation of HBV-light, a peak is defined as a data
        point with a Qobs value that is at least three times the average Qobs.
        Only a single peak is allowed in a window of 15 days.

        Parameters
        ----------
        results_folder : str, optional
            Name of the results folder, default is 'Results'.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the observed discharge values as well
            as the peak flow values.

        Raises
        ------
        ValueError
            If the specified file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the runoff peaks filename.
        filepath = path + 'Q_Peaks.txt'

        # Check if the runoff peaks file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Load the runoff peaks file.
        return pd.read_csv(
                filepath, sep='\t', index_col=0, parse_dates=True,
                infer_datetime_format=True, squeeze=True)


class GAPRun(object):
    """
    Process results from HBV-light GAP run simulations.

    Attributes
    ----------
    bsn_dir : str
        Basin directory.

    """
    def __init__(self, bsn_dir):

        self.bsn_dir = bsn_dir

    def load_results(self, results_folder='Results'):
        """
        Load the results from an HBV-light GAP calibration run.

        Parameters
        ----------
        results_folder : str, optional
            Name of the GAP results folder, default is 'Results'.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the GAP results.

        Raises
        ------
        ValueError
            If the specified file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the results filename.
        filepath = path + 'GA_best1.txt'

        # Check if the results file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Load the results file.
        return pd.read_csv(filepath, sep='\t')


class BatchRun(object):
    """
    Process results from HBV-light batch run simulations.

    Attributes
    ----------
    bsn_dir : str
        Basin directory.

    """
    def __init__(self, bsn_dir):

        self.bsn_dir = bsn_dir

    def load_results(self, results_folder='Results', sc=None):
        """
        Load the results from a batch HBV-light run.

        Parameters
        ----------
        results_folder : str, optional
            Name of the Batch Run results folder,
            default is 'Results'.
        sc : int, optional
            Sub-catchment number, in case there are more than one
            sub-catchments, default is None.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the Batch Run results.

        Raises
        ------
        ValueError
            If the specified file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the results filename.
        if sc is not None:
            filepath = path + 'BatchRun_SubCatchment_' + str(sc) + '.txt'
        else:
            filepath = path + 'BatchRun.txt'

        # Check if the results file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Load the results file.
        return pd.read_csv(filepath, sep='\t')

    def load_runoff(self, results_folder='Results', data='columns', sc=None):
        """
        Load the time series of observed and simulated runoff from
        a batch HBV-light Run.

        Parameters
        ----------
        results_folder : str, optional
            Name of the Batch Run results folder, default is 'Results'.
        data : {'rows', 'columns'}, optional
            Organisation of the data in the results file.
        sc : int, optional
            Sub-catchment number, in case there are more than one
            sub-catchments, default is None.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the Batch Run runoff time series.

        Raises
        ------
        ValueError
            If the corresponding file does not exist.
        ValueError
            If the data structure is not recognised.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Load the data according to the predefined format of the results file.
        if data == 'columns':
            # Set the runoff results filename.
            if sc is not None:
                filepath = path + 'BatchQsim_(InColumns)_' + str(sc) + '.txt'
            else:
                filepath = path + 'BatchQsim_(InColumns).txt'

            # Check if the runoff results file exists.
            if not os.path.exists(filepath):
                raise ValueError('The file does not exist.')

            # Load the runoff results file.
            return pd.read_csv(
                    filepath, sep='\t', parse_dates=True,
                    index_col=0, infer_datetime_format=True)

        elif data == 'rows':
            # Set the runoff results filename.
            if sc is not None:
                filepath = path + 'BatchQsim_' + str(sc) + '.txt'
            else:
                filepath = path + 'BatchQsim.txt'

            # Check if the runoff results file exists.
            if not os.path.exists(filepath):
                raise ValueError('The file does not exist.')

            # Parse the index.
            dates = pd.read_csv(
                    filepath, sep='\t', header=None, nrows=1,
                    index_col=False, squeeze=True).transpose()

            # Parse the data.
            data = pd.read_csv(
                    filepath, sep='\t', header=None, index_col=False,
                    skiprows=1).transpose()

            # Rename the index and convert it to datetime format.
            dates.columns = ['Date']
            dates = pd.to_datetime(dates['Date'], format='%Y%m%d')

            # Merge the index and data into a Pandas.DataFrame structure.
            df = pd.concat([data, dates], axis=1)

            # Set the index and return the DataFrame
            return df.set_index('Date')

        else:
            raise ValueError('Data organisation not recognised.')

    def load_runoff_stats(self, results_folder='Results', sc=None):
        """
        Load the time series of observed and simulated runoff statistics
        from a batch HBV-light Run.

        The statistics contain: Qobs, Qmedian, Qmean, Qp10, Qp90.

        Parameters
        ----------
        results_folder : str, optional
            Name of the Batch Run results folder, default is 'Results'.
        sc : int, optional
            Sub-catchment number, in case there are more than one
            sub-catchments, default is None.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the Batch Run runoff statistics
            time series.

        Raises
        ------
        ValueError
            If the specified file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the runoff statistics filename.
        if sc is not None:
            filepath = path + 'BatchQsimSummary_' + str(sc) + '.txt'
        else:
            filepath = path + 'BatchQsimSummary.txt'

        # Check if the runoff statistics file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Load the runoff statistics file.
        return pd.read_csv(
                filepath, sep='\t', parse_dates=True,
                index_col=0, infer_datetime_format=True)

    def load_runoff_component(
            self, results_folder='Results', component='Snow', sc=None):
        """
        Load the time series of a given runoff component from a batch
        HBV-light run.

        Parameters
        ----------
        results_folder : str, optional
            Name of the Batch Run results folder, default is 'Results'.
        component : {'Rain', 'Snow', 'Glacier', 'Q0', 'Q1', 'Q2'}
            Name of the runoff component to load, default 'Snow'.
        sc : int, optional
            Sub-catchment number, in case there are more than one
            sub-catchments, default is None.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the Batch Run runoff component
            time series.

        Raises
        ------
        ValueError
            If the provided runoff component is not recognised.
        ValueError
            If the specified file does not exist.

        """
        # Check if the provided component is valid.
        if component not in ['Rain', 'Snow', 'Glacier', 'Q0', 'Q1', 'Q2']:
            raise ValueError('Provided runoff compoent not recognised.')

        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the runoff component filename.
        if sc is not None:
            filepath = path + 'BatchQsim_' + component + '_' + str(sc) + '.txt'
        else:
            filepath = path + 'BatchQsim_' + component + '.txt.'

        # Check if the runoff component file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Parse the index.
        dates = pd.read_csv(
                filepath, sep='\t', header=None, nrows=1,
                index_col=False, squeeze=True).transpose()

        # Parse the data.
        data = pd.read_csv(
                filepath, sep='\t', header=None, index_col=False,
                skiprows=1).transpose()

        # Rename the index and convert it to datetime format.
        dates.columns = ['Date']
        dates = pd.to_datetime(dates['Date'], format='%Y%m%d')

        # Merge the index and data into a single Pandas.DataFrame structure.
        df = pd.concat([data, dates], axis=1)

        # Set the index.
        return df.set_index('Date')

    def load_monthly_runoff(self, results_folder='Results', sc=None):
        """
        Load the monthly average simulated runoff from each parameter set
        used for a batch HBV-light run.

        Parameters
        ----------
        results_folder : str, optional
            Name of the Batch Run results folder, default is 'Results'.
        sc : int, optional
            Sub-catchment number, in case there are more than one
            sub-catchments, default is None.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the monthly average runoff values
            from each parameter set.

        Raises
        ------
        ValueError
            If the specifed file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the monthly runoff filename.
        if sc is not None:
            filepath = path + 'Qseasonal_' + str(sc) + '.txt'
        else:
            filepath = path + 'Qseasonal.txt.'

        # Check if the monthly runoff file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Load the monthly runoff file.
        return pd.read_csv(filepath, sep='\t')

    def load_swe(self, results_folder='Results'):
        """
        Load the time series of simulated snow water equivalent for each
        elevation band and parameter set used for a batch HBV-light run.

        NOTE: This method ONLY works if no additional precipitaiton,
        temperature, or evaporation series are provided.

        Parameters
        ----------
        results_folder : str, optional
            Name of the Batch Run results folder, default is 'Results'.

        Returns
        -------
        swe : Pandas.DataFrame
            Data structure containing the simulated snow water equivalent
            data for each elevation band and parameter set.

        Raises
        ------
        ValueError
            If the specifed file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the snow water equivalent filename.
        filepath = path + 'Simulated_SWE.txt.'

        # Check if the snow water equivalent file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Parse the raw data
        raw_data = pd.read_csv(filepath, sep='\t')

        # Get the number of parameter sets
        param_sets = raw_data['Parameterset'].unique()

        # Initialise a dict to write the SWE data
        swe = {ps: None for ps in param_sets}

        # Parse the data
        for ps in param_sets:
            data_tmp = raw_data[raw_data['Parameterset'] == ps]
            data_tmp.set_index('SWE serie', drop=True, inplace=True)
            data = data_tmp.drop(
                    ['Parameterset', 'Prec. serie',
                     'Temp. serie', 'Evap. serie'], axis=1).transpose()
            data.index = pd.to_datetime(data.index, format='%Y%m%d')

            swe[ps] = data

        return swe

    def calculate_swe_stats(self, results_folder, clarea=None):
        """
        Calculate the catchment-wide statistics of snow water equivalent
        taking into account all elevation bands and parameter sets used in
        batch HBV-light run.

        This method provides median and mean catchment snow water equivalent,
        in addition to the 10th and 90th percentiles.

        Parameters
        ----------
        results_folder : str
            Name of the Batch Run results folder.
        clarea : str, optional
            Custom name for the 'Clarea.xml' file. If no value is provided
            the default file name is used, default is None.

        Returns
        -------
        sim_swe : Pandas.Series
            Data structure containing the simulated catchment avverage
            snow water equivalent for each time step.

        """
        # Instatiate the HBVdata class
        hbv_data = HBVdata(self.bsn_dir)

        # Use the default elevation distribution file name if no name is
        # specified.
        if clarea is None:
            clarea = 'Clarea.xml'

        # Check if the elevation distribution file exists.
        if not os.path.exists(hbv_data.data_dir + clarea):
            raise ValueError('The Clarea.xml file does not exist.')

        # Load the area fraction of each elevation band in the catchment.
        elev_dist = hbv_data.load_elev_dist(filename=clarea)

        # Load the distributed simulated snow water equivalent.
        dist_swe = self.load_swe(results_folder)

        # Calculate the catchment average snow water equivalent for
        # each of the parameter sets.
        avg_swe = pd.DataFrame()
        for p_set in dist_swe.keys():
            for i, col in enumerate(dist_swe[p_set].columns):
                area = elev_dist.iloc[i]
                dist_swe[p_set][col] = dist_swe[p_set][col] * area
            avg_swe[p_set] = dist_swe[p_set].sum(axis=1)

        # Calculate the snow water equivalent statistics:
        # median, mean, 10p, and 90p
        swe_stats = pd.DataFrame()
        swe_stats['SWEmedian'] = avg_swe.median(axis=1)
        swe_stats['SWEmean'] = avg_swe.mean(axis=1)
        swe_stats['SWEp10'] = avg_swe.quantile(q=0.1, axis=1)
        swe_stats['SWEp90'] = avg_swe.quantile(q=0.9, axis=1)

        return swe_stats

    def calculate_runoff_quantile(
            self, results_folder='Results', data='Columns', quantile=0.5):
        """
        Calculate the time series of runoff magnitudes corresponding to a given
        quantile.

        NOTE: This method only works if the catchment contains a single
        subcatchment.

        Parameters
        ----------
        results_folder : str, optional
            Name of the Batch Run results folder, default is 'Results'.
        data : {'rows', 'columns'}, optional
            Organisation of the data in the results file.
        quantile : float, optional
            Quantile to get the runoff for, default 0.5.

        Returns
        -------
        Pandas.Series
            Time series of runoff magnitudes corresponding to the given
            quantile.

        """
        # Load the runoff results data
        runoff = self.load_runoff(results_folder, data=data, sc=None)

        # Drop the observed runoff column
        runoff = runoff.drop('Qobs', axis=1)

        # Return the time series of the given runoff quantile.
        return runoff.quantile(quantile, axis=1)


class MonteCarloRun(object):
    """
    Process results from HBV-light Monte Carlo simulations.

    Attributes
    ----------
    bsn_dir : str
        Basin directory.

    """
    def __init__(self, bsn_dir):

        self.bsn_dir = bsn_dir

    def load_results(self, results_folder='Results', sc=None):
        """
        Load the results of a HBV-light Monte Carlo Run.

        Parameters
        ----------
        results_folder : str, optional
            Name of the MC Run results folder, default is 'Results'.
        sc : int, optional
            Sub-catchment number, in case there are more than one
            sub-catchments, default is None.

        Returns
        -------
        Pandas.DataFrame
            Data structure containing the MC Run results.

        Raises
        ------
        ValueError
            If the specified file does not exist.

        """
        # Set the results folder path.
        path = self.bsn_dir + '\\' + results_folder + '\\'

        # Set the results filename.
        if sc is not None:
            filepath = path + 'Multi_SubCatchment_' + str(sc) + '.txt'
        else:
            filepath = path + 'Multi.txt'

        # Check if the results file exists.
        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        # Load the results file.
        return pd.read_csv(filepath, sep='\t', index_col=0)
