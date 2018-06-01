#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hbvpy.config
============

**A package to generate xml configuration files needed to run HBV-light.**

This package is intended to provide functions and methods to generate
the necessary configuration files to run HBV-light. More specifically, it
allows to generate the following files:

* SnowRoutineSettings.xml
* Simulation.xml
* Batch_Simulation.xml
* Parameter.xml
* GAP_Simulation.xml
* MC_Simulation.xml
* Clarea.xml

.. author:: Marc Girons Lopez

"""

import os
import pandas as pd
from lxml import etree as ET
from datetime import datetime as dt


__all__ = ['HBVconfig']


class HBVconfig(object):
    """
    Generate an HBV-light Catchment folder structure and configuration files.

    Attributes
    ----------
    bsn_dir : str
        Basin directory.
    model_type : {'Standard', 'OnlySnowDistributed', 'DistributedSUZ',
                  'ThreeGWBoxes', ThreeGWBoxesDistributedSTZ',
                  'ThreeGWBoxesDistributedSTZandSUZ', 'ResponseDelayed'
                  'OneGWBox'}, optional
        Model type to use, default is 'Standard'.
    model_variant : {'Basic', 'Aspect', 'Glacier'}, optional
        Model variant to use, default is 'Basic'.
    old_suz : bool, optional
        Use UZL and K0 in SUZ-box and therefore two runoff components from
        the storage in the soil upper zone, default is True.
    pcalt_data : bool, optional
        Use observed precipitation lapse rate input data, default if False.
    tcalt_data : bool, optional
        Use observed temperature lapse rate input data, default is False.
    et_corr : bool, optional
        Use long-term monthly or daily air temperature average values to
        correct the potential evapotranspiration, default is False.

    """
    XSI = 'http://www.w3.org/2001/XMLSchema-instance'
    XSD = 'http://www.w3.org/2001/XMLSchema'

    XSI_TYPE = '{%s}type' % XSI

    def __init__(
            self, bsn_dir, model_type='Standard', model_variant='Basic',
            old_suz=True, pcalt_data=False, tcalt_data=False, et_corr=False):

        # The configuration files are stored in the 'Data' subfolder.
        self.data_dir = bsn_dir + '\\Data\\'

        # Create the folder structure if it doesn't exist.
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Model setup
        self.model_type = model_type
        self.model_variant = model_variant
        self.old_suz = old_suz

        # Additional input data
        self.pcalt_data = pcalt_data
        self.tcalt_data = tcalt_data
        self.et_corr = et_corr

    @staticmethod
    def _add_subelement(root, name, value=None, attribute=None):
        """
        Add an xml sub-element and give it a value and/or attribute.

        The value of the element can either be None or any of the following
        types: str, int, float, bool. In all the cases the value is
        transformed to a str.

        Parameters
        ----------
        root : lxml.etree.Element or lxml.etree.SubElement
            Element or sub-element to add a sub-element to.
        name : str
            Name of the sub-element.
        value : str or int or float or bool, optional
            Value of the sub-element, default is None.
        attribute : dict, optional
            Attributes of the sub-element, default is None.

        Returns
        -------
        element : lxml.etree.SubElement
            xml sub-element with its value and/or attribute.

        Raises
        ------
        ValueError
            If the value type is not supported.

        """
        if attribute is not None:
            element = ET.SubElement(root, name, attrib=attribute)

        else:
            element = ET.SubElement(root, name)

        if value is not None:
            if isinstance(value, bool):
                element.text = str(value).lower()

            elif isinstance(value, int) or isinstance(value, float):
                element.text = str(value)

            elif isinstance(value, str):
                element.text = value

            else:
                raise ValueError('Error: type not supported')

        return element

    @staticmethod
    def _write(root, filename):
        """
        Write an root element tree to an xml file.

        Parameters
        ----------
        root : lxml.ElementTree
            Element tree with the xml information.
        filename : str
            Path and name of the xml file to save the element tree.

        """
        tree = ET.ElementTree(root)
        tree.write(filename, pretty_print=True,
                   xml_declaration=True, encoding='UTF-8')

    @staticmethod
    def _date_format(date):
        """
        Set a given date to the correct HBV-light format.

        Parameters
        ----------
        date : '%Y-%m-%d'
            Date.

        Returns
        -------
        '%Y-%m-%dT%H:%M:%S'
            Date in the correct HBV-light format.

        """
        return dt.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%dT%H:%M:%S')

    def model_settings(
            self, filename='Simulation.xml', start_warm_up='1900-01-01',
            start_simulation='1900-01-01', end_simulation='1900-01-01',
            save_results=True, save_dist=True, compute_interval=False,
            intervals=None, compute_peak=True, compute_season=False,
            compute_weight=False, start_day=1, end_day=31,
            start_month='January', end_month='December', weights=None):
        """
        Define the HBV-light model settings.

        Parameters
        ----------
        filename : str, optional
            Path and file name of the model settings file name,
            default is 'Simulation.xml'.
        start_warm_up : '%Y-%m-%d', optional
            Start of the warming-up period, default is '1900-01-01'.
        start_simulation : '%Y-%m-%d', optional
            Start of the simulation period, default is '1900-01-01'.
        end_simulation : '%Y-%m-%d', optional
            End of the simulation period, default is '1900-01-01'.
        save_results : bool, optional
            Choose whether to save simulation results, default is True.
        save_dist : bool, optional
            Choose whether to save distributed simulation results,
            default is True.
        compute_interval : bool, optional
            Choose whether to compute efficiency based on intervals of n
            time steps, default is False.
        intervals : list of int, optional
            Time step intervals to calculate the efficiency for,
            default is None.
        compute_peak : bool, optional
            Choose whether to compute efficiency for peak flows, default is
            True.
        compute_season : bool, optional
            Choose whether to compute efficiency for the season between
            'start_day''start_month' and 'end_day''end_month',
            default is False.
        compute_weight : bool, optional
            Choose whether to compute efficiency based on 'weights',
            default is False.
        start_day : int, optional
            Start day of the season, default is 1.
        end_day : int, optional
            End day of the season, default is 31.
        start_month : str, optional
            Start month of the season, default is 'January'.
        end_month : str, optional
            End month of the season, default is 'December'.
        weights : dict, optional
            Dictionary of {Q: weight}, default is None.

        Raises
        ------
        ValueError
            If compute_interval is True but no Reff Intervals are specified.
        ValueError
            If compute_weight is True but no weights are specified.

        """
        # Generate the XML configuration file
        # ---------------------------------------------------------------------
        root = ET.Element(
                'Simulation', nsmap={'xsi': self.XSI, 'xsd': self.XSD})

        # Model type and additional input data
        # ---------------------------------------------------------------------
        self._add_subelement(root, 'ModelType', self.model_type)
        self._add_subelement(root, 'ModelVariant', self.model_variant)
        self._add_subelement(root, 'UseOldSUZ', self.old_suz)

        # Simulation settings
        # ---------------------------------------------------------------------
        self._add_subelement(root, 'StartOfWarmingUpPeriod',
                             self._date_format(start_warm_up))
        self._add_subelement(root, 'StartOfSimulationPeriod',
                             self._date_format(start_simulation))
        self._add_subelement(root, 'EndOfSimulationPeriod',
                             self._date_format(end_simulation))
        self._add_subelement(root, 'SaveResults', save_results)
        self._add_subelement(root, 'SaveDistributedSimulations', save_dist)
        self._add_subelement(root, 'ComputeReffInterval', compute_interval)
        self._add_subelement(root, 'ComputeReffPeak', compute_peak)
        self._add_subelement(root, 'ComputeReffSeason', compute_season)
        self._add_subelement(root, 'ComputeReffWeightedQ', compute_weight)
        self._add_subelement(root, 'SeasonalReffStartDay', start_day)
        self._add_subelement(root, 'SeasonalReffEndDay', end_day)
        self._add_subelement(root, 'SeasonalReffStartMonth', start_month)
        self._add_subelement(root, 'SeasonalReffEndMonth', end_month)
        ri = self._add_subelement(root, 'ReffInterval')
        if compute_interval is True:
            if intervals is None:
                raise ValueError('Error: No Reff Intervals specified.')
            for value in intervals:
                # reff_intervals is a list of integers
                i = self._add_subelement(ri, 'ReffInterval')
                self._add_subelement(i, 'TimeStep', value)
        self._add_subelement(root, 'PlotPeriod', 365)
        rw = self._add_subelement(root, 'ReffWeights')
        if compute_weight is True:
            if weights is None:
                raise ValueError('Error: No weights specified.')
            for q, weight in weights.items():
                r = self._add_subelement(rw, 'ReffWeight')
                self._add_subelement(r, 'Q', q)
                self._add_subelement(r, 'Weight', weight)

        # Write out the xml file
        # ---------------------------------------------------------------------
        self._write(root, self.data_dir + filename)

    def batch_settings(
            self, filename='Batch_Simulation.xml', save_qsim_rows=False,
            save_qsim_cols=True, save_qsim_comp=True, save_qsim_comp_gw=False,
            header_line=True, save_peaks=False, save_freq_dist=True,
            save_gw_stats=False, create_param_files=False, dynamic_id=False,
            window_width=0, n_classes=0, use_precip_series=False,
            use_temp_series=False, use_evap_series=False, run_all=False,
            seasons=None, gap_file_in=True):
        """
        Generate an XML comfiguration file for the BatchRun function
        of HBV-light.

        Parameters
        ----------
        filename : str, optional
            Path and file name of the BatchRun configuration file,
            default is 'Batch_Simulation.xml'.
        save_qsim_rows : bool, optional
            Save simulated runoff results in rows, default is False.
        save_qsim_cols : bool, optional
            Save simulated runoff results in columns, default is True.
            It also saves a summary of the simulated runoff results.
        save_qsim_comp : bool, optional
            Save runoff components (rain, snow, glacier), default is True.
        save_qsim_comp_gw : bool, optional
            Save runoff groundwater components, default is False.
        header_line : bool, optional
            Set a header line with column names in all files, default is True.
        save_peaks : bool, optional
            Save simulated runoff value, snow amount, and relative soil
            moisture for the peak data points, in addition to weather
            statistics, default is False.
        save_freq_dist : bool, optional
            Save simulation runoff percentiles, maximum and minimum seasonal
            runoff values and their associated dates, in addition to average
            monthly simulated runoff values, default is True.
        save_gw_stats : bool, optional
            Save the 10th, 50th, and 90th quantiles of the water content in
            each groundwater box, default is False.
        create_param_files : bool, optional
            Create a Parameter.xml file for each parameter set,
            default is False.
        dynamic_id : bool, optional
            Compute statistics for dynamic identification, default is False.
        window_width : int, optional
            Window width for the dynamic identification, default is 0.
        n_classes : int, optional
            Number of classes for the dynamic identification, default is 0.
        use_precip_series : bool, optional
            Check each parameter set against multiple precipitation series,
            default is False.
        use_temp_series : bool, optional
            Check each parameter set against multiple temperature series,
            default is False.
        use_evap_series : bool, optional
            Check each parameter set against multiple evaporation series,
            default is False.
        run_all : bool, optional
            Check each parameter set against all possible combinations of
            precipitation, temperature, and evaporation series, default is
            False.
        seasons : list of tuples
            List of the start of each desired season, default is None.
            e.g. [(1, January), (1, March), ...]
        gap_file_in : bool, optional
            Use a GAP Calibration file format to run the BatchRun,
            default is True.

        """
        # Generate the XML configuration file
        # ---------------------------------------------------------------------
        root = ET.Element(
                'BatchSimulation', nsmap={'xsi': self.XSI, 'xsd': self.XSD})

        # Batch settings
        # ---------------------------------------------------------------------
        self._add_subelement(root, 'SaveQsim', save_qsim_rows)
        self._add_subelement(root, 'SaveQsimInColumns', save_qsim_cols)
        self._add_subelement(root, 'SaveQsimSourceComponents', save_qsim_comp)
        self._add_subelement(root, 'SaveQsimFluxComponents', save_qsim_comp_gw)
        self._add_subelement(root, 'SaveHeaderLine', header_line)
        self._add_subelement(root, 'SavePeaks', save_peaks)
        self._add_subelement(root, 'SaveFreqDist', save_freq_dist)
        self._add_subelement(root, 'SaveGroundwaterStatistics', save_gw_stats)
        self._add_subelement(root, 'CreateParameterFiles', create_param_files)
        self._add_subelement(root, 'DynamicIdentification', dynamic_id)
        self._add_subelement(root, 'WindowWidth', window_width)
        self._add_subelement(root, 'NoOfClasses', n_classes)
        self._add_subelement(root, 'UsePrecipitationSeries', use_precip_series)
        self._add_subelement(root, 'UseTemperatureSeries', use_temp_series)
        self._add_subelement(root, 'RunAllClimateSeriesCombinations', run_all)
        self._add_subelement(root, 'UseEvaporationSeries', use_evap_series)
        season = self._add_subelement(root, 'Seasons')
        if seasons is not None:
            for element in seasons:
                # seasons is a list of tuples (e.g. [(1, January), ...])
                self._add_subelement(season, 'Day', int(element[0]))
                self._add_subelement(season, 'Month', element[1])
        self._add_subelement(root, 'GapInputFile', gap_file_in)

        # Write out the xml file
        # ---------------------------------------------------------------------
        self._write(root, self.data_dir + filename)

    def parameter_settings(
            self, filename='Parameter.xml', cps=None, vgps=None):
        """
        Define the HBV-light Parameter.xml settings file.

        Parameters
        ----------
        filename : str, optional
            Path and file name of the Parameter settings file,
            default is 'Parameter.xml'.
        cps : dict, optional
            Dictionary containing catchment parameter names and values,
            default is None.
        vgps : dict, optional
            Dictionary containing vegetation zone parameter names and values,
            default is None.

        """
        # Define the catchment and vegetation zone parameters
        # ---------------------------------------------------------------------
        # Catchment parameters
        if cps is None:
            cps = {'KSI': 0, 'KGmin': 0, 'RangeKG': 0, 'AG': 0, 'PERC': 1,
                   'Alpha': 0, 'UZL': 30, 'K0': 0.25, 'K1': 0.1, 'K2': 0.01,
                   'MAXBAS': 1, 'Cet': 0.1, 'PCALT': 10, 'TCALT': 0.6,
                   'Pelev': 0, 'Telev': 0, 'PART': 0.5, 'DELAY': 1}

        # Vegetation zone parameters
        if vgps is None:
            vgps = {'TT': 0, 'CFMAX': 3, 'SFCF': 1, 'CFR': 0.05, 'CWH': 0.1,
                    'CFGlacier': 0, 'CFSlope': 1, 'FC': 120, 'LP': 1,
                    'BETA': 2}

        # Generate the XML configuration file
        # ---------------------------------------------------------------------
        root = ET.Element(
                'Catchment', nsmap={'xsi': self.XSI, 'xsd': self.XSD})

        # Parameter settings
        # ---------------------------------------------------------------------
        cp = self._add_subelement(root, 'CatchmentParameters')
        for parameter, value in cps.items():
            self._add_subelement(cp, parameter, value=value)
        vg = self._add_subelement(root, 'VegetationZone')
        vgp = self._add_subelement(vg, 'VegetationZoneParameters')
        for parameter, value in vgps.items():
            self._add_subelement(vgp, parameter, value=value)
        sc = self._add_subelement(root, 'SubCatchment')
        scp = self._add_subelement(sc, 'SubCatchmentParameters')
        for parameter, value in cps.items():
            self._add_subelement(scp, parameter, value=value)
        scvg = self._add_subelement(sc, 'SubCatchmentVegetationZone')
        scvgp = self._add_subelement(
                scvg, 'SubCatchmentVegetationZoneParameters')
        for parameter, value in vgps.items():
            self._add_subelement(scvgp, parameter, value=value)

        # Write out the xml file
        # ---------------------------------------------------------------------
        self._write(root, self.data_dir + filename)

    @staticmethod
    def _set_param_units(param_name, time_step='day'):
        """
        Set the appropriate unit for a given HBV-light parameter.

        Parameters
        ----------
        param_name : str
            Name of the HBV-light parameter.
        time_step : {'day', 'hour'}, optional
            Time step of the HBV-light input data, default is 'day'.

        Returns
        -------
        str
            Units of the given HBV-light parameter.

        Raises
        ------
        ValueError
            If the provided time step is not recognised.
        ValueError
            If the provided parameter name is not recognised.

        """
        # Set the appropriate format for the given time step.
        # ---------------------------------------------------------------------
        if time_step == 'day':
            ts = 'd'

        elif time_step == 'hour':
            ts = 'h'

        else:
            raise ValueError('Time step not recognised.')

        # Set the appropriate unit for the given HBV-light parameter.
        # ---------------------------------------------------------------------

        # Set the unicode for the degree sign
        degree_sign = u'\N{DEGREE SIGN}'

        if param_name in ['KSI', 'KGmin', 'RangeKG', 'K0', 'K1', 'K2']:
            return '1/' + ts

        elif param_name in ['Alpha', 'PART', 'SP', 'SFCF', 'CFR', 'CWH',
                            'CFGlacier', 'CFSlope', 'LP', 'BETA']:
            return '-'

        elif param_name in ['UZL', 'FC']:
            return 'mm'

        elif param_name == 'TT':
            return degree_sign + 'C'

        elif param_name == 'CFMAX':
            return 'mm/' + ts + ' ' + degree_sign + 'C'

        elif param_name in ['Pelev', 'Telev']:
            return 'm'

        elif param_name == 'PCALT':
            return '%/100m'

        elif param_name == 'TCALT':
            return degree_sign + 'C/100m'

        elif param_name == 'Cet':
            return '1/' + degree_sign + 'C'

        elif param_name in ['MAXBAS', 'DELAY']:
            return ts

        elif param_name == 'AG':
            return '1/mm'

        elif param_name == 'PERC':
            return 'mm/' + ts

        else:
            raise ValueError('Parameter name not recognised.')

    @staticmethod
    def _set_param_disp_name(param_name):
        """
        Return the display name of a given HBV-light parameter.

        Parameters
        ----------
        param_name : str
            HBV-light parameter name.

        Returns
        -------
        str
            HBV-light parameter display name.

        """
        if param_name == 'Pelev':
            return 'Elev. of P'

        elif param_name == 'Telev':
            return 'Elev. of T'

        else:
            return param_name

    @staticmethod
    def objective_function_name(obj_function):
        """
        Set the HBV-light name for a given objective function.

        Parameters
        ----------
        obj_fun : str
            Name of the objective function used in the analysis.

        Returns
        -------
        str
            Name of the objective function according to HBV-light.

        """
        if obj_function not in [
                'Reff', 'ReffWeighted', 'LogReff', 'R2', 'MeanDiff',
                'VolumeError', 'LindstromMeasure', 'MAREMeasure',
                'FlowWeightedReff', 'ReffInterval(i)', 'ReffSeason',
                'ReffPeak', 'SpearmanRank', 'ReffQObsSample',
                'SnowCover_RMSE', 'GW_Rspear', 'Glacier_MAE',
                'SWE_Reff', 'SWE_MANE', 'F_time', 'F_flow']:
            return 'PythonScript'

        else:
            return obj_function

    def gap_settings(
            self, catchment_params, veg_zone_params,
            filename='GAP_Simulation.xml', time_step='day',
            runs=5000, powell_runs=1000, parameter_sets=50, populations=2,
            exchange_freq=0, exchange_count=0, prob_optimisation=0.01,
            prob_mutation=0.02, prob_optimised=0, prob_random=0.16,
            prob_one=0.82, small_change=0, c_value=2, multiple_times=False,
            calibrations=100, obj_functions={'Reff': 1}):
        """
        Generate an XML configuration file for the GAP Calibration function
        of HBV-light.

        # TODO: So far it only works for one sub-catchment and one vegetation
        zone.

        Parameters
        ----------
        catchment_params : dict
            Dictionary containing the upper and lower limits for each
            catchment parameter, e.g.
            catchment_params={'K0': {'LowerLimit': 0.1, 'UpperLimit': 0.5}}.
        veg_zone_params : dict
            Dictionary containing the upper and lower limits for each
            vegetation zone parameter, e.g.
            veg_zone_params={'TT': {'LowerLimit': -0.5, 'UpperLimit': 0.5}}.
        filename : str, optional
            Path and file name of the GAP calibration configuration file,
            default is 'GAP_Simulation.xml'.
        time_step : {'day', 'hour'}, optional
            Time step of the simulation, default is 'day'.
        runs : int, optional
            Number of model runs, default is 5000.
        powell_runs : int, optional
            Number of runos for local optimisation, default is 1000.
        paramter_sets : int, optional
            Number of parameter sets per population, default is 50.
        populations : int, optional
            Number of populations, default is 2.
        exchange_freq : int, optional
            Frequency of parameter set exchange between different populations,
            default is 0.
        exchnage_count : int, optional
            Number of parameter sets that exchange between two populations,
            default is 0.
        prob_optimisation : float, optional
            Probability for optimisation between two sets, default is 0.01.
        prob_mutation : float, optional
            Probability for a mutation, default is 0.02.
        prob_optimised : float, optional
            Probability for optimising a value, default is 0.
        prob_random : float, optional
            Probability for a random value between the old values, default
            is 0.16.
        prob_one : float, optional
            Probability for taking one of the old values, default is 0.82.
        small_change : float, optional
            Portion of range for small change, default is 0.
        c_value : int, optional
            Value of C, default is 2.
        multiple_times : bool, optional
            Select whether the calibration should be repeated multiple times,
            default is False.
        calibrations : int, optional
            Number of times to repeat the calibration, default is 100.
        obj_functions : dict, optional
            Dictionary containing the objective functions to use and their
            associated wheights. Default is {'Reff': 1}.
        free_params : {'Snow', 'Soil_moisture', 'Response',
                       'Routing, 'All'}, optional
            HBV-light routine to get the free parameters for calibration from,
            default is 'Snow'.
        input_data_params : bool, optional
            Use routine-independent parameters as free_parameters,
            default is False.

        """
        # Generate the XML configuration file
        # ---------------------------------------------------------------------
        root = ET.Element(
                'GAPSimulation', nsmap={'xsi': self.XSI, 'xsd': self.XSD})

        # Model type and additional input data
        # ---------------------------------------------------------------------
        self._add_subelement(root, 'ModelType', self.model_type)
        self._add_subelement(root, 'ModelVariant', self.model_variant)
        self._add_subelement(root, 'UseOldSUZ', self.old_suz)
        self._add_subelement(root, 'ETCorrection', self.et_corr)
        self._add_subelement(root, 'PCALTFile', self.pcalt_data)
        self._add_subelement(root, 'TCALTFile', self.tcalt_data)

        # Catchment parameters
        # TODO: Parameter values for multiple subcatchments.
        # ---------------------------------------------------------------------
        c_params = self._add_subelement(root, 'GAPCatchmentParameters')
        for param in catchment_params:
            # Get parameter information
            unit = self._set_param_units(param, time_step='day')
            display_name = self._set_param_disp_name(param)
            value_low = catchment_params[param]['LowerLimit']
            value_high = catchment_params[param]['UpperLimit']
            # Append information to the XML tree
            cp = self._add_subelement(c_params, 'GAPCatchmentParameter')
            self._add_subelement(cp, 'Name', param)
            self._add_subelement(cp, 'DisplayName', display_name)
            self._add_subelement(cp, 'Unit', unit)
            self._add_subelement(cp, 'LowerLimit', value_low)
            self._add_subelement(cp, 'UpperLimit', value_high)
            self._add_subelement(cp, 'ValuesPerSubCatchment', False)

        # Vegetation zone parameters
        # TODO: Parameter values for multiple vegetation zones.
        # ---------------------------------------------------------------------
        vz_params = self._add_subelement(root, 'GAPVegetationZoneParameters')
        for param in veg_zone_params:
            # Get parameter information
            unit = self._set_param_units(param, time_step='day')
            display_name = self._set_param_disp_name(param)
            value_low = veg_zone_params[param]['LowerLimit']
            value_high = veg_zone_params[param]['UpperLimit']
            # Append information to the XML tree
            vzp = self._add_subelement(vz_params, 'GAPVegetationZoneParameter')
            self._add_subelement(vzp, 'Name', param)
            self._add_subelement(vzp, 'DisplayName', display_name)
            self._add_subelement(vzp, 'Unit', unit)
            self._add_subelement(vzp, 'LowerLimit', value_low)
            self._add_subelement(vzp, 'UpperLimit', value_high)
            self._add_subelement(vzp, 'ValuesPerSubCatchment', False)
            self._add_subelement(vzp, 'ValuesPerVegetationZone', False)
            self._add_subelement(vzp, 'RandomValuePerVegetationZone', False)

        # GAP run parameters
        # ---------------------------------------------------------------------
        self._add_subelement(root, 'NumberOfRuns', runs)
        self._add_subelement(root, 'NumberOfPowellRuns', powell_runs)
        self._add_subelement(root, 'NumberOfParameterSets', parameter_sets)
        self._add_subelement(root, 'NumberOfPopulations', populations)
        self._add_subelement(root, 'ExchangeFrequency', exchange_freq)
        self._add_subelement(root, 'ExchangeCount', exchange_count)
        self._add_subelement(root, 'ProbOptimizeBetweeSets', prob_optimisation)
        self._add_subelement(root, 'ProbMutation', prob_mutation)
        self._add_subelement(root, 'ProbOptimized', prob_optimised)
        self._add_subelement(root, 'ProbRandom', prob_random)
        self._add_subelement(root, 'ProbOne', prob_one)
        self._add_subelement(root, 'SmallChange', small_change)
        self._add_subelement(root, 'CValue', c_value)
        self._add_subelement(root, 'CalibrateMultipleTimes', multiple_times)
        self._add_subelement(root, 'NumberOfCalibrations', calibrations)
        pps = self._add_subelement(root, 'Populations')
        for p in range(populations):
            pp = self._add_subelement(pps, 'GAPPopulation')
            self._add_subelement(pp, 'Name', 'Population_' + str(p+1))
            ofws = self._add_subelement(pp, 'ObjFunctionWeights')
            for name, weight in obj_functions.items():
                ofw = self._add_subelement(ofws, 'ObjFunctionWeight')
                of = self._add_subelement(ofw, 'ObjFunction')
                self._add_subelement(of, 'Name', name)
                self._add_subelement(of, 'Index', -1)
                self._add_subelement(ofw, 'Weight', weight)
        # TODO: Set weights for different subcatchments.
        self._add_subelement(root, 'ObjFunctionUsage', 'UseOutlet')
        self._add_subelement(root, 'SubCatchmentWeights')

        # Write out the xml file
        # ---------------------------------------------------------------------
        self._write(root, self.data_dir + filename)

    def mc_settings(
            self, catchment_params, veg_zone_params, gap_folder=None,
            eff_measure=None, filename='MC_Simulation.xml', time_step='day',
            multi_periods=False, periods=None, runs=1000, save_runs='SaveAll',
            min_reff_val=0.6, gaussian=False, obj_functions=None):
        """
        Generate an XML configuration file for the Monte Carlo Simulation
        function of HBV-light.

        Parameters
        ----------
        catchment_params : dict
            Dictionary containing the upper and lower limits for each
            catchment parameter, e.g.
            catchment_params={'K0': {'LowerLimit': 0.1, 'UpperLimit': 0.5}}.
        veg_zone_params : dict
            Dictionary containing the upper and lower limits for each
            vegetation zone parameter, e.g.
            veg_zone_params={'TT': {'LowerLimit': -0.5, 'UpperLimit': 0.5}}.
        filename : str, optional
            Path and file name of the MonteCarlo configuration file,
            default is 'GAP_Simulation.xml'.
        time_step : {'day', 'hour'}, optional
            Time step of the simulation, default is 'day'.
        gap_folder : str, optional
            Name of the folder in which the GAP results are stored,
            relavant only for sensitivity analysis. Default is None.
        eff_measure : str or list, optional
            List of the efficiency measures to use to sort the calibrations,
            relevant for sensitivity analysis. Default is None.
        filename : str, optional
            Path and file name of the GAP calibration configuration file,
            default is 'MC_Simulation.xml'.
        multi_periods : bool, optional
            Divide the simulation period into multiple parts. The efficiency
            of the model run will be computed for each of the periods,
            default is False.
        periods : list, optional
            List of dates representing the start of each period. The dates
            should be provided with the the format '%Y-%m-%d'. Default is None.
        runs : int, optional
            Number of model runs that will be carried out during the Monte
            Carlo simulation, default is 1000.
        save_runs : {'SaveAll', 'SaveOnlyIf', 'Save100Best'}, optional
            Specify which model runs should be save, default is 'SaveAll'.
        min_reff_val : float, optional
            Select the model efficiency value above which model runs are being
            saved if the the save_runs parameter is set to 'SaveOnlyIf',
            default is 0.6.
        gaussian : bool, optional
            Selected whether random numbers should be generated from a
            Gaussian continuous probability distribution, default is False.
        obj_functions : dict, optional
            Dictionary containing the objective functions to use and their
            associated wheights, default is None. Example: {'Reff': 1}

        """
        # Generate the XML configuration file
        # ---------------------------------------------------------------------
        root = ET.Element('MonteCarloSimulation',
                          nsmap={'xsi': self.XSI, 'xsd': self.XSD})

        # Model type and additional input data
        # ---------------------------------------------------------------------
        self._add_subelement(root, 'ModelType', self.model_type)
        self._add_subelement(root, 'ModelVariant', self.model_variant)
        self._add_subelement(root, 'UseOldSUZ', self.old_suz)
        self._add_subelement(root, 'ETCorrection', self.et_corr)
        self._add_subelement(root, 'PCALTFile', self.pcalt_data)
        self._add_subelement(root, 'TCALTFile', self.tcalt_data)

        # Catchment parameters
        # TODO: Parameter values for multiple subcatchments.
        # ---------------------------------------------------------------------
        c_params = self._add_subelement(root, 'MonteCarloCatchmentParameters')
        for param in catchment_params:
            # Get parameter information
            unit = self._set_param_units(param, time_step='day')
            display_name = self._set_param_disp_name(param)
            value_low = catchment_params[param]['LowerLimit']
            value_high = catchment_params[param]['UpperLimit']
            if gaussian is True:
                mean = catchment_params[param]['Mean']
                sd = catchment_params[param]['SD']
            # Append information to the XML tree
            cp = self._add_subelement(c_params, 'MonteCarloCatchmentParameter')
            self._add_subelement(cp, 'Name', param)
            self._add_subelement(cp, 'DisplayName', display_name)
            self._add_subelement(cp, 'Unit', unit)
            self._add_subelement(cp, 'LowerLimit', value_low)
            self._add_subelement(cp, 'UpperLimit', value_high)
            if gaussian is True:
                self._add_subelement(cp, 'Mean', mean)
                self._add_subelement(cp, 'SD', sd)
            self._add_subelement(cp, 'ValuesPerSubCatchment', False)
            self._add_subelement(cp, 'CatchmentOption', 'Random')

        # Vegetation zone parameters
        # TODO: Parameter values for multiple vegetation zones.
        # ---------------------------------------------------------------------
        vz_params = self._add_subelement(
                root, 'MonteCarloVegetationZoneParameters')
        for param in veg_zone_params:
            # Get parameter information
            unit = self._set_param_units(param, time_step='day')
            display_name = self._set_param_disp_name(param)
            value_low = veg_zone_params[param]['LowerLimit']
            value_high = veg_zone_params[param]['UpperLimit']
            if gaussian is True:
                mean = veg_zone_params[param]['Mean']
                sd = veg_zone_params[param]['SD']
            # Append information to the XML tree
            vzp = self._add_subelement(
                    vz_params, 'MonteCarloVegetationZoneParameter')
            self._add_subelement(vzp, 'Name', param)
            self._add_subelement(vzp, 'DisplayName', display_name)
            self._add_subelement(vzp, 'Unit', unit)
            self._add_subelement(vzp, 'LowerLimit', value_low)
            self._add_subelement(vzp, 'UpperLimit', value_high)
            if gaussian is True:
                self._add_subelement(vzp, 'Mean', mean)
                self._add_subelement(vzp, 'SD', sd)
            self._add_subelement(vzp, 'ValuesPerSubCatchment', False)
            self._add_subelement(vzp, 'ValuesPerVegetationZone', False)
            self._add_subelement(vzp, 'CatchmentOption', 'Random')
            self._add_subelement(vzp, 'VegetationZoneOption', 'Random')

        # Monte Carlo parameters
        # ---------------------------------------------------------------------
        self._add_subelement(root, 'MultiPeriods', multi_periods)
        if multi_periods is False:
            p = self._add_subelement(root, 'Periods')
        elif multi_periods is True and periods is None:
            raise ValueError('Periods need to be specified.')
        else:
            for element in periods:
                self._add_subelement(p, 'Period', element)
        self._add_subelement(root, 'NumberOfRuns', runs)
        self._add_subelement(root, 'SaveRunsOption', save_runs)
        self._add_subelement(root, 'ObjFunctionUsage', 'UseOutlet')
        self._add_subelement(root, 'MinReffValue', min_reff_val)
        self._add_subelement(root, 'Gaussian', gaussian)
        ofws = self._add_subelement(root, 'ObjFunctionWeights')
        if save_runs in ['SaveOnlyIf', 'Save100Best']:
            for name, weight in obj_functions.items():
                ofw = self._add_subelement(ofws, 'ObjFunctionWeight')
                of = self._add_subelement(ofw, 'ObjFunction')
                self._add_subelement(of, 'Name', name)
                self._add_subelement(of, 'Index', -1)
                self._add_subelement(ofw, 'Weight', weight)
        # TODO: Set weights for different subcatchments.
        self._add_subelement(root, 'SubCatchmentWeights')

        # Write out the xml file
        # ---------------------------------------------------------------------
        self._write(root, self.data_dir + filename)

    @classmethod
    def _add_evu_element(cls, root, model_variant='Basic'):
        """
        Parameters
        ----------
        root : lxml.etree.Element or lxml.etree.SubElement
            Element or sub-element to add a sub-element to.
        model_variant : {'Basic', 'Aspect', 'Glacier'}, optional
            Name of the model variant, default is 'Basic'.

        Returns
        -------
            XML root subelement containing an 'EVU' type.

        """
        return cls._add_subelement(
                root, 'EVU', attribute={cls.XSI_TYPE: model_variant + '_EVU'})

    @staticmethod
    def _get_area_value(df, idx, scc, vzc, apc, igc=None):
        """
        Parameters
        ----------
        df : Pandas.DataFrame
            DataFrame containing the area fractions for each elevation zone,
            sub-catchment, and vegetation zone
        idx : int
            Index of the elevation zone
        scc : int
            Sub-catchment count value
        vzc : int
            Vegetation zone count value
        apc : str
            Aspect count
        igc : str
            Glacier count

        Returns
        -------
        Pandas.DataFrame cell value
            Value corresponding to the given sub-catchment, vegetation zone,
            elevation, zone, aspect, and glacier.

        """
        if igc is None:
            return df['Area_' + scc + '_' + vzc + '_' + apc].iloc[idx]

        else:
            return df['Area_' + scc + '_' + vzc +
                      '_' + igc + '_' + apc].iloc[idx]

    @classmethod
    def _set_aspect_value(cls, root, aspect, value):
        """
        Parameters
        ----------
        root : lxml.etree.Element or lxml.etree.SubElement
            xml element to add the aspect value to.
        aspect : {'n', 's', 'ew'}
            Aspect to set the area for.
        value : float
            Area value for the given aspect.

        Raises
        ------
        ValueError
            If an unknown aspect is provided.

        """
        if aspect == 'n':
            cls._add_subelement(root, 'NorthArea', value)

        elif aspect == 's':
            cls._add_subelement(root, 'SouthArea', value)

        elif aspect == 'ew':
            cls._add_subelement(root, 'EastWestArea', value)

        else:
            raise ValueError('Error: Unknown aspect')

    def catchment_settings(
            self, data, filename='Clarea.xml', lakes=None, areas=None):
        """
        Generate a clarea.xml file based on data structured as a Pandas
        DataFrame.

        # NOTE: Even if this method belongs with the HBV-light configuration
        settings (xml), it is a part of the input data for the model. It is
        defined as a class method so it can be used within the data module
        without any restrictions.

        Parameters
        ----------
        data : Pandas.DataFrame
            Where the elevations are in the index and each column
            is a different SubCatchment/VegetationZone.
            | Index || Area_1_1 | Area_1_2 | ...
            | 1800  || 0.211    | 0.39     | ...
            | ...
            Where the first index indicates the SubCatchment and
            the second index indicates the VegetationZone.
        filename : str, optional
            Path and filename of the output file, default is 'Clarea.xml'.
        lakes : Dict, optional
            lakes = {'Lake_1': {'Area': 0,
                                'Elevation': 0,
                                'TT': 0,
                                'SFCF': 0},
                     'Lake_2': ...}, default is None.
        areas : Dict, optional
            areas = {'Area_1': 0, 'Area_2': 0, ...}, default is None.

        Raises
        ------
        ValueError
            If an invalid model variant is provided.

        """
        # Get the number of vegetation zones
        # ---------------------------------------------------------------------
        vg = []
        for value in data.columns:
            vg.append(int(value[7]))
        vegetation_zone_count = max(vg)

        # Get the number of subcatchments
        # ---------------------------------------------------------------------
        sc = []
        for value in data.columns:
            sc.append(int(value[5]))
        sub_catchment_count = max(sc)

        # Generate the XML configuration file
        # ---------------------------------------------------------------------
        root = ET.Element(
                'Catchment', nsmap={'xsi': self.XSI, 'xsd': self.XSD})

        # Elevation zones
        # ---------------------------------------------------------------------
        self._add_subelement(
                root, 'VegetationZoneCount', vegetation_zone_count)
        elevz = self._add_subelement(root, 'ElevationZoneHeight')
        for value in data.index:
            self._add_subelement(elevz, 'double', int(value))

        # Fraction areas
        # ---------------------------------------------------------------------
        for scc in range(1, sub_catchment_count+1):
            scc = str(scc)
            subc = self._add_subelement(root, 'SubCatchment')
            for vzc in range(1, vegetation_zone_count+1):
                vzc = str(vzc)
                vegz = self._add_subelement(subc, 'VegetationZone')
                if self.model_variant == 'Basic':
                    for value in data['Area_' + scc + '_' + vzc]:
                        evu = self._add_evu_element(
                                vegz, model_variant=self.model_variant)
                        self._add_subelement(evu, 'Area', value)

                elif self.model_variant == 'Aspect':
                    for idx in range(len(data.index)):
                        idx = str(idx)
                        evu = self._add_evu_element(
                                vegz, model_variant=self.model_variant)
                        for apc in ['n', 's', 'ew']:
                            value = self._get_area_value(
                                    data, idx, scc, vzc, apc)
                            self._set_aspect_value(evu, apc, value)

                elif self.model_variant == 'Glacier':
                    if data['Area_' + scc + '_' + vzc + '_g_n']:
                        is_glacier = True
                        igc = 'g'  # glacier
                    elif data['Area_' + scc + '_' + vzc + '_c_n']:
                        is_glacier = False
                        igc = 'c'  # clear (no glacier)
                    else:
                        raise ValueError('Glacier definition not valid')
                    for idx in range(len(data.index)):
                        idx = str(idx)
                        evu = self._add_evu_element(
                                vegz, model_variant=self.model_variant)
                        self._add_subelement(evu, 'IsGlacier', is_glacier)
                        for apc in ['n', 's', 'ew']:
                            value = self._get_area_value(
                                    data, idx, scc, vzc, apc, igc)
                            self._set_aspect_value(evu, apc, value)

                else:
                    raise ValueError('Invalid model variant provided')

            # Lake parameters
            # -----------------------------------------------------------------
            n = str(scc)
            lake = self._add_subelement(subc, 'Lake')
            if lakes is None:
                self._add_subelement(lake, 'Area', 0)
                self._add_subelement(lake, 'Elevation', 0)
                self._add_subelement(lake, 'TT', 0)
                self._add_subelement(lake, 'SFCF', 0)
            else:
                self._add_subelement(lake, 'Area', lakes['Lake_' + n]['Area'])
                self._add_subelement(
                        lake, 'Elevation', lakes['Lake_' + n]['Elevation'])
                self._add_subelement(lake, 'TT', lakes['Lake_' + n]['TT'])
                self._add_subelement(lake, 'SFCF', lakes['Lake_' + n]['SFCF'])

            # Absolute area
            # -----------------------------------------------------------------
            if areas is None:
                self._add_subelement(subc, 'AbsoluteArea', 0)
            else:
                self._add_subelement(subc, 'AbsoluteArea', areas['Area_' + n])

        # Write out the xml file
        # ---------------------------------------------------------------------
        self._write(root, self.data_dir + filename)

    def load_catchment_settings(self, filename='Clarea.xml'):
        """
        Load the elevation distribution data for the catchment.

        Parameters
        ----------
        filename : str, optional
            Name for the catchment settings file.'Clarea.xml' file,
            default is 'Clarea.xml'.

        Returns
        -------
        Pandas.Series
            Data structure containing the mean elevation and area fraction
            of each elevation band in which the catchment is partitioned.

        """
        # Parse the file contents
        tree = ET.parse(self.data_dir + filename)
        root = tree.getroot()

        # Initialise lists to store the average elevation and
        # fraction area of each elevation band.
        elevs = []
        areas = []

        for child in root:

            if child.tag == 'ElevationZoneHeight':
                for item in child:
                    elevs.append(int(item.text))

            if child.tag == 'SubCatchment':
                for item in child:
                    if item.tag == 'VegetationZone':
                        for area in item:
                            areas.append(float(area[0].text))

        return pd.Series(data=areas, index=elevs, name='Area')
