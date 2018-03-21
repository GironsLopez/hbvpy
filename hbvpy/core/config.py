#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hbvpy.config
============

**A package to generate xml configuration files needed to run HBV-light.**

This package is intended to provide functions and method to generate
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

import numpy as np
import pandas as pd
from lxml import etree as ET
from datetime import datetime as dt


__all__ = ['HBVconfig']


class HBVconfig(object):
    """
    Generate an HBV-light Catchment folder structure and configuration files.

    """
    XSI = 'http://www.w3.org/2001/XMLSchema-instance'
    XSD = 'http://www.w3.org/2001/XMLSchema'

    XSI_TYPE = '{%s}type' % XSI

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

    @classmethod
    def model_settings(
            cls, filename='Simulation.xml', model_type='Standard',
            model_variant='Basic', start_warm_up='1900-01-01',
            start_simulation='1900-01-01', end_simulation='1900-01-01',
            save_results=True, save_dist=True, compute_interval=False,
            intervals=None, compute_peak=True, compute_season=False,
            compute_weight=False, start_day=1, end_day=31,
            start_month='January', end_month='December', weights=None,
            old_suz=True):
        """
        Define the HBV-light model settings.

        Parameters
        ----------
        filename : str, optional
            Path and file name of the model settings file name,
            default is 'Simulation.xml'.
        model_type : {'Standard', 'OnlySnowDistributed', 'DistributedSUZ',
                      'ThreeGWBoxes', ThreeGWBoxesDistributedSTZ',
                      'ThreeGWBoxesDistributedSTZandSUZ', 'ResponseDelayed'
                      'OneGWBox'}, optional
            Model type to use, default is 'Standard'.
        model_variant : {'Basic', 'Aspect', 'Glacier'}, optional
            Model variant to use, default is 'Basic'.
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
        old_suz : bool, optional
            Use UZL and K0 in SUZ-box and therefore two runoff components from
            the storage in the soil upper zone, default is True.

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
                'Simulation', nsmap={'xsi': cls.XSI, 'xsd': cls.XSD})

        # Model type and additional input data
        # ---------------------------------------------------------------------
        cls._add_subelement(root, 'ModelType', model_type)
        cls._add_subelement(root, 'ModelVariant', model_variant)
        cls._add_subelement(root, 'UseOldSUZ', old_suz)

        # Simulation settings
        # ---------------------------------------------------------------------
        cls._add_subelement(root, 'StartOfWarmingUpPeriod',
                            cls._date_format(start_warm_up))
        cls._add_subelement(root, 'StartOfSimulationPeriod',
                            cls._date_format(start_simulation))
        cls._add_subelement(root, 'EndOfSimulationPeriod',
                            cls._date_format(end_simulation))
        cls._add_subelement(root, 'SaveResults', save_results)
        cls._add_subelement(root, 'SaveDistributedSimulations', save_dist)
        cls._add_subelement(root, 'ComputeReffInterval', compute_interval)
        cls._add_subelement(root, 'ComputeReffPeak', compute_peak)
        cls._add_subelement(root, 'ComputeReffSeason', compute_season)
        cls._add_subelement(root, 'ComputeReffWeightedQ', compute_weight)
        cls._add_subelement(root, 'SeasonalReffStartDay', start_day)
        cls._add_subelement(root, 'SeasonalReffEndDay', end_day)
        cls._add_subelement(root, 'SeasonalReffStartMonth', start_month)
        cls._add_subelement(root, 'SeasonalReffEndMonth', end_month)
        ri = cls._add_subelement(root, 'ReffInterval')
        if compute_interval is True:
            if intervals is None:
                raise ValueError('Error: No Reff Intervals specified.')
            for value in intervals:
                # reff_intervals is a list of integers
                i = cls._add_subelement(ri, 'ReffInterval')
                cls._add_subelement(i, 'TimeStep', value)
        cls._add_subelement(root, 'PlotPeriod', 365)
        rw = cls._add_subelement(root, 'ReffWeights')
        if compute_weight is True:
            if weights is None:
                raise ValueError('Error: No weights specified.')
            for q, weight in weights.items():
                r = cls._add_subelement(rw, 'ReffWeight')
                cls._add_subelement(r, 'Q', q)
                cls._add_subelement(r, 'Weight', weight)

        # Write out the xml file
        # ---------------------------------------------------------------------
        cls._write(root, filename)

    @classmethod
    def batch_settings(
            cls, filename='Batch_Simulation.xml', save_qsim_rows=False,
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
                'BatchSimulation', nsmap={'xsi': cls.XSI, 'xsd': cls.XSD})

        # Batch settings
        # ---------------------------------------------------------------------
        cls._add_subelement(root, 'SaveQsim', save_qsim_rows)
        cls._add_subelement(root, 'SaveQsimInColumns', save_qsim_cols)
        cls._add_subelement(root, 'SaveQsimSourceComponents', save_qsim_comp)
        cls._add_subelement(root, 'SaveQsimFluxComponents', save_qsim_comp_gw)
        cls._add_subelement(root, 'SaveHeaderLine', header_line)
        cls._add_subelement(root, 'SavePeaks', save_peaks)
        cls._add_subelement(root, 'SaveFreqDist', save_freq_dist)
        cls._add_subelement(root, 'SaveGroundwaterStatistics', save_gw_stats)
        cls._add_subelement(root, 'CreateParameterFiles', create_param_files)
        cls._add_subelement(root, 'DynamicIdentification', dynamic_id)
        cls._add_subelement(root, 'WindowWidth', window_width)
        cls._add_subelement(root, 'NoOfClasses', n_classes)
        cls._add_subelement(root, 'UsePrecipitationSeries', use_precip_series)
        cls._add_subelement(root, 'UseTemperatureSeries', use_temp_series)
        cls._add_subelement(root, 'RunAllClimateSeriesCombinations', run_all)
        cls._add_subelement(root, 'UseEvaporationSeries', use_evap_series)
        season = cls._add_subelement(root, 'Seasons')
        if seasons is not None:
            for element in seasons:
                # seasons is a list of tuples (e.g. [(1, January), ...])
                cls._add_subelement(season, 'Day', int(element[0]))
                cls._add_subelement(season, 'Month', element[1])
        cls._add_subelement(root, 'GapInputFile', gap_file_in)

        # Write out the xml file
        # ---------------------------------------------------------------------
        cls._write(root, filename)

    @classmethod
    def parameter_settings(cls, filename='Parameter.xml'):
        """
        Define the HBV-light Parameter.xml settings file.

        Parameters
        ----------
        filename : str, optional
            Path and file name of the Parameter settings file,
            default is 'Parameter.xml'

        """
        # Define the catchment and vegetation zone parameters
        # ---------------------------------------------------------------------
        # Catchments parameters
        cps = {'KSI': 0, 'KGmin': 0, 'RangeKG': 0, 'AG': 0, 'PERC': 1,
               'Alpha': 0, 'UZL': 30, 'K0': 0.25, 'K1': 0.1, 'K2': 0.01,
               'MAXBAS': 1, 'Cet': 0.1, 'PCALT': 10, 'TCALT': 0.6,
               'Pelev': 0, 'Telev': 0, 'PART': 0.5, 'DELAY': 1}

        # Vegetation zone parameters
        vgps = {'TT': 0, 'CFMAX': 3, 'SFCF': 1, 'CFR': 0.05, 'CWH': 0.1,
                'CFGlacier': 0, 'CFSlope': 1, 'FC': 120, 'LP': 1, 'BETA': 2}

        # Generate the XML configuration file
        # ---------------------------------------------------------------------
        root = ET.Element(
                'Catchment', nsmap={'xsi': cls.XSI, 'xsd': cls.XSD})

        # Parameter settings
        # ---------------------------------------------------------------------
        cp = cls._add_subelement(root, 'CatchmentParameters')
        for parameter, value in cps.items():
            cls._add_subelement(cp, parameter, value=value)
        vg = cls._add_subelement(root, 'VegetationZone')
        vgp = cls._add_subelement(vg, 'VegetationZoneParameters')
        for parameter, value in vgps.items():
            cls._add_subelement(vgp, parameter, value=value)
        sc = cls._add_subelement(root, 'SubCatchment')
        scp = cls._add_subelement(sc, 'SubCatchmentParameters')
        for parameter, value in cps.items():
            cls._add_subelement(scp, parameter, value=value)
        scvg = cls._add_subelement(sc, 'SubCatchmentVegetationZone')
        scvgp = cls._add_subelement(
                scvg, 'SubCatchmentVegetationZoneParameters')
        for parameter, value in vgps.items():
            cls._add_subelement(scvgp, parameter, value=value)

        # Write out the xml file
        # ---------------------------------------------------------------------
        cls._write(root, filename)

    @staticmethod
    def _set_param_units(param_name, time_step='day'):
        """
        Set the appropriate unit for a given HBV-light parameter.

        Parameters
        ----------
        param_name : str
            Name of the HBV-light parameter.
        time_step : {'day', 'hour'}, optional
            Time step of the HBV-light input data.

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
        if time_step == 'day':
            ts = 'd'
        elif time_step == 'hour':
            ts = 'h'
        else:
            raise ValueError('Time step not recognised.')

        # Set the appropriate unit for the given HBV-light parameter.
        if param_name in ['KSI', 'KGmin', 'RangeKG', 'K0', 'K1', 'K2']:
            return '1/' + ts
        elif param_name in ['Alpha', 'PART', 'SP', 'SFCF', 'CFR', 'CWH',
                            'CFGlacier', 'CFSlope', 'LP', 'BETA']:
            return '-'
        elif param_name in ['UZL', 'FC']:
            return 'mm'
        elif param_name == 'TT':
            return '°C'
        elif param_name == 'CFMAX':
            return 'mm/' + ts + ' °C'
        elif param_name in ['Pelev', 'Telev']:
            return 'm'
        elif param_name == 'PCALT':
            return '%/100m'
        elif param_name == 'TCALT':
            return '°C/100m'
        elif param_name == 'Cet':
            return '1/°C'
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

    @classmethod
    def gap_settings(
            cls, catchment_params, veg_zone_params,
            filename='GAP_Simulation.xml', time_step='day',
            model_type='Standard', model_variant='Basic', old_suz=True,
            et_correction=False, pcalt_data=False, tcalt_data=False,
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
        et_correction : bool, optional
            Choose whether to correct the potential evapotranspiration with
            the long-term average air temperature for each day of the year,
            default is False
        pcalt_data : bool, optional
            Choose whether to use measured precipitation lapse rate values
            instead of a model parameter, default is False.
        tcalt_data : bool, optional
            Choose whether to use measure temperature lapse rate values
            instead of a model parameter, default is False.
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
                'GAPSimulation', nsmap={'xsi': cls.XSI, 'xsd': cls.XSD})

        # Model type and additional input data
        # ---------------------------------------------------------------------
        cls._add_subelement(root, 'ModelType', model_type)
        cls._add_subelement(root, 'ModelVariant', model_variant)
        cls._add_subelement(root, 'UseOldSUZ', old_suz)
        cls._add_subelement(root, 'ETCorrection', et_correction)
        cls._add_subelement(root, 'PCALTFile', pcalt_data)
        cls._add_subelement(root, 'TCALTFile', tcalt_data)

        # Catchment parameters
        # TODO: Parameter values for multiple subcatchments.
        # ---------------------------------------------------------------------
        c_params = cls._add_subelement(root, 'GAPCatchmentParameters')
        for param in catchment_params:
            # Get parameter information
            unit = cls._set_param_units(param, time_step='day')
            display_name = cls._set_param_disp_name(param)
            value_low = catchment_params[param]['LowerLimit']
            value_high = catchment_params[param]['UpperLimit']
            # Append information to the XML tree
            cp = cls._add_subelement(c_params, 'GAPCatchmentParameter')
            cls._add_subelement(cp, 'Name', param)
            cls._add_subelement(cp, 'DisplayName', display_name)
            cls._add_subelement(cp, 'Unit', unit)
            cls._add_subelement(cp, 'LowerLimit', value_low)
            cls._add_subelement(cp, 'UpperLimit', value_high)
            cls._add_subelement(cp, 'ValuesPerSubCatchment', False)

        # Vegetation zone parameters
        # TODO: Parameter values for multiple vegetation zones.
        # ---------------------------------------------------------------------
        vz_params = cls._add_subelement(root, 'GAPVegetationZoneParameters')
        for param in veg_zone_params:
            # Get parameter information
            unit = cls._set_param_units(param, time_step='day')
            display_name = cls._set_param_disp_name(param)
            value_low = veg_zone_params[param]['LowerLimit']
            value_high = veg_zone_params[param]['UpperLimit']
            # Append information to the XML tree
            vzp = cls._add_subelement(vz_params, 'GAPVegetationZoneParameter')
            cls._add_subelement(vzp, 'Name', param)
            cls._add_subelement(vzp, 'DisplayName', display_name)
            cls._add_subelement(vzp, 'Unit', unit)
            cls._add_subelement(vzp, 'LowerLimit', value_low)
            cls._add_subelement(vzp, 'UpperLimit', value_high)
            cls._add_subelement(vzp, 'ValuesPerSubCatchment', False)
            cls._add_subelement(vzp, 'ValuesPerVegetationZone', False)
            cls._add_subelement(vzp, 'RandomValuePerVegetationZone', False)

        # GAP run parameters
        # ---------------------------------------------------------------------
        cls._add_subelement(root, 'NumberOfRuns', runs)
        cls._add_subelement(root, 'NumberOfPowellRuns', powell_runs)
        cls._add_subelement(root, 'NumberOfParameterSets', parameter_sets)
        cls._add_subelement(root, 'NumberOfPopulations', populations)
        cls._add_subelement(root, 'ExchangeFrequency', exchange_freq)
        cls._add_subelement(root, 'ExchangeCount', exchange_count)
        cls._add_subelement(root, 'ProbOptimizeBetweeSets', prob_optimisation)
        cls._add_subelement(root, 'ProbMutation', prob_mutation)
        cls._add_subelement(root, 'ProbOptimized', prob_optimised)
        cls._add_subelement(root, 'ProbRandom', prob_random)
        cls._add_subelement(root, 'ProbOne', prob_one)
        cls._add_subelement(root, 'SmallChange', small_change)
        cls._add_subelement(root, 'CValue', c_value)
        cls._add_subelement(root, 'CalibrateMultipleTimes', multiple_times)
        cls._add_subelement(root, 'NumberOfCalibrations', calibrations)
        pps = cls._add_subelement(root, 'Populations')
        for p in range(populations):
            pp = cls._add_subelement(pps, 'GAPPopulation')
            cls._add_subelement(pp, 'Name', 'Population_' + str(p+1))
            ofws = cls._add_subelement(pp, 'ObjFunctionWeights')
            for name, weight in obj_functions.items():
                ofw = cls._add_subelement(ofws, 'ObjFunctionWeight')
                of = cls._add_subelement(ofw, 'ObjFunction')
                cls._add_subelement(of, 'Name', name)
                cls._add_subelement(of, 'Index', -1)
                cls._add_subelement(ofw, 'Weight', weight)
        # TODO: Set weights for different subcatchments.
        cls._add_subelement(root, 'ObjFunctionUsage', 'UseOutlet')
        cls._add_subelement(root, 'SubCatchmentWeights')

        # Write out the xml file
        # ---------------------------------------------------------------------
        cls._write(root, filename)

    @classmethod
    def mc_settings(
            cls, catchment_params, veg_zone_params,
            filename='MC_Simulation.xml', time_step='day',
            model_type='Standard', model_variant='Basic', old_suz=True,
            et_correction=False, pcalt_data=False, tcalt_data=False,
            gap_folder=None, eff_measure=None, multi_periods=False,
            periods=None, runs=1000, save_runs='SaveAll', min_reff_val=0.6,
            gaussian=False, obj_functions=None):
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
        et_correction : bool, optional
            Choose whether to correct the potential evapotranspiration with
            the long-term average air temperature for each day of the year,
            default is False
        pcalt_data : bool, optional
            Choose whether to use measured precipitation lapse rate values
            instead of a model parameter, default is False.
        tcalt_data : bool, optional
            Choose whether to use measure temperature lapse rate values
            instead of a model parameter, default is False.
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
                          nsmap={'xsi': cls.XSI, 'xsd': cls.XSD})

        # Model type and additional input data
        # ---------------------------------------------------------------------
        cls._add_subelement(root, 'ModelType', model_type)
        cls._add_subelement(root, 'ModelVariant', model_variant)
        cls._add_subelement(root, 'UseOldSUZ', old_suz)
        cls._add_subelement(root, 'ETCorrection', et_correction)
        cls._add_subelement(root, 'PCALTFile', pcalt_data)
        cls._add_subelement(root, 'TCALTFile', tcalt_data)

        # Catchment parameters
        # TODO: Parameter values for multiple subcatchments.
        # ---------------------------------------------------------------------
        c_params = cls._add_subelement(root, 'MonteCarloCatchmentParameters')
        for param in catchment_params:
            # Get parameter information
            unit = cls._set_param_units(param, time_step='day')
            display_name = cls._set_param_disp_name(param)
            value_low = catchment_params[param]['LowerLimit']
            value_high = catchment_params[param]['UpperLimit']
            if gaussian is True:
                mean = catchment_params[param]['Mean']
                sd = catchment_params[param]['SD']
            # Append information to the XML tree
            cp = cls._add_subelement(c_params, 'MonteCarloCatchmentParameter')
            cls._add_subelement(cp, 'Name', param)
            cls._add_subelement(cp, 'DisplayName', display_name)
            cls._add_subelement(cp, 'Unit', unit)
            cls._add_subelement(cp, 'LowerLimit', value_low)
            cls._add_subelement(cp, 'UpperLimit', value_high)
            if gaussian is True:
                cls._add_subelement(cp, 'Mean', mean)
                cls._add_subelement(cp, 'SD', sd)
            cls._add_subelement(cp, 'ValuesPerSubCatchment', False)
            cls._add_subelement(cp, 'CatchmentOption', 'Random')

        # Vegetation zone parameters
        # TODO: Parameter values for multiple vegetation zones.
        # ---------------------------------------------------------------------
        vz_params = cls._add_subelement(
                root, 'MonteCarloVegetationZoneParameters')
        for param in veg_zone_params:
            # Get parameter information
            unit = cls._set_param_units(param, time_step='day')
            display_name = cls._set_param_disp_name(param)
            value_low = veg_zone_params[param]['LowerLimit']
            value_high = veg_zone_params[param]['UpperLimit']
            if gaussian is True:
                mean = veg_zone_params[param]['Mean']
                sd = veg_zone_params[param]['SD']
            # Append information to the XML tree
            vzp = cls._add_subelement(
                    vz_params, 'MonteCarloVegetationZoneParameter')
            cls._add_subelement(vzp, 'Name', param)
            cls._add_subelement(vzp, 'DisplayName', display_name)
            cls._add_subelement(vzp, 'Unit', unit)
            cls._add_subelement(vzp, 'LowerLimit', value_low)
            cls._add_subelement(vzp, 'UpperLimit', value_high)
            if gaussian is True:
                cls._add_subelement(vzp, 'Mean', mean)
                cls._add_subelement(vzp, 'SD', sd)
            cls._add_subelement(vzp, 'ValuesPerSubCatchment', False)
            cls._add_subelement(vzp, 'ValuesPerVegetationZone', False)
            cls._add_subelement(vzp, 'CatchmentOption', 'Random')
            cls._add_subelement(vzp, 'VegetationZoneOption', 'Random')

        # Monte Carlo parameters
        # ---------------------------------------------------------------------
        cls._add_subelement(root, 'MultiPeriods', multi_periods)
        if multi_periods is False:
            p = cls._add_subelement(root, 'Periods')
        elif multi_periods is True and periods is None:
            raise ValueError('Periods need to be specified.')
        else:
            for element in periods:
                cls._add_subelement(p, 'Period', element)
        cls._add_subelement(root, 'NumberOfRuns', runs)
        cls._add_subelement(root, 'SaveRunsOption', save_runs)
        cls._add_subelement(root, 'ObjFunctionUsage', 'UseOutlet')
        cls._add_subelement(root, 'MinReffValue', min_reff_val)
        cls._add_subelement(root, 'Gaussian', gaussian)
        ofws = cls._add_subelement(root, 'ObjFunctionWeights')
        if save_runs in ['SaveOnlyIf', 'Save100Best']:
            for name, weight in obj_functions.items():
                ofw = cls._add_subelement(ofws, 'ObjFunctionWeight')
                of = cls._add_subelement(ofw, 'ObjFunction')
                cls._add_subelement(of, 'Name', name)
                cls._add_subelement(of, 'Index', -1)
                cls._add_subelement(ofw, 'Weight', weight)
        # TODO: Set weights for different subcatchments.
        cls._add_subelement(root, 'SubCatchmentWeights')

        # Write out the xml file
        # ---------------------------------------------------------------------
        cls._write(root, filename)

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

    @classmethod
    def catchment_settings(
            cls, data, model_variant='Basic', filename='Clarea.xml',
            lakes=None, areas=None):
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
        model_variant : {'Basic', 'Aspect', 'Glacier'}, optional
            Name of the model variant, default is 'Basic'.
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
                'Catchment', nsmap={'xsi': cls.XSI, 'xsd': cls.XSD})

        # Elevation zones
        # ---------------------------------------------------------------------
        cls._add_subelement(
                root, 'VegetationZoneCount', vegetation_zone_count)
        elevz = cls._add_subelement(root, 'ElevationZoneHeight')
        for value in data.index:
            cls._add_subelement(elevz, 'double', int(value))

        # Fraction areas
        # ---------------------------------------------------------------------
        for scc in range(1, sub_catchment_count+1):
            scc = str(scc)
            subc = cls._add_subelement(root, 'SubCatchment')
            for vzc in range(1, vegetation_zone_count+1):
                vzc = str(vzc)
                vegz = cls._add_subelement(subc, 'VegetationZone')
                if model_variant == 'Basic':
                    for value in data['Area_' + scc + '_' + vzc]:
                        evu = cls._add_evu_element(
                                vegz, model_variant=model_variant)
                        cls._add_subelement(evu, 'Area', value)
                elif model_variant == 'Aspect':
                    for idx in range(len(data.index)):
                        idx = str(idx)
                        evu = cls._add_evu_element(
                                vegz, model_variant=model_variant)
                        for apc in ['n', 's', 'ew']:
                            value = cls._get_area_value(
                                    data, idx, scc, vzc, apc)
                            cls._set_aspect_value(evu, apc, value)
                elif model_variant == 'Glacier':
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
                        evu = cls._add_evu_element(
                                vegz, model_variant=model_variant)
                        cls._add_subelement(evu, 'IsGlacier', is_glacier)
                        for apc in ['n', 's', 'ew']:
                            value = cls._get_area_value(
                                    data, idx, scc, vzc, apc, igc)
                            cls._set_aspect_value(evu, apc, value)
                else:
                    raise ValueError('Error: invalid model variant provided')

            # Lake parameters
            # -----------------------------------------------------------------
            n = str(scc)
            lake = cls._add_subelement(subc, 'Lake')
            if lakes is None:
                cls._add_subelement(lake, 'Area', 0)
                cls._add_subelement(lake, 'Elevation', 0)
                cls._add_subelement(lake, 'TT', 0)
                cls._add_subelement(lake, 'SFCF', 0)
            else:
                cls._add_subelement(lake, 'Area', lakes['Lake_' + n]['Area'])
                cls._add_subelement(
                        lake, 'Elevation', lakes['Lake_' + n]['Elevation'])
                cls._add_subelement(lake, 'TT', lakes['Lake_' + n]['TT'])
                cls._add_subelement(lake, 'SFCF', lakes['Lake_' + n]['SFCF'])

            # Absolute area
            # -----------------------------------------------------------------
            if areas is None:
                cls._add_subelement(subc, 'AbsoluteArea', 0)
            else:
                cls._add_subelement(subc, 'AbsoluteArea', areas['Area_' + n])

        # Write out the xml file
        # ---------------------------------------------------------------------
        cls._write(root, filename)
