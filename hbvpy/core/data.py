#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hbvpy.data
==========

**A package to pre-process the necessary data for HBV-light**

This package is intended to provide functions and methods to pre-process
all the types of input data necessary for running HBV-light in Swiss
catchments. More specifically, it allows to process the following
data types (products):

* NetCDF :
    TabsD, TmaxD, TminD, RhiresD, RdisaggH (all from MeteoSwiss).
* Raster :
    SWE (SLF), MOD10A1 (MODIS), DEM (swisstopo).
* Shape :
    Basin shape (FOEN).
* Other :
    runoff (FOEN)

.. author:: Marc Girons Lopez

"""

import os
import ssl
import glob
import shutil
import numpy as np
import pandas as pd
import datetime as dt
from io import BytesIO
from pyproj import Proj
from zipfile import ZipFile
from netCDF4 import Dataset
from osgeo import gdal, ogr, osr
from urllib.request import urlopen
from scipy.stats import linregress

from . import HBVconfig
from hbvpy.ThirdParty import ncdump


gdal.UseExceptions()


__all__ = [
        'BasinShape', 'Evap', 'DEM', 'HBVdata', 'MOD10A1', 'RdisaggH',
        'RhiresD', 'Runoff', 'SWE', 'TabsD', 'TmaxD', 'TminD']


def remove_file(filename):
    """
    Iteratively remove a file if a PermissionError is encountered.

    Parameters
    ----------
    filename : str
        Path and file name of the file to be removed.

    """
    for retry in range(100):
        try:
            os.remove(filename)
            break
        except PermissionError:
            continue


class NetCDF(object):
    """
    Methods to work with MeteoSwiss NetCDF data files.

    Attributes
    ----------
    filename : str
        Path and file name of the NetCDF file object.

    """
    # Variables
    LON = None   # Name of the longitude variable
    LAT = None   # Name of the latitude variable
    DATA = None  # Name of the data variable

    # Projection
    PROJ4 = None  # Proj4 format

    # Data resolution
    RES = None

    def __init__(self, filename):

        self.fn = filename

        # HACK: Correct MeteoSwiss datasets prior to 2014.
        self._check_time_var()

    def _check_time_var(self):
        """
        Check the time variable name of a NetCDF dataset.

        In 2014 MeteoSwiss changed the time variable name from 'time' to
        'REFERENCE_TS'. This function detects which variable name is used in
        the NetCDF object and sets it as default for the NetCDF instance.

        """
        ds = Dataset(self.fn, 'r')

        if 'time' in ds.variables:
            self.TIME = 'time'

        elif 'REFERENCE_TS' in ds.variables:
            self.TIME = 'REFERENCE_TS'

        else:
            raise ValueError('The NetCDF file does not have a '
                             'recognised time variable.')

        ds.close()

    def ncdump(self, verb=True):
        """
        Retrieve the dimensions, variables, and attributes of a NetCDf dataset.

        This method calls the 'ncdump' function developed by Crhis Slocum
        from Colorado State University.

        Parameters
        ----------
        verb : bool
            Select whether to print the NetCDF dimensions, variables, and
            attributes, default is True.

        Returns
        -------
        nc_attrs : list
            List of the global attributes of the NetCDF file.
        nc_dims : list
            List of the dimensions of the NetCDF file.
        nc_vars : list
            List of the variables of the NetCDF file.

        """
        ds = Dataset(self.fn, 'r')

        nc_attrs, nc_dims, nc_vars = ncdump(ds, verb=verb)

        ds.close()

        return nc_attrs, nc_dims, nc_vars

    def load(self):
        """
        Load a NetCDF dataset.

        Returns
        -------
        data : Numpy array
            Array containing the gridded NetCDF data.

        """
        ds = Dataset(self.fn, 'r')

        data = ds.variables[self.DATA][:]

        # Set "_FillValue" (no data value) to NaN
        if '_FillValue' in ds.variables[self.DATA].ncattrs():
            no_data = ds.variables[self.DATA].getncattr('_FillValue')
            data[data == no_data] = np.nan

        elif 'missing_value' in ds.variables[self.DATA].ncattrs():
            no_data = ds.variables[self.DATA].getncattr('missing_value')
            data[data == no_data] = np.nan

        else:
            pass

        # Load the latitude variable to check for inverted array.
        lat = ds.variables[self.LAT][:]

        if lat[-1] > lat[0]:
            # HACK: Correct MeteoSwiss datasets prior to 2014.
            # Flip the latitude coordinate if the latitude variable is
            # inverted. This is the case for MeteoSwiss datasets before 2014.

            if isinstance(data, np.ma.MaskedArray):
                # HACK: Prevent FutureWarning
                data.unshare_mask()

            for i in range(len(ds.variables[self.TIME][:])):
                # Loop over the time variable and flip the latitude coordinate.
                data[i] = np.flipud(data[i])

        ds.close()

        return data

    def geotransform(self):
        """
        Get the geotransform information of the NetCDF dataset using the
        default format of the GDAL library.

        Format: (min(lon), res(lon), ??, max(lat), ??, -res(lat))

        Returns
        -------
        tuple
            Tuple containing the geotransform information.

        """
        ds = Dataset(self.fn, 'r')

        # Get the relevant limits of the netCDF dataset
        # (smallest longitude and larges latitude).
        xmin = min(ds.variables[self.LON])
        ymax = max(ds.variables[self.LAT])

        ds.close()

        return (xmin, self.RES, 0.0, ymax, 0.0, np.negative(self.RES))

    def meshgrid(self):
        """
        Get the coordinate meshgrid of the NetCDF dataset.

        Returns
        -------
        xx : Numpy array
            Array of longitude coordinates with the shape of the dataset.
        yy : Numpy array
            Array of latitude coordinates with the shape of the dataset.

        """
        ds = Dataset(self.fn, 'r')

        lon = ds.variables[self.LON][:]
        lat = ds.variables[self.LAT][:]

        # HACK: Correct MeteoSwiss datasets prior to 2014.
        if lat[-1] > lat[0]:
            lat = lat[::-1]

        xx, yy = np.meshgrid(lon, lat)

        ds.close()

        return xx, yy

    def lonlat_meshgrid(self):
        """
        Get the coordinate meshgrid of the NetCDF
        dataset in lon/lat (deg) units.

        Returns
        -------
        lon : Numpy array
            Array of longitude coordinates with the shape of the dataset.
        lat : Numpy array
            Array of latitude coordinates with the shape of the dataset.

        """
        xx, yy = self.meshgrid()

        ds = Dataset(self.nc_fn, 'r')

        if 'degrees' in ds.variables[self.LON].units:
            # Do nothing if the coordinates of the dataset
            # are already in degree units
            lon = xx
            lat = yy

        else:
            p = Proj(self.PROJ4)
            lon, lat = p(xx, yy, inverse=True)

        ds.close()

        return lon, lat

    def datenum_range(self):
        """
        Get the range of date numbers of the NetCDF dataset.

        Returns
        -------
        dates : Numpy array
            Array of datenums of the NetCDF file.

        """
        ds = Dataset(self.fn, 'r')

        dates = ds.variables[self.TIME][:]

        ds.close()

        return dates

    def date(self, datenum):
        """
        Obtain the date of a given date number of the NetCDF dataset.

        Parameters
        ----------
        datenum : int
            Date number to convert to datetime.

        Returns
        -------
        Datetime object
            Datetime object containing the corresponding date
            to the given datenum.

        Raises
        ------
        ValueError
            If the time step is not recognised.

        """
        ds = Dataset(self.fn, 'r')

        time_units = ds.variables[self.TIME].units

        ds.close()

        # HACK: Correct MeteoSwiss datasets prior to 2014.
        try:
            # After 2014 MeteoSwiss changed the format of the reference date
            # for daily gridded datasets, excluding sub-daily time units.
            orig = dt.datetime.strptime(time_units[-10:], '%Y-%m-%d')

        except ValueError:
            orig = dt.datetime.strptime(time_units[-19:], '%Y-%m-%d %H:%M:%S')

        if 'hours' in time_units:
            d = dt.timedelta(hours=datenum)

        elif 'days' in time_units:
            d = dt.timedelta(days=datenum)

        else:
            raise ValueError('Error: Time step not recognised!')

        return orig + d

    def mask(self, shp_fn, touch_all=False):
        """
        Mask the NetCDF dataset using a shapefile.

        Parameters
        ----------
        shp_fn : str
            Path and file name of the shapefile.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        masked_data : Numpy array
            Array containing the masked NetCDF data.

        """
        # Load the NetCDF file and extract the necessary information
        data = self.load()
        ntimes, nrows, ncols = np.shape(data)
        src_gt = self.geotransform()
        proj = self.PROJ4

        # Mask eventual invalid values (e.g. NaN) in the data array.
        data = np.ma.masked_invalid(data)

        # Update the geotransform information so xmin and ymax correspond
        # to the edge of the cells (Gdal) instead of the mid-point (NetCDF).
        dst_gt = (src_gt[0] - src_gt[1] * 0.5, src_gt[1], src_gt[2],
                  src_gt[3] - src_gt[5] * 0.5, src_gt[4], src_gt[5])

        # Rasterise the shapefile
        shp = Shape(shp_fn)
        basin = shp.rasterise(nrows, ncols, dst_gt, proj, touch_all=touch_all)

        # Invert the values of the basin array so it can be used as a mask.
        basin_mask = np.logical_not(basin)

        # Mask the NetCDF dataset using the rasterised shapefile
        masked_data = np.empty_like(data)
        for d in range(ntimes):
            masked_data[d] = np.ma.array(data[d], mask=basin_mask)

        return masked_data

    def average(
            self, shp_fn=None, date_list=None, value_list=None,
            start=None, end=None, touch_all=False):
        """
        Calculate the average value for each time step.

        Parameters
        ----------
        shp_fn : str, optional
            Path and file name of the masking shapefile, default is None.
        date_list : list, optional
            List containing datetime objects, default is None.
        value_list : list, optional
            List containing average values, default is None.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        date_list : list
            List containing datetime objects.
        value_list : list
            List containing average values.

        """
        if date_list is None:
            date_list = []

        if value_list is None:
            value_list = []

        # Get the date number array of the file
        datenums = self.datenum_range()

        if shp_fn is None:
            # Load the data.
            data = self.load()

        else:
            # Load the data and mask it with the shapefile.
            data = self.mask(shp_fn, touch_all=touch_all)

        # Get the period in datetime format
        if start is not None:
            start_date = dt.datetime.strptime(start, '%Y-%m-%d')

        if end is not None:
            end_date = dt.datetime.strptime(end, '%Y-%m-%d')

        # Iterate over the dates and average the data
        for d, datenum in enumerate(datenums):
            # Get the date from the date number
            date = self.date(datenum)

            if start is not None:
                # Continue to the next date if the date is
                # before the start of the period.
                if date < start_date:
                    continue

            if end is not None:
                # Break the loop if the date is after the end of the period.
                if date > end_date:
                    break

            # Calculate the average value for the catchment
            if data[d].mask.all():
                # No valid values in the array
                data_avg = np.nan
            else:
                data_avg = np.nanmean(data[d])

            # Append values to the lists
            date_list.append(date)
            value_list.append(data_avg)

        return date_list, value_list

    def reproject(self, src_fn, src_proj, dst_fn):
        """
        Reproject and resample a raster dataset to match the values of
        the NetCDF instance.

        Parameters
        ----------
        src_fn : str
            Path and file name of the raster dataset to reproject.
        src_proj : str
            Projection string (Proj4) of the raster dataset.
        dst_fn : str
            Path and filename of the reprojected raster dataset.

        """
        # GeoTransform and projecction information of the NetCDF dataset
        # used as a reference for the reprojection and resampling process.
        ndates, nrows, ncols = np.shape(self.load())
        ref_gt = self.geotransform()
        ref_proj = self.PROJ4

        # Update the information of the dataset edges so xmin and ymax
        # correspond to the edge of the cells (Gdal) instead of
        # the mid-point (NetCDF).
        xmin = ref_gt[0] - ref_gt[1] * 0.5
        xmax = ref_gt[0] + ref_gt[1] * ncols - ref_gt[1] * 0.5
        ymin = ref_gt[3] + ref_gt[5] * nrows - ref_gt[5] * 0.5
        ymax = ref_gt[3] - ref_gt[5] * 0.5

        # Reproject and resample the source raster
        gdal.Warp(dst_fn, src_fn, srcSRS=src_proj, dstSRS=ref_proj,
                  xRes=ref_gt[1], yRes=ref_gt[5], outputBoundsSRS=ref_proj,
                  outputBounds=(xmin, ymin, xmax, ymax))

    def mean_elev(self, dem_fn, shp_fn=None, touch_all=False):
        """
        Calculate the mean elevation of (a fraction of) the dataset.

        Parameters
        ----------
        dem_fn : str
            Path and file name of the DEM file to get the elevation data from.
        shp_fn : str, optional
            Path and file name of the masking shapefile, default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        float
            Mean elevation of the dataset (m a.s.l.).

        """
        # Resample and reproject the dem_fn to match the NetCDF data.
        dem_fn_tmp = os.getcwd() + '\\dem_tmp.tif'
        self.reproject(dem_fn, DEM.PROJ4, dem_fn_tmp)

        # Load the DEM file and set the correct projection
        dem_ds = DEM(dem_fn_tmp)
        dem_ds.PROJ4 = self.PROJ4

        if shp_fn is None:
            dem = dem_ds.load()

        else:
            dem = dem_ds.mask(shp_fn, touch_all=touch_all)

        # Remove the temporary DEM file
        remove_file(dem_fn_tmp)

        return np.nanmean(dem.compressed())

    @staticmethod
    def data_gradient(data, d, dem):
        """
        Get the gradient (slope) of data values with elevation.

        Parameters
        ----------
        data : NetCDF
            3D NETCDF array of data values with time.
        d : int
            NetCDF date index.
        dem : Numpy.array
            Numpy array of elevation values with the same size as "data".

        Returns
        -------
        slp : float
            Data gradient with elevation.

        """
        # Preallocate space
        elevs = []
        vals = []

        # Pair the cells of the data and dem arrays
        for i, elev in enumerate(dem.compressed()):
            elevs.append(elev)
            vals.append(data[d].compressed()[i])

        # Get the linear regression of the data-elevation pairs
        slp, intr, r, p, err = linregress(elevs, vals)

        return slp

    @classmethod
    def lapse_rate_value(cls, data, d, dem, step=100, method='absolute'):
        """
        Calculate the lapse rate of a given data array.

        Parameters
        ----------
        data : NetCDF
            3D NETCDF array of data values with time.
        d : int
            NetCDF date index.
        dem : Numpy.array
            Numpy array of elevation values with the same size as "data".
        step : float or int, optional
            Step to calculate the lapse rate for, default is 100.
            In the case of temperature lapse rate this would give deg C / 100m.
        method : {'absolute', 'relative'}, optional
            Choose whether to calculate the absolute or relative lapse rate,
            default is 'absolute'.

        Returns
        -------
        lapse_rate : float
            Lapse rate value.

        Raises
        ------
        ValueError
            If the method provided is not recognised.

        """
        # Get the data gradient
        slp = cls.data_gradient(data, d, dem)

        # Calculate the lapse rate
        if method == 'absolute':
            return slp * step

        elif method == 'relative':
            # Calculate the data mean over the area
            avg = np.nanmean(data[d, :, :])

            if avg == 0:
                return 0

            else:
                return (((avg + slp) / avg) - 1) * step

        else:
            raise ValueError('The provided method is not recognised.')

    def lapse_rate(
            self, dem_fn, shp_fn=None, date_list=None, value_list=None,
            step=100, method='absolute', start=None, end=None,
            touch_all=False):
        """
        Calculate the lapse rate of a NetCDF variable with elevation.

        Parameters
        ----------
        dem_fn : str
            Path and file name of the DEM file to get the elevation data from.
        shp_fn : str, optional
            Path and file name of the masking shapefile, default is None.
        date_list : list, optional
            List containing datetime objects, default is None.
        value_list : list, optional
            List containing lapse rate values, default is None.
        step : int or float, optional
            Elevation unit to calculate the lapse rate for, default is 100.
            In the case of temperature lapse rate this would give deg C / 100m.
        method : {'absolute', 'relative'}, optional
            Choose whether to calculate the absolute or relative lapse rate,
            default is 'absolute'.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        date_list : list
            List containing datetime objects.
        value_list : list
            List containing lapse rate values.

        Raises
        ------
        ValueError
            If the method provided is not recognised.

        """
        if date_list is None:
            date_list = []

        if value_list is None:
            value_list = []

        # Reproject and resample the DEM file
        dem_fn_tmp = os.getcwd() + '\\dem_tmp.tif'
        self.reproject(dem_fn, DEM.PROJ4, dem_fn_tmp)

        # Load the DEM file and set the correct projection
        dem_ds = DEM(dem_fn_tmp)
        dem_ds.PROJ4 = self.PROJ4

        # Get the datenum range of the NetCDF instance
        datenums = self.datenum_range()

        if shp_fn is None:
            # Load the necessary data
            data = self.load()
            dem = dem_ds.load()

        else:
            # Load and mask the necessary data
            data = self.mask(shp_fn, touch_all=touch_all)
            dem = dem_ds.mask(shp_fn, touch_all=touch_all)

        # Get the period in datetime format
        if start is not None:
            start_date = dt.datetime.strptime(start, '%Y-%m-%d')

        if end is not None:
            end_date = dt.datetime.strptime(end, '%Y-%m-%d')

        # Extract the relevant cells to calculate the lapse rate
        for d, datenum in enumerate(datenums):
            # Get the date from the date number
            date = self.date(datenum)

            if start is not None:
                # Continue to the next date if the date is
                # before the start of the period.
                if date < start_date:
                    continue

            if end is not None:
                # Break the loop if the date is after the end of the period.
                if date > end_date:
                    break

            date_list.append(date)
            value_list.append(self.lapse_rate_value(
                    data, d, dem, step=step, method=method))

        # Remove the temporary DEM file
        remove_file(dem_fn_tmp)

        return date_list, value_list


class Raster(object):
    """
    Methods to work with raster datasets.

    Attributes
    ----------
    filename : str
        Path and file name of the raster dataset.

    """
    PROJ4 = None

    NO_DATA = []

    def __init__(self, filename):

        self.fn = filename

    def load(self):
        """
        Load a raster dataset.

        Returns
        -------
        data : Numpy Array
            Array containing the raster data.

        """
        ds = gdal.Open(self.fn, gdal.GA_ReadOnly)
        data = ds.ReadAsArray().astype(float)

        ds = None

        # Set the values specified as no-data as np.NaN
        for value in self.NO_DATA:
            data[data == value] = np.nan

        return data

    def geotransform(self):
        """
        Get the geotransform information of the raster dataset.

        Returns
        -------
        gt : tuple
            Tuple containing the geotransform information.

        """
        ds = gdal.Open(self.fn, gdal.GA_ReadOnly)
        gt = ds.GetGeoTransform()

        ds = None

        return gt

    def meshgrid(self):
        """
        Obtain the coordinate meshgrid of the raster dataset.

        Returns
        -------
        xx : Numpy array
            Array of longitude coordinates with the shape of the dataset.
        yy : Numpy array
            Array of latitude coordinates with the shape of the dataset.

        """
        ds = gdal.Open(self.fn, gdal.GA_ReadOnly)
        gt = ds.GetGeoTransform()

        # get the edge coordinates and add half the resolution
        # to go to center coordinates
        xmin = gt[0]
        xmax = gt[0] + (gt[1] * ds.RasterXSize)
        ymin = gt[3] + (gt[5] * ds.RasterYSize)
        ymax = gt[3]

        lons = np.linspace(xmin, xmax, ds.RasterXSize)
        lats = np.linspace(ymax, ymin, ds.RasterYSize)

        ds = None

        xx, yy = np.meshgrid(lons, lats)

        return xx, yy

    def lonlat_meshgrid(self):
        """
        Get the coordinate meshgrid of the
        raster dataset in lon/lat (deg) units.

        Returns
        -------
        lon : Numpy array
            Array of longitude coordinates with the shape of the dataset.
        lat : Numpy array
            Array of latitude coordinates with the shape of the dataset.

        """
        p = Proj(self.PROJ4)

        xx, yy = self.meshgrid()

        lon, lat = p(xx, yy, inverse=True)

        return lon, lat

    def mask(self, shp_fn, touch_all=False):
        """
        Mask the raster dataset using a shapefile.

        Parameters
        ----------
        shp_fn : str
            Path and file name of the shapefile.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        masked_data : Numpy array
            Array containing the masked NetCDF data.

        """
        # Load the NetCDF file and extract the necessary information
        data = self.load()
        nrows, ncols = np.shape(data)
        gt = self.geotransform()
        proj = self.PROJ4

        # Mask eventual invalid values (e.g. NaN) in the data array.
        data = np.ma.masked_invalid(data)

        # Rasterise the shapefile
        shp = Shape(shp_fn)
        array = shp.rasterise(nrows, ncols, gt, proj, touch_all=touch_all)

        # Invert the values of the basin array to use it as a mask.
        data_mask = np.logical_not(array)

        # Mask the raster dataset using the rasterised shapefile
        masked_data = np.empty_like(data)
        masked_data = np.ma.array(data, mask=data_mask)

        return masked_data

    def average(self, shp_fn=None, touch_all=False):
        """
        Calculate the average value of the raster dataset.

        Parameters
        ----------
        shp_fn : str, optional
            Pand and file name of the masking shapefile, default: None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        float
            Average value of the raster dataset.

        """
        if shp_fn is None:
            data = self.load()
        else:
            data = self.mask(shp_fn, touch_all=touch_all)

        if data.mask.all():
            return np.nan
        else:
            return np.nanmean(data)

    def reproject(self, src_fn, src_proj, dst_fn):
        """
        Reproject and resample a raster dataset to match the values of
        the raster instance.

        Parameters
        ----------
        src_fn : str
            Path and file name of the raster dataset to reproject.
        src_proj : str
            Projection string (Proj4) of the raster dataset.
        dst_fn : str
            Path and filename of the reprojected raster dataset.

        """
        # Load the reference raster GeoTransform and projection
        ref_ds = gdal.Open(self.fn, gdal.GA_ReadOnly)
        gt = ref_ds.GetGeoTransform()
        ref_proj = self.PROJ4

        # Get the edges of the reference dataset
        xmin = gt[0]
        xmax = gt[0] + gt[1] * ref_ds.RasterXSize
        ymin = gt[3] + gt[5] * ref_ds.RasterYSize
        ymax = gt[3]

        # Reproject and resample the source raster
        gdal.Warp(dst_fn, src_fn, srcSRS=src_proj, dstSRS=ref_proj,
                  xRes=gt[1], yRes=gt[5], outputBoundsSRS=ref_proj,
                  outputBounds=(xmin, ymin, xmax, ymax))

    def lapse_rate(
            self, dem_fn, shp_fn=None, step=100,
            method='absolute', touch_all=False):
        """
        Calculate the lapse rate of the raster dataset with elevation.

        Parameters
        ----------
        dem_fn : str
            Path and file name of the DEM file to get the elevation data from.
        shape_fn : str, optional
            Path and file name of the masking shapefile, default: None.
        step : int or float, optional
            Elevation unit to calculate the lapse rate for, default: 100
            In the case of temperature lapse rate this would give deg C / 100m
        method : {'absolute', 'relative'}, optional
            Choose whether to calculate the absolute or relative lapse rate,
            default is 'absolute'.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        float
            Lapse rate value.

        Raises
        ------
        ValueError
            If the method provided is not recognised.

        """
        # Reproject and resample the DEM file
        dem_fn_tmp = os.getcwd() + '\\dem_tmp.tif'
        self.reproject(dem_fn, DEM.PROJ4, dem_fn_tmp)

        # Load the DEM file and set the correct projection
        dem_ds = DEM(dem_fn_tmp)
        dem_ds.PROJ4 = self.PROJ4

        if shp_fn is None:
            # Load the necessary data
            data = self.load()
            dem = dem_ds.load()

        else:
            # Load and mask the necessary data
            data = self.mask(shp_fn, touch_all=touch_all)
            dem = dem_ds.mask(shp_fn, touch_all=touch_all)

        # Extract the relevant cells to calculate the lapse rate
        elevs = []
        vals = []
        for i, elev in enumerate(dem.compressed()):
            elevs.append(elev)
            vals.append(data.compressed()[i])

        # The lapse rate is the slope of the linear regression
        slp, intr, r, p, err = linregress(elevs, vals)

        # Remove the temporary DEM file
        os.remove(dem_fn_tmp)

        if method == 'absolute':
            return slp * step

        elif method == 'relative':
            avg = np.nanmean(data)
            if avg == 0:
                return 0
            else:
                return (((avg + slp) / avg) - 1) * step

        else:
            raise ValueError('The provided method is not recognised.')


class Shape(object):
    """
    Methods to work with shapefiles.

    Attributes
    ----------
    filename : str
        Path and file name of the shapefile.

    """
    def __init__(self, filename):

        self.fn = filename

    def rasterise(
            self, dst_nrows, dst_ncols, dst_gt,
            dst_proj, dst_fn=None, touch_all=False):
        """
        Convert a shapefile into a raster.

        If no output shapefile is specified the output is stored
        in memory.

        Parameters
        ----------
        dst_nrows : int
            Number of rows of the output raster.
        dst_ncols : int
            Number of columns of the output raster.
        dst_gt : tuple
            Geotransformation informaiton of the output raster.
        dst_proj : Proj4
            Projection of the output raster (Proj4)
        dst_fn : str, optional
            Path and filename of the output raster file (e.g. *.tif),
            default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        array : Numpy array
            Array containing the data of the output raster.

        """
        # Load the shapefile
        driver = ogr.GetDriverByName('ESRI Shapefile')
        src_ds = driver.Open(self.fn, gdal.GA_ReadOnly)
        src_lyr = src_ds.GetLayer()

        # Initialise the rasterfile
        if dst_fn is None:
            driver = gdal.GetDriverByName('MEM')
            dst_ds = driver.Create('', dst_ncols, dst_nrows, 1, gdal.GDT_Byte)

        else:
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                    dst_fn, dst_ncols, dst_nrows, 1, gdal.GDT_Byte)

        # Set the attributes of the rasterfile
        dst_rb = dst_ds.GetRasterBand(1)
        dst_rb.Fill(0)
        dst_rb.SetNoDataValue(0)
        dst_ds.SetGeoTransform(dst_gt)

        # Set the reference system of the rasterfile
        srs = osr.SpatialReference()
        srs.ImportFromProj4(dst_proj)
        dst_ds.SetProjection(srs.ExportToWkt())

        # Rasterise the shapefile
        if touch_all is True:
            gdal.RasterizeLayer(dst_ds, [1], src_lyr, burn_values=[1],
                                options=['ALL_TOUCHED=TRUE'])
        else:
            gdal.RasterizeLayer(dst_ds, [1], src_lyr, burn_values=[1])

        # Read the output as a Numpy array.
        array = dst_rb.ReadAsArray()

        # Close the data source and target
        del src_ds, dst_ds

        return array

    def lonlat(self):
        """
        Get the latitude and longitude of the centre of mass
        of the polgon shapefile.

        TODO: So far it only works for shapefiles with a single feature.

        Returns
        -------
        lon : float
            Longitude of the centre of mass of the polygon shapefile (deg).
        lat : float
            Latitude of the centre of mass of the polygon shapefile (deg).

        """
        # Load the shapefile
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.Open(self.fn, gdal.GA_ReadOnly)
        lyr = ds.GetLayer()

        # Get the geometry
        feature = lyr.GetNextFeature()
        geometry = feature.GetGeometryRef()

        # Get the projection of the shapefile
        src_ref = geometry.GetSpatialReference()

        # Get the output projection (WGS84)
        # TODO: Provide an option to choose the output coordinate system.
        # HACK: ImportFromEPSG is currently not working...
        dst_ref = osr.SpatialReference()
        epsg_4326 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        dst_ref.ImportFromProj4(epsg_4326)

        # Get the centre of mass of the polygon.
        centroid = geometry.Centroid()

        # Transform the coordinate system of the centroid geometry
        transform = osr.CoordinateTransformation(src_ref, dst_ref)
        centroid.Transform(transform)

        # Get the latitude of the centroid
        lon = centroid.GetX()
        lat = centroid.GetY()

        # Close the dataset
        del ds

        return lon, lat

    def area(self):
        """
        Calculate the area of a polygon shapefile.

        # NOTE: So far it only works for shapefiles with a single feature.

        Returns
        -------
        area : float
            Area of the polygon shapefile (in the projection units).

        """
        # Load the shapefile
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.Open(self.fn, gdal.GA_ReadOnly)
        lyr = ds.GetLayer()

        # Get the geometry
        feature = lyr.GetNextFeature()
        geometry = feature.GetGeometryRef()

        # Calculate the area of the basin
        area = geometry.GetArea()

        del ds

        return area


class TabsD(NetCDF):
    """
    Daily mean air temperature gridded dataset.
    Author: MeteoSwiss

    Attributes
    ----------
    filename : str
        Path and filename of the TabsD NetCDF file.

    """
    LON = 'lon'
    LAT = 'lat'
    DATA = 'TabsD'

    PROJ4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

    RES = 1.25 / 60  # Minutes to degrees

    def __init__(self, filename):

        super().__init__(filename)


class TmaxD(NetCDF):
    """
    Daily maximum air temperature gridded dataset.
    Author: MeteoSwiss

    Attributes
    ----------
    filename : str
        Path and filename of the TmaxD NetCDF file.

    """
    LON = 'lon'
    LAT = 'lat'
    DATA = 'TmaxD'

    PROJ4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

    RES = 1.25 / 60  # Minutes to degrees

    def __init__(self, filename):

        super().__init__(filename)


class TminD(NetCDF):
    """
    Daily minimum air temperature gridded dataset.
    Author: MeteoSwiss

    Attributes
    ----------
    filename : str
        Path and filename of the TminD NetCDF file.

    """
    LON = 'lon'
    LAT = 'lat'
    DATA = 'TminD'

    PROJ4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

    RES = 1.25 / 60  # Minutes to degrees

    def __init__(self, filename):

        super().__init__(filename)


class RhiresD(NetCDF):
    """
    Daily precipitation (final analysis) gridded dataset.
    Author: MeteoSwiss

    Attributes
    ----------
    filename : str
        Path and filename of the RhiresD NetCDF file.

    """
    LON = 'lon'
    LAT = 'lat'
    DATA = 'RhiresD'

    PROJ4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

    RES = 1.25 / 60  # Minutes to degrees

    def __init__(self, filename):

        super().__init__(filename)


class RdisaggH(NetCDF):
    """
    Hourly precipitation (experimental) gridded dataset.
    Author: MeteoSwiss

    Attributes
    ----------
    filename : str
        Path and filename of the RdisaggH NetCDF file.

    """
    LON = 'chx'
    LAT = 'chy'
    DATA = 'RdisaggH'

    PROJ4 = '+proj=somerc +lat_0=46.95240555555556 '\
            '+lon_0=7.439583333333333 +k_0=1 +x_0=600000 '\
            '+y_0=200000 +ellps=bessel +towgs84=674.374,'\
            '15.056,405.346,0,0,0,0 +units=m +no_defs'

    RES = 1000  # metres

    def __init__(self, filename):

        super().__init__(filename)


class SWE(Raster):
    """
    Daily Snow Water Equivalent (SWE) gridded dataset for Switzerland.
    Author: SLF

    Attributes
    ----------
    filename : str
        Path and filename of the SWE raster file.

    """
    PROJ4 = '+proj=somerc +lat_0=46.95240555555556 '\
            '+lon_0=7.439583333333333 +k_0=1 +x_0=600000 '\
            '+y_0=200000 +ellps=bessel +towgs84=674.374,'\
            '15.056,405.346,0,0,0,0 +units=m +no_defs'

    NO_DATA = [-9999.]

    def __init__(self, filename):

        super().__init__(filename)

    def date(self):
        """
        Obtain the date of the SWE dataset.

        Returns
        -------
        Datetime object
            Datetime object containing the date of the SWE dataset.

        """
        date = dt.datetime.strptime(self.fn[-23:-13], '%Y-%m-%d')

        return date - dt.timedelta(days=1)

    def elev_dist(self, dem_fn, shp_fn=None, step=100, touch_all=False):
        """
        Calculate the elevation distribution of snow water equivalent.

        Parameters
        ----------
        dem_fn : str
            Path and filename of the DEM file used to get the elevation data.
        shp_fn : str, optional
            Path and filename of the shapefile to use for masking the data,
            default is None.
        step : int or float, optional
            Elevation band width to perform the calculations, default is 100.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        vals : list
            List of average SWE values for each elevation band.

        """
        # Instatiate a DEM dataset object
        dem_ds = DEM(dem_fn)

        if shp_fn is None:
            swe = self.load()
            dem = dem_ds.load()

        else:
            swe = self.mask(shp_fn, touch_all=touch_all)
            dem = dem_ds.mask(shp_fn, touch_all=touch_all)

        # Get the elevation distribution of the catchment.
        hist, bin_edges = DEM.histogram(dem, width=step)
        names = DEM.bin_names(bin_edges, header_type='range')

        # Preallocate space to store the average snow cover fraction values.
        obs_swe = pd.DataFrame(columns=names, index=[self.date()])

        # HACK: Get rid of NaN values in the DEM dataset in order to be able
        # to extract the elevation bands later on.
        dem.unshare_mask()
        dem[np.isnan(dem)] = -9999

        # Loop over the elevation bands.
        for i, n in enumerate(hist):
            # return NaN if no cells in the elevation band
            if n == 0:
                obs_swe.loc[self.date(), names[i]] = np.nan

            else:
                # Create and elevation mask to filter the cells outside
                # of the elevation band of interest
                elev_m = np.ones_like(dem)
                elev_m[(dem >= bin_edges[i]) & (dem < bin_edges[i+1])] = 0

                # Mask the data and extract the mean value
                swe_m = np.ma.masked_array(data=swe, mask=elev_m)
                obs_swe.loc[self.date(), names[i]] = np.mean(swe_m)

        # Delete empty columns
        obs_swe = obs_swe.dropna(axis=1, how='all')

        # return vals
        return obs_swe


class MOD10A1(Raster):
    """
    MODIS snow cover gridded dataset.

    Attributes
    ----------
    filename : str
        Path and filename of the MOD10A1 file.

    """
    PROJ4 = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 '\
            '+b=6371007.181 +units=m +no_defs'  # SR-ORG:6842

    NO_DATA = [200., 201., 211., 237., 239., 250., 254., 255.]
    # NO_DATA = [200., 201.]

    def __init__(self, filename):

        self.file = filename

        # NOTE: Only designed to return the 'NDSI_Snow_Cover' band.
        self.fn = 'HDF4_EOS:EOS_GRID:"' + filename + \
                  '":MOD_Grid_Snow_500m:NDSI_Snow_Cover'

        super().__init__(self.fn)

    def date(self):
        """
        Parse the date of the snow cover dataset.

        Returns
        -------
        Datetime object
            Date of the snow cover dataset.

        """
        return dt.datetime.strptime(self.file[-36:-29], '%Y%j')


class DEM(Raster):
    """
    Methods to work with Digital Elevation Models (DEMs) from swisstopo.

    Attributes
    ----------
    filename : str
        Path and filename of the DEM file.

    """
    PROJ4 = '+proj=somerc +lat_0=46.95240555555556 '\
            '+lon_0=7.439583333333333 +k_0=1 +x_0=600000 '\
            '+y_0=200000 +ellps=bessel +towgs84=674.374,'\
            '15.056,405.346,0,0,0,0 +units=m +no_defs'

    NO_DATA = [-9999.]

    def __init__(self, filename):

        super().__init__(filename)

    @staticmethod
    def histogram(dem_array, width=100):
        """
        Get the elevation histogram of the DEM dataset.

        Parameters
        ----------
        dem_array : Numpy array
            Array containing the DEM data.
        width : int or float, optional
            Width of the elevation bins to calculate the histogram for,
            default is 100.

        Returns
        -------
        hist : Numpy array
            The values of the histogram.
        bin_edges : Numpy array of type float
            Return the bin edges (length(hist)+1).

        """
        # Remove NaN cells
        dem = dem_array.compressed()

        # Set the bins with the given width and min/max array values
        bin_vals = np.arange(np.floor(min(dem) / width) * width,
                             np.ceil(max(dem) / width) * width, width)

        # Calculate the histogram
        hist, bin_edges = np.histogram(a=dem, density=False, bins=bin_vals)

        return hist, bin_edges

    @staticmethod
    def bin_names(bin_edges, header_type='range'):
        """
        Generate a list of bin names for a DEM histogram.

        Parameters
        ----------
        bin_edges : array of dtype float
            Array containing the bin edges (ouput from numpy.histogram).
        header_type : {'range', 'mid-point'}, optional
            Type of information to display in the bin names, default is
            'range'.

        Returns
        -------
        names : list
            List of bin names.

        """
        names = []

        for i in range(len(bin_edges)-1):
            if header_type == 'range':
                names.append(str(int(bin_edges[i])) + '-' +
                             str(int(bin_edges[i+1])))

            elif header_type == 'mid-point':
                names.append(str(int((bin_edges[i] + bin_edges[i+1]) / 2)))

            else:
                raise ValueError('Header type not recognised')

        return names

    def elev_area_dist(self, shp_fn=None, step=100, touch_all=False):
        """
        Calculate the area distribution with elevation.

        # TODO: This method only supports a 'Basic' model type with
        one vegetation zone and one sub-catchment.

        Parameters
        ----------
        shp_fn : str, optional
            Path and filename of the shapefile to use for masking the data,
            default is None.
        step : int or float, optional
            Elevation band width to perform the calculations. default is 100.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        df : Pandas.DataFrame
            DataFrame containing the area percentage data for each elevation
            band.

        """
        if shp_fn is None:
            # Load the DEM file
            dem = self.load()

        else:
            # Load and mask the DEM file
            dem = self.mask(shp_fn, touch_all=touch_all)

        # Calculate the elevation histogram of the catchment
        hist, bin_edges = self.histogram(dem, width=step)

        elevs = self.bin_names(bin_edges, header_type='mid-point')
        areas = []

        for band_cells in hist:
            # Calculate the percentage of cells within the elevation band
            areas.append(band_cells / sum(hist))

        # Create a Pandas.DataFrame with the area percentages
        df = pd.DataFrame(data=areas, index=elevs, columns=['Area_1_1'])

        # HACK: Save only elevation zones with non-zero area percentages
        return df[df['Area_1_1'] != 0]


class Evap(object):
    """
    Methods to generate potential evaporation and evapotranspiration data.

    """
    # Constants
    SC = 0.0820  # Solar constant (MJ m-2 min-1),
    L = 2.26     # Latent heat flux (MJ Kg-1)
    RHO = 1000   # Density of water (Kg m-3)

    @classmethod
    def extraterrestrial_radiation(cls, lat, j):
        """
        Calculate the potential extraterrestrial solar radiation.

        Based on Eq. 21, 23, 24, 25 in:
        Allen et al. (1998) Crop evapotranspiration - Guidelines
        for computing crop water requirements - FAO Irrigation and
        drainage paper 56.

        Parameters
        ----------
        lat : float
            Latitude of the catchment (degrees).
        j : int
            Day of the year.

        Returns
        -------
        float
            Potential extraterrestrial solar radiation (kJ m-2 day-1).

        """

        # Transform latitude from degrees to radians
        lat_rad = lat * (np.pi / 180)

        # Calculate the solar declination
        dec = 0.409 * np.sin(((2 * np.pi) / 365) * j - 1.39)

        # Calculate the inverse relative distance Earth-Sun
        dr = 1 + 0.033 * np.cos(((2 * np.pi) / 365) * j)

        # Calculate the sunset hour angle
        sha = np.arccos(-np.tan(lat_rad) * np.tan(dec))

        # Calculate the extraterrestrial solar radiation (MJ m-2 day-1)
        return ((24 * 60) / np.pi) * cls.SC * dr * (
                sha * np.sin(lat_rad) * np.sin(dec) +
                np.cos(lat_rad) * np.cos(dec) * np.sin(sha))

    @classmethod
    def mean_monthly_radiation(cls, lat):
        """
        Calculate the mean monthly potential extraterrestrial radiation for
        an arbitrary year.

        Parameters
        ----------
        lat : float
            Latitude of the catchment (degrees).

        Returns
        -------
        Pandas.Series
            Pandas Series containing the mean monthly radiation values.

        """
        # Calculate the extraterrestrial radiation for each day of the year.
        js = np.arange(1, 366, 1)
        re = cls.extraterrestrial_radiation(lat, js)

        # Create an array of dates for a random year
        idx = np.arange(dt.datetime(2017, 1, 1), dt.datetime(2018, 1, 1),
                        dt.timedelta(days=1)).astype(dt.datetime)

        # Calculate the monthly mean
        ds = pd.Series(data=re, index=idx)

        return ds.groupby(ds.index.month).mean()

    @classmethod
    def potential_evaporation(cls, re_m, temp_m):
        """
        Calculate the mean monthly potential evaporation for a catchment.

        Based on Eq. 3 in:
        Oudin et al. 2005, Which potential evapotranspiration input for
        a lumped rainfall-runoff model? Part 2 --- Towards a simple and
        efficient potential evapotranspiration model for rainfall-runoff
        modelling. Journal of Hydrology, 303, p. 290-306.

        Parameters
        ----------
        re_m : float
            Mean monthly extraterrestrial solar radiation for a given month.
        temp_m : float
            Mean air temperature of a given month.

        Returns
        -------
        float
            Potential evapotranspiration (mm).

        """
        # Calculate the potential evaporation (m/day)
        pet = (re_m / (cls.L * cls.RHO)) * ((temp_m + 5) / 100)

        # If temperature is below -5 deg C pe is considered to be 0
        pet[temp_m <= -5] = 0

        # Transform the PET into mm
        return pet * 1000


class Runoff(object):
    """
    Methods to work with daily streamflow datasets from BAFU.

    Attributes
    ----------
    stn_code : int
        BAFU code of the streamflow station.

    """

    def __init__(self, filename):

        self.fn = filename

    def load(self):
        """
        Load streamflow data for a given BAFU hydrometric station.

        Returns
        -------
        Pandas.DataFrame
            DataFrame including streamflow data for each time step.

        """
        return pd.read_csv(
                self.fn, sep='-|;', skiprows=7, skipinitialspace=True,
                usecols=[1, 3], index_col=0, names=['Date', 'Q'], header=None,
                parse_dates=True, squeeze=True, infer_datetime_format=True,
                engine='python')

    def units(self):
        """
        Get the units of the stream runoff data.

        BAFU stream runoff files usually have a line specifying the runoff
        units. This method reads the runoff file line by line until it finds
        the relevant line. It then extracts and returns the runoff units.

        Returns
        -------
        str
            Units of the stream runoff data.

        """
        with open(self.fn, 'r') as f:
            lines = f.readlines()

        # Loop through the lines until the line containing 'Abfluss' is
        # reached (this is the line containing information about the units).
        # u_line = ''
        for line in lines:
            if 'Abfluss' in line:
                units_line = line
                break

        # Split the line into header ('Abfluss') and units
        header, units = units_line.split(sep=' ')

        # HACK: Return the units removing the new string command ("\n")
        return units[:-1]


class BasinShape(object):
    """
    Define the contributing BAFU partial subcatchments for a given stream
    gauge station and generate the corresponding polygon shapefile.

    Given a BAFU station code number the code downloads the shapefile
    all the stream gauging stations from BAFU and extracts the coordinates
    of the given station. It then downloads the shapefile with all the
    sub-basins in which Switzerland is divided and computes which catchments
    are upstream of the given station. The code finally returns a shapefile
    with a single polygon representing the basin area corresponding to the
    given station.

    Attributes
    ----------
    stn_code : int
        BAFU code of the desired hydrometric station.

    """
    # URL of the zip file containing the BAFU hydrometrical stations data
    STATIONS_URL = (
            'https://data.geo.admin.ch/'
            'ch.bafu.hydrologie-hydromessstationen/'
            'data.zip'
            )

    # URL of the zip file containing the BAFU subdivision of Switzerland in
    # partial hydrological catchments
    BASINS_URL = (
            'https://www.bafu.admin.ch/dam/bafu/de/'
            'dokumente/wasser/geodaten/'
            'einzugsgebietsgliederungschweizausgabe2015.zip.download.zip/'
            'einzugsgebietsgliederungschweizausgabe2015.zip'
            )

    def __init__(self, stn_code):

        self.code = stn_code

        self.path_tmp = os.getcwd() + '\\BAFU_data\\'

        if not os.path.exists(self.path_tmp):
            os.makedirs(self.path_tmp)

    def _download_data(self, zipurl):
        """
        Download a zipfile and extract its contents.

        Parameters
        ----------
        zipurl : str
            URL of the zip file to download.

        """
        # Download the zipped data and extract its contents to the temp folder
        context = ssl._create_unverified_context()
        with urlopen(zipurl, context=context) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(self.path_tmp)

        if os.path.exists(self.path_tmp + 'lhg_UBST.zip'):
            with ZipFile(self.path_tmp + 'lhg_UBST.zip') as zfile:
                zfile.extractall(self.path_tmp)
            os.remove(self.path_tmp + 'lhg_UBST.zip')

    def _get_station_coordinates(self):
        """
        Get the coordinates of a given station.

        Download the BAFU shapefile containing the location of all the BAFU
        hydrometric stations, and extract the coordinates of the desired
        station.

        Returns
        -------
        x : float
            Longitude coordinate of the selected BAFU station.
        y : float
            Latitude coordinate of the selected BAFU station.

        """
        # Download and extract the data on the FOEN hydrometric stations
        if not os.path.exists(self.path_tmp + 'lhg_UBST.shp'):
            self._download_data(self.STATIONS_URL)

        # Load the shapefile containing the data
        driver = ogr.GetDriverByName('ESRI Shapefile')
        src_ds = driver.Open(self.path_tmp + 'lhg_UBST.shp', 0)
        src_lyr = src_ds.GetLayer()

        # Filter the point feature representing the selected station
        query = 'EDV_NR4 = {}'.format(self.code)
        src_lyr.SetAttributeFilter(query)

        # Get the geometry of the feature
        feature = src_lyr.GetNextFeature()
        geometry = feature.GetGeometryRef()

        # Get the coordinates of the station from the feature's geometry
        x = geometry.GetX()
        y = geometry.GetY()

        # Close the data source
        del src_ds

        return x, y

    def _get_auxiliary_codes(self, x, y):
        """
        Get the auxiliary codes necessary to generate the upstream basin.

        Download the BAFU geodatabase containing all the partial sub-basins of
        Switzerland and load it. Loop over the partial sub-basins and find
        the one including the coordinates of the desired station. Finally,
        get the auxiliary codes (H1 and H2) for the selected partial sub-basin.

        Parameters
        ----------
        x : float
            Longitude coordinate of the selected BAFU station.
        y : float
            Latitude coordinate of the selected BAFU station.

        Returns
        -------
        h1 : float
            Auxiliary code to generate the basin polygon.
        h2 : float
            Auxiliary code to generate the basin polygon.

        """
        # Download and extract the data on the FOEN partial catchments
        if not os.path.exists(self.path_tmp + 'EZGG2015.gdb'):
            self._download_data(self.BASINS_URL)

        # Load the database in which the data is stored and extract the
        # shapefile where partial catchments are stored
        driver = ogr.GetDriverByName('OpenFileGDB')
        src_ds = driver.Open(self.path_tmp + 'EZGG2015.gdb', 0)
        src_lyr = src_ds.GetLayer('basisgeometrie')

        # Create a point feature using the coordinates of the station
        point = ogr.Geometry(ogr.wkbPoint)
        point.SetPoint(0, x, y)

        # Loop over the different partial catchments to locate the partial
        # subcatchment in which the hydrometric station is located and
        # extract the "H1" and "H2" auxiliary field codes.
        for feature in src_lyr:
            polygon = feature.GetGeometryRef()
            if point.Within(polygon):
                h1 = feature.GetField('H1')
                h2 = feature.GetField('H2')

        # Close the data source
        del src_ds

        return h1, h2

    def _generate_shapefile(self, h1, h2, shp_fn):
        """
        Generate a basin polygon shapefile for a given FOEN hydrometric
        station code.

        Generate a polygon feature based on the auxiliary codes H1 and H2
        and save it as a new shapefile.

        Parameters
        ----------
        h1 : float
            Auxiliary code to generate the basin polygon.
        h2 : float
            Auxiliary code to generate the basin polygon.
        shp_fn : str
            Path and file name of the output shapefile.

        """
        # Load the database in which the data is stored and extract the
        # shapefile where partial catchments are stored
        driver = ogr.GetDriverByName('OpenFileGDB')
        src_ds = driver.Open(self.path_tmp + 'EZGG2015.gdb', 0)
        src_lyr = src_ds.GetLayer('basisgeometrie')
        src_proj = src_lyr.GetSpatialRef()

        # Create a new polygon geometry feature and merge all the partial
        # catchments the are upstream of the given hydrometric station
        # (as given by H1 and H2)
        union_polygon = ogr.Geometry(ogr.wkbPolygon)
        for feature in src_lyr:
            if feature.GetField('H1') >= h1 and feature.GetField('H1') < h2:
                geometry = feature.GetGeometryRef()
                union_polygon = union_polygon.Union(geometry)

        # Create a new ogr driver to write the resulting shapefile
        driver = ogr.GetDriverByName("ESRI Shapefile")

        # Check if a file with the same name exists, and if so delete it
        if os.path.exists(shp_fn):
            driver.DeleteDataSource(shp_fn)

        # Create a new layer to store the output data
        dst_ds = driver.CreateDataSource(shp_fn)
        dst_lyr = dst_ds.CreateLayer(
            'basin_' + str(self.code), src_proj, geom_type=ogr.wkbPolygon)

        # Assign an id field to the new layer
        field_id = ogr.FieldDefn("id", ogr.OFTInteger)
        dst_lyr.CreateField(field_id)

        # Write the output data to the newly created layer
        feature_defn = dst_lyr.GetLayerDefn()
        feature = ogr.Feature(feature_defn)
        feature.SetGeometry(union_polygon)
        feature.SetField("id", self.code)
        dst_lyr.CreateFeature(feature)

        # Close the data source and target
        del src_ds, dst_ds

    def generate(self, shp_fn, keep_files=False):
        """
        """
        # Get the coordinates of the selected station
        x, y = self._get_station_coordinates()

        # Get the corresponding auxiliary codes
        h1, h2 = self._get_auxiliary_codes(x, y)

        # Generate the basin shapefile
        self._generate_shapefile(h1, h2, shp_fn)

        if keep_files is False:
            # Clean the temporary files
            shutil.rmtree(self.path_tmp)

    def station_name(self, keep_files=False):
        """
        """
        # Download and extract the data on the FOEN hydrometric stations
        if not os.path.exists(self.path_tmp + 'lhg_UBST.shp'):
            self._download_data(self.STATIONS_URL)

        # Load the shapefile containing the data
        driver = ogr.GetDriverByName('ESRI Shapefile')
        src_ds = driver.Open(self.path_tmp + 'lhg_UBST.shp', 0)
        src_lyr = src_ds.GetLayer()

        # Filter the point feature representing the selected station
        query = 'EDV_NR4 = {}'.format(self.code)
        src_lyr.SetAttributeFilter(query)

        # Get the feature and retrieve the name of the station
        feature = src_lyr.GetNextFeature()
        name = feature['lhg_name']

        # Close the data source and clean the temporary files
        del src_ds

        if keep_files is False:
            # Clean the temporary files
            shutil.rmtree(self.path_tmp)

        return name


class HBVdata(object):
    """
    Generate an HBV-light Catchment folder structure and data files.

    # TODO: Provide the possibility to choose between daily and hourly steps
    (so far only daily time steps are used).

    # TODO: Provide the possibility to choose the input precipitation and
    temperature data products (so far only RhiresD and TabsD are used).

    Attributes
    ----------
    bsn_dir : str
        Basin directory.

    """
    def __init__(self, bsn_dir):

        self.bsn_dir = bsn_dir

        # The data files are stored in the 'Data' subfolder.
        self.data_dir = bsn_dir + '\\Data\\'

        # Create the folder structure if it doesn't exist.
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _write_txt(self, df, filename, idx=False, header=None):
        """
        Write a Pandas.DataFrame as a text file.

        Parameters
        ----------
        df : Pandas.DataFrame
            Pandas DataFrame containing the data to write as a text file.
        filename : str
            Name of the text file.
        idx : bool, optional
            Select whether the file should have an index column, default is
            False.
        header : str, optional
            File header, default is None.

        """
        with open(self.data_dir + filename, 'w') as txt_file:
            if header is not None:
                # Write an additional header line
                txt_file.write(str(header) + '\n')

            if idx is False:
                # Save the file without an index.
                df.to_csv(txt_file, sep='\t', header=True,
                          index=False, na_rep=-9999)

            else:
                # Save the file with an index.
                df.to_csv(txt_file, sep='\t', index_label='Date',
                          date_format='%Y%m%d', na_rep=-9999, header=True)

    @staticmethod
    def _parse_precip(precip_dir, shp_fn=None, output='average',
                      dem_fn=None, start=None, end=None, touch_all=False):
        """
        Parse the average precipitation data for the catchment.

        Parameters
        ----------
        precip_dir : str
            Directory where the precipitation data is stored.
        shp_fn : str, optional
            Path and filename of the shapefile defining the catchment boundary,
            default is None.
        output : {'average', 'lapse_rate'}, optional
            Operation to perform to the precipitation data,
            default is 'average'.
        dem_fn : str, optional
            Path and filename of the DEM file to get the elevation data from.
            Needed if op == 'TCALT', default is None.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        Pandas.Series
            Pandas Series structure containing the precipitation data.

        Raises
        ------
        ValueError
            If the output format is not provided.

        """
        # Preallocate space to store dates and precipitation data.
        dates = []
        vals = []

        for file in glob.glob(precip_dir + '*.nc'):
            # Loop over the precipitation NetCDF files in the directory.

            # Create an instance of the RhiresD class.
            precip = RhiresD(file)

            if output == 'average':
                # Get the average precipitation over the catchment.
                dates, vals = precip.average(
                        shp_fn=shp_fn, date_list=dates, value_list=vals,
                        start=start, end=end, touch_all=touch_all)
                series_name = 'P'

            elif output == 'lapse_rate':
                # Calculate the temperature lapse rate over the catchment.
                if dem_fn is None:
                    raise ValueError('DEM file for lapse rate '
                                     'calculation not found.')
                dates, vals = precip.lapse_rate(
                        dem_fn, shp_fn=shp_fn, date_list=dates,
                        value_list=vals, method='relative', start=start,
                        end=end, touch_all=touch_all)
                series_name = 'PCALT'

            else:
                raise ValueError('Output format is not recognised.')

        # Return a Pandas.Series object containing the precipitation data.
        return pd.Series(data=vals, index=dates, name=series_name)

    @staticmethod
    def _parse_temp(temp_dir, shp_fn=None, output='average',
                    dem_fn=None, start=None, end=None, touch_all=False):
        """
        Parse the temperature data for the catchment.

        Parameters
        ----------
        temp_dir : str
            Directory where the temperature data is stored.
        shp_fn : str, optional
            Path and filename of the shapefile defining the catchment boundary,
            default is None.
        output : {'average', 'lapse_rate'}, optional
            Operation to perform to the temperature data, default is 'average'.
        dem_fn : str, optional
            Path and filename of the DEM file to get the elevation data from.
            Needed if op == 'TCALT', default is None.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        Returns
        -------
        Pandas.Series
            Pandas Series structure containing the temperature data.

        Raises
        ------
        ValueError
            If the output format is not provided.

        """
        # Preallocate space to store dates and tempeature data.
        dates = []
        vals = []

        for file in glob.glob(temp_dir + '*.nc'):
            # Loop over the temperature NetCDF files in the directory.

            # Create an instance of the TabsD class.
            temp = TabsD(file)

            if output == 'average':
                # Average temperature over the catchment.
                dates, vals = temp.average(
                        shp_fn=shp_fn, date_list=dates, value_list=vals,
                        start=start, end=end, touch_all=touch_all)
                series_name = 'T'

            elif output == 'lapse_rate':
                # Calculate the temperature lapse rate over the catchment.
                if dem_fn is None:
                    raise ValueError('DEM file for lapse rate '
                                     'calculation not found.')
                dates, vals = temp.lapse_rate(
                        dem_fn, shp_fn=shp_fn, date_list=dates,
                        value_list=vals, method='absolute', start=start,
                        end=end, touch_all=touch_all)
                series_name = 'TCALT'

            else:
                raise ValueError('Output format is not recognised.')

        # Return a Pandas.Series object containing the temperature data
        return pd.Series(data=vals, index=dates, name=series_name)

    @staticmethod
    def _parse_q(q_dir, shp_fn, stn_code, start=None, end=None):
        """
        Parse the stream runoff data for the catchment outlet.

        Parameters
        ----------
        q_dir : str
            Directory where the stream runoff data is stored.
        shp_fn : str
            Path and filename of the shapefile defining the catchment boundary.
        stn_code : int
            Identification code of the BAFU hydrometric station defining
            the catchment outlet.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.

        Returns
        -------
        Pandas.Series
            Pandas Series structure containing the stream runoff data
            in mm day-1.

        """
        # Get the appropriate data file given the BAFU station code
        fn = q_dir + 'Q_' + str(stn_code) + '_Tagesmittel.asc'

        if not os.path.exists(fn):
            raise ValueError('No streamflow data is available for the '
                             'given BAFU station.')

        # Load the runoff data
        q = Runoff(fn).load()
        # Get the runoff units to perform the calculations (m3 s-1 or l s-1)
        u = Runoff(fn).units()
        # Calculate the catchment area (m2)
        area = Shape(shp_fn).area()

        # Re-index the data
        if start is not None or end is not None:
            date_index = pd.date_range(start=start, end=end, freq='D')
            q = q.reindex(date_index)

        # Transform the runoff units to mm and return the resulting data.
        if u == 'l/s':
            return (q / area) * 3600 * 24

        else:  # m3 s-1
            return (q / area) * 1000 * 3600 * 24

    def generate_ptq(
            self, precip_dir, temp_dir, q_dir, shp_fn, stn_code, step=100,
            start=None, end=None, touch_all=False, filename='PTQ.txt'):
        """
        Generate an HBV-light PTQ.txt file.

        Parameters
        ----------
        precip_dir : str
            Directory where the precipitation data is stored.
        temp_dir : str
            Directory where the temperature data is stored.
        q_dir : str
            Directory where the stream runoff data is stored.
        shp_fn : str
            Path and filename of the basin shapefile delimiting the catchment.
        stn_code : int
            Identification code of the BAFU hydrometric station defining
            the catchment.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.
        filename : str
            Name of the PTQ file, default is 'PTQ.txt'.

        Returns
        -------
        ptq : Pandas.DataFrame
            Pandas Dataframe containing the ptq data for the catchment.

        """
        print('...' + filename)

        # Parse the precipitation, temperature, and stream runoff data
        p = self._parse_precip(
                precip_dir, shp_fn=shp_fn, output='average',
                start=start, end=end, touch_all=touch_all)
        # Interpolate eventual missing data
        p.interpolate(method='cubic', inplace=True)
        # HACK: Filter potential negative precip values
        p[p < 0] = 0

        t = self._parse_temp(
                temp_dir, shp_fn=shp_fn, output='average',
                start=start, end=end, touch_all=touch_all)
        # Interpolate eventual missing data
        t.interpolate(method='cubic', inplace=True)

        q = self._parse_q(q_dir, shp_fn, stn_code, start=start, end=end)

        # Merge the data into a Pandas.DataFrame and save it as a text file.
        ptq = pd.concat([p, t, q], axis=1)

        # Re-index the data
        if start is not None or end is not None:
            date_index = pd.date_range(start=start, end=end, freq='D')
            ptq = ptq.reindex(date_index)

        # Round the values.
        ptq = ptq.round(decimals=3)

        # Set the header of the file
        name = BasinShape(stn_code).station_name()
        header = str(stn_code) + ' - ' + name

        self._write_txt(ptq, filename, idx=True, header=header)

        return ptq

    def generate_ptcalt(
            self, precip_dir, temp_dir, dem_fn, shp_fn, stn_code,
            start=None, end=None, pcalt=True, tcalt=True, touch_all=False,
            filename='PTCALT.txt'):
        """
        Generate an HBV-light PTCALT.txt file.

        Parameters
        ----------
        precip_dir : str
            Directory where the precipitation data is stored.
        temp_dir : str
            Directory where the temperature data is stored.
        dem_fn : str
            Path and filename of the DEM file to get the elevation data from.
        shp_fn : str
            Path and filename of the basin shapefile delimiting the catchment.
        stn_code : int
            Identification code of the BAFU hydrometric station defining
            the catchment.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        pcalt : bool, optional
            Choose whether to include precipitation lapse rate time series,
            default is True.
        tcalt : bool, optional
            Choose whether to include temperature lapse rate time series,
            default is True.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.
        filename : str
            Name of the PTCALT file, default is 'PTCALT.txt'.

        Returns
        -------
        ptcalt : Pandas.Series
            Pandas Series containing the temperature lapse rate.

        """
        print('...' + filename)

        if pcalt is False and tcalt is False:
            # Return nothing if no variable is selected
            return None

        else:
            if pcalt is True and tcalt is False:
                # Parse the precipitation data
                ptcalt = self._parse_precip(
                        precip_dir, shp_fn=shp_fn, output='lapse_rate',
                        dem_fn=dem_fn, start=start, end=end,
                        touch_all=touch_all)
                # Convert fraction to percentage
                ptcalt = ptcalt * 100

            elif pcalt is False and tcalt is True:
                # Parse the temperature data
                ptcalt = self._parse_temp(
                        temp_dir, shp_fn=shp_fn, output='lapse_rate',
                        dem_fn=dem_fn, start=start, end=end,
                        touch_all=touch_all)
                # Reverse the temperature lapse rate (HBV-light convention)
                ptcalt = -ptcalt

            else:
                # Parse the precipitation and temperature data
                p_calt = self._parse_precip(
                        precip_dir, shp_fn=shp_fn, output='lapse_rate',
                        dem_fn=dem_fn, start=start, end=end,
                        touch_all=touch_all)
                # Convert fraction to percentage
                p_calt = p_calt * 100
                t_calt = self._parse_temp(
                        temp_dir, shp_fn=shp_fn, output='lapse_rate',
                        dem_fn=dem_fn, start=start, end=end,
                        touch_all=touch_all)
                # Reverse the temperature lapse rate (HBV-light convention)
                t_calt = -t_calt
                # Concatenate the precipitation and temperature series
                ptcalt = pd.concat([p_calt, t_calt], axis=1)

            # Re-index the data
            if start is not None or end is not None:
                date_index = pd.date_range(start=start, end=end, freq='D')
                ptcalt = ptcalt.reindex(date_index)

            # Round the number of decimals and save it as a text file.
            ptcalt = ptcalt.round(decimals=3)

            # Set the header of the file
            name = BasinShape(stn_code).station_name()
            header = str(stn_code) + ' - ' + name

            self._write_txt(ptcalt, filename, idx=True, header=header)

            return ptcalt

    def generate_snow_cover(
            self, sc_dir, shp_fn, start=None, end=None,
            touch_all=False, filename='SnowCover.txt'):
        """
        Generate an HBV-light SnowCover.txt file.

        Parameters
        ----------
        sc_dir : str
            Directory where the snow cover fraction data is stored.
        shp_fn : str
            Path and filename of the shapefile to use for masking the data.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.
        filename : str
            Name of the SnowCover file, default is 'SnowCover.txt'.

        Returns
        -------
        obs_sc : Pandas.DataFrame
            Pandas DataFrame containing the average snow cover fraction values
            for each elevation band and time step.

        """
        print('...' + filename)

        # Preallocate space to store dates and snow cover data.
        dates = []
        scs = []

        # Get the period in datetime format
        if start is not None:
            start_date = dt.datetime.strptime(start, '%Y-%m-%d')

        if end is not None:
            end_date = dt.datetime.strptime(end, '%Y-%m-%d')

        for file in glob.glob(sc_dir + '*.hdf'):
            # Loop over the snow cover hdf files in the directory.

            # Create an instance of the MOD10A1 class.
            mod = MOD10A1(file)

            # Get the date of the current file.
            date = mod.date()

            if start is not None:
                # Continue to the next file if the end date of the file is
                # before the start of the period.
                if date < start_date:
                    continue

            if end is not None:
                # Break the loop if the end date of the file is after
                # the end of the period.
                if date > end_date:
                    break

            # Calculate the average snow fraction over the catchment and
            # append the date and value to the preallocated lists.
            dates.append(date)
            scs.append(mod.average(shp_fn=shp_fn, touch_all=touch_all))

        # Store the data in a Pandas.Series object.
        snow_cover = pd.Series(data=scs, index=dates, name='SnowCover')

        # Re-index the data
        if start is not None or end is not None:
            date_index = pd.date_range(start=start, end=end, freq='D')
            snow_cover = snow_cover.reindex(date_index)

        # Round the values to 3 decimals.
        snow_cover = snow_cover.round(decimals=3)

        self._write_txt(snow_cover, filename, idx=True, header=None)

        return snow_cover

    def generate_swe(
            self, swe_dir, dem_fn, shp_fn, output='elev_dist', step=100,
            start=None, end=None, touch_all=False, filename='ObsSWE.txt'):
        """
        Generate an HBV-light ObsSWE.txt file.

        Parameters
        ----------
        swe_dir : str
            Directory where the snow water equivalent data is stored.
        dem_fn : str
            Path and filename of the DEM file used to get the elevation data.
        shp_fn : str
            Path and filename of the shapefile to use for masking the data.
        output : {'elev_dist', 'average'}
            Choose between calculating the average SWE value for each
            elevation zone ('elev_dist') or for the entire catchment
            ('average'), default is 'elev_dist'.
        step : int or float, optional
            Elevation band width to perform the calculations, default is 100.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.
        filename : str
            Name of the SWE file, default is 'ObsSWE.txt'.

        Returns
        -------
        obs_swe : Pandas.DataFrame
            Pandas DataFrame containing the average SWE values for each
            elevation band and time step.

        """
        print('...' + filename)

        # Initialise a Pandas.DataFrame object to store the ObsSWE data.
        obs_swe = pd.DataFrame()

        # Get the period in datetime format
        if start is not None:
            start_date = dt.datetime.strptime(start, '%Y-%m-%d')

        if end is not None:
            end_date = dt.datetime.strptime(end, '%Y-%m-%d')

        for file in glob.glob(swe_dir + '*.asc'):
            # Loop over the SWE asc files in the directory.

            # Create an instance of the SWE class.
            swe = SWE(file)

            # Get the date of the current file.
            date = swe.date()

            if start is not None:
                # Continue to the next file if the end date of the file is
                # before the start of the period.
                if date < start_date:
                    continue

            if end is not None:
                # Break the loop if the end date of the file is after
                # the end of the period.
                if date > end_date:
                    break

            # Calculate the elevation distribution of SWE and append the date
            # and value to the preallocated lists.
            if output == 'elev_dist':
                data = swe.elev_dist(
                        dem_fn, shp_fn=shp_fn, step=step, touch_all=touch_all)

            elif output == 'average':
                data = swe.average(shp_fn=shp_fn, touch_all=touch_all)
                # HACK: data needs to be formatted as a Pandas.DataFrame!
                data = pd.DataFrame(
                        data=data, index=[swe.date()], columns=['SWE'])

            else:
                raise ValueError('Selected output file not recognised.')

            obs_swe = obs_swe.append(data)

        # Re-index the data
        if start is not None or end is not None:
            date_index = pd.date_range(start=start, end=end, freq='D')
            obs_swe = obs_swe.reindex(date_index)

        # Round the values.
        obs_swe = obs_swe.round(decimals=3)

        self._write_txt(obs_swe, filename, idx=True, header=None)

        return obs_swe

    def generate_tmean(
            self, temp_dir, shp_fn, freq='month', start=None,
            end=None, touch_all=False, save_file=True, filename='T_mean.txt'):
        """
        Generate an HBV-light T_mean.txt file.

        Parameters
        ----------
        temp_dir : str
            Directory where the temperature data is stored.
        shp_fn : str
            Path and filename of the basin shapefile delimiting the catchment.
        freq : {'month', 'day'}, optional
            Frequency of the long-term temperature averages,
            default is 'month'.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.
        save_file : boolean, optional
            Choose whether to save a file with the resulting data,
            default is True.
        filename : str, optional
            Name of the T_mean file, default is 'T_mean.txt'.

        Returns
        -------
        t_mean : Pandas.Series
            Pandas series containing the monthly or daily average temperature
            values.

        Raises
        ------
        ValueError
            If the provided averaging frequency is not recognised.

        """
        if save_file is True:
            print('...' + filename)

        # Calculate the average temperature over the catchment.
        temp = self._parse_temp(
                temp_dir, shp_fn=shp_fn, output='average',
                start=start, end=end, touch_all=touch_all)

        if freq == 'month':
            # Calculate the monthly mean values.
            temp_avg = temp.groupby(temp.index.month).mean()

        elif freq == 'day':
            # Calculate the mean values for each day of the year (j).
            temp_avg = temp.groupby([temp.index.month, temp.index.day]).mean()
            # Remove February 29th as the list should be of 365 values.
            temp_avg.drop((2, 29), inplace=True)

        else:
            raise ValueError('Averaging frequency not recognised.')

        # Round the data values and rename the Pandas.Series object.
        t_mean = temp_avg.round(decimals=3)
        t_mean.name = 'T_mean'

        if save_file is True:
            self._write_txt(t_mean, filename, idx=False, header=None)

        return t_mean

    def generate_evap(self, temp_dir, shp_fn, start=None,
                      end=None, touch_all=False, filename='EVAP.txt'):
        """
        Generate an HBV-light EVAP.txt file.

        Parameters
        ----------
        temp_dir : str
            Directory where the temperature data is stored.
        shp_fn : str
            Path and filename of the basin shapefile delimiting the catchment.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.
        filename : bool, optional
            Name of the Evap file, default is 'EVAP.txt'.

        Returns
        -------
        pet : Pandas.Series
            Pandas Series containing the mean monthly
            evapotranspiration values.

        """
        print('...' + filename)

        # Get the monthly average temperature over the catchment
        temp_m = self.generate_tmean(
                temp_dir, shp_fn, freq='month', start=start,
                end=end, touch_all=touch_all, save_file=False)

        # The the latitude of the centroid of the basin (deg)
        lon, lat = Shape(shp_fn).lonlat()

        # Get the monthly average extraterrestrial radiation
        re_m = Evap.mean_monthly_radiation(lat)

        # Calculate the mean monthly potential evaporation
        pet = Evap.potential_evaporation(re_m, temp_m)

        # Round the data values and rename the Pandas.Series object.
        pet = pet.round(decimals=3)
        pet.name = 'EVAP'

        self._write_txt(pet, filename, idx=False, header=None)

        return pet

    def generate_clarea(
            self, dem_fn, shp_fn, step=100,
            touch_all=False, filename='Clarea.xml'):
        """
        Generate an HBV-light Clarea.xml file.

        Parameters
        ----------
        dem_fn : str, optional
            Path and filename of the DEM file to get the elevation data from.
        shp_fn : str
            Path and filename of the shapefile defining the catchment boundary.
        step : int or float, optional
            Elevation band width to perform the calculations, default is 100.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.
        filename : str
            Name of the elevation distribution file, default is 'Clarea.xml'.

        Returns
        -------
        clarea : Pandas.DataFrame
            Pandas DataFrame containing the elevation distribution area
            percentages at a given time step for the given catchment.

        """
        print('...' + 'Clarea.xml')

        clarea = DEM(dem_fn).elev_area_dist(
                shp_fn=shp_fn, step=step, touch_all=touch_all)

        HBVconfig(self.bsn_dir).catchment_settings(clarea, filename=filename)

        return clarea

    def generate_metadata(
            self, precip_dir, temp_dir, dem_fn, shp_fn, stn_code,
            area_calc='shape', touch_all=False, filename='metadata.txt'):
        """
        Generate a metadata file for the given catchment.

        The metadata currently being generated are:
            - FOEN station code
            - FOEN station name
            - Latitude and longitude of the catchment centroid
            - Average elevation of the precipitation and temperature data
            - Average, maximum, and minimum catchment elevation
            - Catchment area

        Parameters
        ----------
        precip_dir : str
            Directory where the precipitation data is stored.
        temp_dir : str
            Directory where the temperature data is stored.
        dem_fn : str
            Path and filename of the DEM file to get the elevation data from.
        shp_fn : str
            Path and filename of the basin shapefile delimiting the catchment.
        stn_code : int
            Identification code of the BAFU hydrometric station defining
            the catchment.
        area_calc : {'shape', 'dem'}, optional
            Select if the catchment area should be calculated from the
            shapefile or from the DEM, default is 'shape'.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.
        filename : bool, optional
            Name of the metadata file, default is 'metadata.txt'.

        Returns
        -------
        meta : Pandas.DataFrame
            Data structure containing the catchment metadata.

        Raises
        ------
        ValueError
            If the provided area_calc parameter is not 'shape' or 'dem'.

        """
        print('...' + filename)

        # Initialise a Pandas.DataFrame
        meta = pd.DataFrame(index=[str(stn_code)])

        # Get the name of the catchment
        meta['Catchment'] = BasinShape(stn_code).station_name()

        # The the latitude of the centroid of the basin (deg)
        meta['lon'], meta['lat'] = Shape(shp_fn).lonlat()

        # Load the DEM dataset
        dem_ds = DEM(dem_fn)
        dem = dem_ds.mask(shp_fn, touch_all=touch_all)

        # Get the average elevation of the precipitation data
        p_file = glob.glob(precip_dir + '*.nc')[0]
        meta['Pelev'] = RhiresD(p_file).mean_elev(
                dem_fn, shp_fn=shp_fn, touch_all=touch_all)

        # Get the average elevation of the temperature data
        t_file = glob.glob(temp_dir + '*.nc')[0]
        meta['Telev'] = TabsD(t_file).mean_elev(
                dem_fn, shp_fn=shp_fn, touch_all=touch_all)

        # Get the minimum, mean, and maximum elevation of the catchment
        meta['Zmin'] = np.nanmin(dem)
        meta['Zavg'] = np.nanmean(dem)
        meta['Zmax'] = np.nanmax(dem)
        # TODO: Calculate catchment slope.

        # Get the area of the catchment (in km2)
        if area_calc == 'shape':
            meta['Area'] = Shape(shp_fn).area() / 1e6

        elif area_calc == 'dem':
            meta['Area_data'] = dem.count()

        else:
            raise ValueError('Area calculation method not recognised.')

        meta = meta.round(decimals=3)

        with open(self.data_dir + filename, 'w') as f:
            meta.to_csv(f, sep='\t', index_label='Code', header=True)

        return meta

    def generate_input_data(
            self, precip_dir, temp_dir, q_dir, swe_dir, sc_dir, dem_fn,
            stn_code, start=None, end=None, elev_step=100,
            t_mean_freq='month', touch_all=False):
        """
        Generate the necessary input data to run HBV-light.

        # TODO: Provide the possibility to decide which files to generate.

        Parameters
        ----------
        precip_dir : str
            Directory where the precipitation data is stored.
        temp_dir : str
            Directory where the temperature data is stored.
        q_dir : str
            Directory where the stream runoff data is stored.
        swe_dir : str
            Directory where the snow water equivalent data is stored.
        sc_dir : str
            Directory where the snow cover fraction data is stored.
        dem_fn : str
            Path and filename of the DEM file to get the elevation data from.
        stn_code : int
            Identification code of the BAFU hydrometric station defining
            the catchment.
        start : '%Y-%m-%d', optional
            Start date of the output dataset, default is None.
        end : '%Y-%m-%d', optional
            End date of the outpout dataset, default is None.
        elev_step : int or float, optional
            Width of the elevation bands to do the calculations for, default
            is 100.
        t_mean_freq : {'month', 'day'}, optional
            Frequency of average temperature values, default is 'month'.
        touch_all : bool, optional
            May be set to True to set all pixels touched by the line or
            polygons, not just those whose center is within the polygon or
            that are selected by brezenhams line algorithm, default is False.

        """
        print('Processing the input data for ' + str(stn_code) + '...')

        # Generate the basin shapefile (shp_fn)
        shp_fn = self.data_dir + 'basin.shp'
        BasinShape(stn_code).generate(shp_fn)

        # Generate the PTQ.txt file
        self.generate_ptq(
                precip_dir, temp_dir, q_dir, shp_fn, stn_code,
                start=start, end=end, touch_all=touch_all)

        # Generate the PTCALT.txt file (considering the three alternatives)
        self.generate_ptcalt(
                precip_dir, temp_dir, dem_fn, shp_fn, stn_code,
                start=start, end=end, pcalt=True, tcalt=True,
                touch_all=touch_all)

        # Generate the ObsSWE.txt file
        self.generate_swe(
                swe_dir, dem_fn, shp_fn, output='elev_dist', step=elev_step,
                start=start, end=end, touch_all=touch_all)

        # Generate the SnowCover.txt file
        self.generate_snow_cover(
                sc_dir, shp_fn, start=start, end=end, touch_all=touch_all)

        # Generate the EVAP.txt file
        self.generate_evap(
                temp_dir, shp_fn, start=start, end=end, touch_all=touch_all)

        # Generate the T_mean.txt file
        self.generate_tmean(
                temp_dir, shp_fn, freq=t_mean_freq, start=start,
                end=end, touch_all=touch_all, save_file=True)

        # Generate the Clarea.xml file
        self.generate_clarea(
                dem_fn, shp_fn=shp_fn, step=elev_step, touch_all=touch_all)

        # Generate a metadata.txt file
        self.generate_metadata(
                precip_dir, temp_dir, dem_fn, shp_fn,
                stn_code, touch_all=touch_all)

        # Remove temporary files
        for file in glob.glob(self.data_dir + 'basin*'):
            os.remove(file)

    def load_input_data(self, filename, no_data=-9999):
        """
        Load the data from a predefined HBV-light input data file.

        NOTE: Only default input data names are currently accepted. See
        the documentation of HBV-light for a description of the input data
        and the default file names.

        Parameters
        ----------
        filename : {'EVAP.txt', 'PTCALT.txt', 'ObsSWE.txt', 'PTQ.txt',
                    'SnowCover.txt', 'T_mean.txt'}
            Name of the input data file.
        no_data : int or float, optional
            Invalid data value, default is -9999.

        Returns
        -------
        Pandas.DataFrame or Pandas.Serie
            Data structure containing the selected input data type.

        Raises
        ------
        ValueError
            If the specified file does not exist.

        """
        filepath = self.data_dir + filename

        if not os.path.exists(filepath):
            raise ValueError('The file does not exist.')

        if filename.lower() == 'ptq.txt':
            board = pd.read_csv(
                    filepath, sep='\t', na_values=no_data, index_col=0,
                    parse_dates=True, skiprows=1, infer_datetime_format=True)
            board.index.rename('Date', inplace=True)

        elif filename.lower() in ['evap.txt', 't_mean.txt']:
            board = pd.read_csv(filepath)
            if len(board.index) == 12:
                board['Month'] = np.arange(1, 13)
                board.set_index('Month', inplace=True)
            elif len(board.index) == 365:
                board['Day'] = np.arange(1, 366)
                board.set_index('Day', inplace=True)

        elif filename.lower() == ' ptcalt.txt':
            board = pd.read_csv(
                    filepath, sep='\t', index_col=0, parse_dates=True,
                    skiprows=1, infer_datetime_format=True, squeeze=True)
            board.index.rename('Date', inplace=True)

        elif filename.lower() in ['snowcover.txt', 'obsswe.txt']:
            board = pd.read_csv(
                    filepath, sep='\t', index_col=0, parse_dates=True,
                    na_values=no_data, infer_datetime_format=True,
                    squeeze=True)
            board.index.rename('Date', inplace=True)

        else:
            raise ValueError('The specified filename is not recognised.')

        return board


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
