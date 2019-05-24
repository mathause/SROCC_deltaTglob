import numpy as np
import pandas as pd
import re
import xarray as xr

from glob import glob
from os import path


from . import common
from . import xarray_utils


def cos_wgt(obj, lat_name='lat'):
    """cosine-weighted latitude"""
    return np.cos(np.deg2rad(obj[lat_name]))


# =============================================================================

def calc_anomaly(data, start, end):
    """calculate reference anomaly wrt a reference period"""

    def _calc_anomaly(ds):
        mean = ds.sel(year=slice(start, end)).mean('year')
        return ds - mean

    if isinstance(data, dict):
        data_out = dict()
        for scen in data.keys():
            data_out[scen] = _calc_anomaly(data[scen].copy(deep=True))
    else:
        data_out = _calc_anomaly(data.copy(deep=True))

    return data_out


# =============================================================================

def select_first_ens(tas_anom):
    """selects the first ensemble number for each model"""

    tas_anom_one_ens = dict()
    for scen in tas_anom.keys():
        tas_anom_one_ens[scen] = tas_anom[scen].sel(ens_number=0)
    
    return tas_anom_one_ens


# =============================================================================

def calc_warming(tas_anom, period):
    """calculate global mean warming relative to reference period"""

    print("Period: {} to {}".format(*period))
    
    for scen in tas_anom.keys():
        dta = tas_anom[scen]
        
        dta = dta.sel(year=slice(*period))
    
        mn = dta.mean().tas.values
        std = dta.std().tas.values
    
        likely_min = mn - std * 1.64
        likely_max = mn + std * 1.64
    
        print(f"{scen}: {mn:3.1f} --> {likely_min:3.1f} to {likely_max:3.1f}")


# =============================================================================

def list_ensmble_members(tas_anom):
    """create pandas data array showing used ens members"""

    all_mods = list()
    for scen in tas_anom.keys():
        df = tas_anom[scen].ens.to_pandas()
        df.name = scen
        all_mods.append(df)

    df = pd.concat(all_mods, axis=1, sort=True)
    df = df.fillna('-')
    
    return df


# =============================================================================
# sorting in human order


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    a_list.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html

    Example
    -------
    > l = ['a10', 'a1']
    > l.sort(key=natural_keys)
    > l
    ['a1', 'a10']

    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# =============================================================================

class CMIP_ng:
    """class to read and postprocess cmipX-ng data for the SROCC report

        For data access see https://data.iac.ethz.ch/atmos/

    """
    def __init__(self, folder_root, scens, cmip):
        """
        set paths etc for cmip3_ng and cmip5_ng

        Parameters
        ----------
        folder_root: string
            Root of the data repository.
        scens: list of string
            List of all scenarios that are avaliable.
        cmip: 'cmip3' | 'cmip5'
            Indicates which cmip version we are using.
        """
        self.folder_root = folder_root
        self.scens = scens
        self.cmip = cmip

        # format of the files (it's the same for cmip3-ng and cmip5-ng)
        self.file_format = "{var}_{time}_{model}_{scen}_{ens}_{res}.nc"

        self._tas_all_scens = None

    def __repr__(self):
        msg = f"<CMIP_ng> Utilities to read and postprocess {self.cmip} data"

        return msg

    def filename(self, var, time, model, scen, ens, res='native'):
        """
        list cmip5 filenames according to criteria

        Parameters
        ----------
        var : string
            Variable name.
        time : string
            Time resolution, e.g. 'ann', 'seas'.
        model : string
            Models to look for, e.g. '*', 'NorESM1'
        scen : string
            Scenario, e.g. 'rcp85', ...
        ens : string
            Which ensemble members, e.g. '*', 'r1i1p?', 'r1i1p1'
        res : string
            Resolution, 'native' or 'g025'. Optional, default: 'g025'.

        ..note::

        All arguments can take wildcards.

        """

        folder = path.join(self.folder_root, var)

        kwargs = dict(var=var, time=time, model=model, scen=scen, ens=ens,
                      res=res)
        file = self.file_format.format(**kwargs)

        fN = path.join(folder, file)

        fNs = glob(fN)

        # sort such that 'r1i1' is before 'r10i1'
        fNs.sort(key=natural_keys)

        if not fNs:
            raise RuntimeError('No simulations found')

        return fNs

    @staticmethod
    def process_one_model(fN):
        """calculate global mean annual mean for one cmip5 model
        """

        ds = xr.open_dataset(fN)

        # read model name and ensemble number
        model = [ds.attrs['source_model']]
        ens = [ds.attrs['source_ensemble']]

        # assign info as coordinate
        ds = ds.assign_coords(model=model, ens=ens)

        # get cosine-weighted latitude
        wgt = common.cos_wgt(ds)

        # compute weighted global mean
        ds = xarray_utils.Weighted(ds, wgt).mean(('lat', 'lon'))

        # convert from time index to year index
        ds = ds.assign_coords(year=ds.year.dt.year)

        ds = ds.set_index(model_ens=('model', 'ens'))

        unique_years = np.unique(ds.year)
        if len(ds.year) != len(unique_years):
            msg = 'Removing duplicate years for {} {}'
            print(msg.format(model, ens))

            _, index = np.unique(ds.year, return_index=True)
            ds = ds.isel(year=index)

        return ds

    def get_name_postprocess(self, scen):
        """get the name for the postprocessed data"""

        filename = '{cmip}_tas_ann_globmean_{scen}.nc'
        filename = filename.format(cmip=self.cmip, scen=scen)
        return path.join('..', 'data', self.cmip, filename)


    def postprocess_global_mean_tas(self):
        """postprocess and save all models

            Note:
            this only needs to be done once
        """

        # loop through all scenarios
        for scen in self.scens:

            # get the filename to save the data
            fN_out = self.get_name_postprocess(scen)

            print("=========================")

            # do not compute again
            if path.isfile(fN_out):
                msg = 'File for {scen} exists! -- skipping \n{fN_out}'
                print(msg.format(scen=scen, fN_out=fN_out))
                continue

            print(scen)

            # get all filenames for one scenario
            fNs = self.filename('tas', 'ann', '*', scen, '*')

            # accumulate all data for one scen
            all_data = list()
            ens_number = list()
            model_before = ""
            for i, fN in enumerate(fNs):
                ds = self.process_one_model(fN)

                model = ds.model.values
                if (model == model_before):
                    ens_number.append(ens_number[i - 1] + 1)
                else:
                    ens_number.append(0)
                
                model_before = model

                all_data.append(ds)

                print("File {: 2d} of {:02d}".format(i + 1, len(fNs)))

            # concatenate the data
            ds = xr.concat(all_data, dim='model_ens')

            # add ens_number as coordinate
            ds =  ds.assign_coords(ens_number=('model_ens', ens_number))

            # we need to get rid of the multiindex, as xarray cannot save it
            ds = ds.reset_index('model_ens')

            # save to netcdf
            ds.to_netcdf(fN_out, format='NETCDF4_CLASSIC')

    def _get_tas(self, scen):
        # get tas for one scenario

        fN = self.get_name_postprocess(scen)

        ds = xr.open_dataset(fN)

        # create the multiindex again
        return ds.set_index(model_ens=('model', 'ens_number'))

    @property
    def tas_all_scens(self):
        """get tas for all scenarios"""

        if self._tas_all_scens is None:
            dta = dict()

            for scen in self.scens:
                dta[scen] = self._get_tas(scen)

            self._tas_all_scens = dta

        return self._tas_all_scens
