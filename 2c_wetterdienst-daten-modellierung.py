# -*- coding: utf-8 -*-
"""

Beispiel Wetterdaten verarbeiten und 

mit einem Gaussian anpassen

ref: https://github.com/earthobservations/wetterdienst/blob/main/example/climate_observations.ipynb

"""

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from wetterdienst import Settings

from wetterdienst.provider.dwd.observation import DwdObservationRequest, \
    DwdObservationPeriod, DwdObservationResolution, DwdObservationParameter, DwdObservationDataset

# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd

from numpy import loadtxt

from lmfit.models import GaussianModel, LinearModel

#%%
# all
print("All available parameters")
print(
    DwdObservationRequest.discover()
)
# selection
print("Selection of daily data")
print(
    DwdObservationRequest.discover(
        filter_=DwdObservationResolution.DAILY
    )
)

Settings.si_units = False

STATION_ID = 917
DWD_parameters = DwdObservationParameter.DAILY.TEMPERATURE_AIR_MEAN_200

#%%

def wetterdienst_plot_deutschland():
    '''
    # 1. First check the metadata to inform yourself of available stations

    #(here we choose historical daily precipitation - hdp)
    
    ref: https://github.com/earthobservations/wetterdienst/blob/main/example/climate_observations.ipynb

    '''

    request = DwdObservationRequest(
        parameter=DwdObservationDataset.PRECIPITATION_MORE,
        resolution=DwdObservationResolution.DAILY,
        period=DwdObservationPeriod.HISTORICAL
    )
    print("Number of stations with available data: ", request.all().df.sum())
    print("Some of the stations:")
    request.all().df.head()
    
    cmap = cm.get_cmap('viridis')
    bounds = request.all().df.height.quantile([0, 0.25, 0.5, 0.75, 1]).values
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot = request.all().df.plot.scatter(
        x="longitude", y="latitude", c="height", cmap=cmap, ax=ax) #norm=norm, 
    plot.set_title("Map of daily precipitation stations in Germany\n"
                   "Color refers to height of station")
    plt.show()

wetterdienst_plot_deutschland()
    
#%%

def wetterdaten_laden(DWD_parameters): 
    
   
    DA_temp_request = DwdObservationRequest(
        parameter=DWD_parameters,
        resolution=DwdObservationResolution.DAILY,
        # period=DwdObservationPeriod.RECENT,
        start_date="2018-12-25",
        end_date="2022-01-01" 
    ).filter_by_station_id(station_id=(STATION_ID, ))
    
    DA_temp_station_values = DA_temp_request.values.all().df.dropna(axis=0)
    
    DA_temp_station_values['MA'] = DA_temp_station_values.value.rolling(3).mean()
    
    for (parameter, group) in DA_temp_station_values.groupby('parameter'):
        group.plot.scatter(x='date', y='value', title=parameter)

    DA_temp_station_values.plot.scatter(x='date', y='MA')
    return DA_temp_station_values

DA_temp_station_values = wetterdaten_laden(DWD_parameters)

#%%
def fit_gaussian_model(DA_temp_station_values, savefig=True):
    '''
    
    Verwendet:
    lmfit 
    https://lmfit.github.io/lmfit-py/
    Non-Linear Least-Squares Minimization and Curve-Fitting for Python
    
    Ziel:
    ein Gaussian Kurve an die Daten anpassen.
    

    '''
      
    number_of_years = DA_temp_station_values.date.dt.year.nunique()
    
    x = DA_temp_station_values.index.to_numpy()
    y = DA_temp_station_values.value.to_numpy()
    
    
    g19 = GaussianModel(prefix='g19_')
    pars = g19.make_params()
    pars['g19_center'].set(value=1*x.max()/6, min=x.max()/8, max=x.max()/3)
    pars['g19_sigma'].set(value=x.max()/12, min=3, max=x.max()/3)
    pars['g19_amplitude'].set(value=5*y.max(), min=10)
    pars['g19_height'].set(value=5*y.max(), min=10)

    
    g20 = GaussianModel(prefix='g20_')
    pars.update(g20.make_params())
    
    pars['g20_center'].set(value=3*x.max()/6, min=1.5*x.max()/6, max=4.5*x.max()/6)
    pars['g20_sigma'].set(value=x.max()/12, min=3, max=x.max()/3)
    pars['g20_amplitude'].set(value=5*y.max(), min=10)
    
    g21 = GaussianModel(prefix='g21_') 
    pars.update(g21.make_params())
    
    pars['g21_center'].set(value=5*x.max()/6, min=4*x.max()/6, max=x.max())
    pars['g21_sigma'].set(value=x.max()/12, min=3, max=x.max()/3)
    pars['g21_amplitude'].set(value=5*y.max(), min=10)
    
    lin = LinearModel(prefix='lin_')
    pars.update(lin.make_params())
    
    pars['lin_slope'].set(value=0.01, min=1E-3, max=5)
    pars['lin_intercept'].set(value=1, min=1E-9, max=20)
    
    mod =  g19 + g20 + g21 + lin
    # mod.make_params()
    # pars = mod.guess(y, x=x)
    out = mod.fit(y, pars, x=x)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    out.plot_fit()
    #plt.show()
    if savefig:
        plt.savefig('fit_gaussian_model.png', dpi=300, bbox_inches='tight')
    
    print(out.fit_report(min_correl=0.25))

fit_gaussian_model(DA_temp_station_values)
#%%
class FitYearlyGaussians:
    
    '''
    
    Input:
        
    
    Verwendet:
    lmfit 
    https://lmfit.github.io/lmfit-py/
    Non-Linear Least-Squares Minimization and Curve-Fitting for Python
    
    Ziel:
    ein Gaussian Kurve an die Daten anpassen, wobie die Daten durchlaufen 
    wurden und nur die komplette Jahren für die Anpassung verwendet werden.
    
    
    '''
    
    def __init__(self, station_data ):
        self._station_data = station_data
        
        self.valid_years = self.select_valid_years()
        self.valid_years_data = pd.concat([i[1] for i in self.valid_years])
        self.number_of_years = len(self.valid_years )
        
        
        self.x = self.valid_years_data.index.to_numpy()
        self.y = self.valid_years_data.value.to_numpy()
        
        self.index_per_year = self.x.max()/self.number_of_years
        
        self.models, self.pars = self.make_composite_model()
        
        self.out = self.fit_models()
        
        # self.out.plot_fit()
        self.plot_result_fit()
        print(self.out.fit_report(min_correl=0.25))
    
    def select_valid_years(self):
        
        lst = []
        
        for (year, group) in self._station_data.groupby(self._station_data.date.dt.year):
            
            if not (group.date.min().month <= 2 and group.date.max().month > 10): 
                print(f'skip year {year}')
                continue
            lst.append((year, group))
        
        return lst
        
    
    def make_composite_model(self):
        
        pars = None
        models = None
        
        
        for year, group in self.valid_years:
            gmod = GaussianModel(prefix=f'g{year}_')
            if not pars:
                pars = gmod.make_params()
            else:
                pars.update(gmod.make_params())
            
                
            pars = self.update_pars(year, group, pars)
            if not models:
                models = gmod
            else:
                models = models + gmod
        return models, pars
    
    
    def update_pars(self, year, group, pars):
        idx = group.index.to_numpy()
        mean_index, max_index, min_index = idx.mean(), idx.max(), idx.min()
        pars[f'g{year}_center'].set(value=mean_index, min=0.75*mean_index, max=1.25*mean_index)
        # pars[f'g{year}_sigma'].set(value=self.index_per_year/2, min=3, max=self.index_per_year)
        pars[f'g{year}_sigma'].set(value=self.index_per_year/4, min=3, max=100)
        pars[f'g{year}_amplitude'].set(value=5*self.y.max(), min=10)
        # pars[f'g{year}_height'].set(value=20, min=2, max=45, vary=True)
        return pars
    
    def fit_models(self):
        out = self.models.fit(self.y, self.pars, x=self.x)
        print(f'Result: {out.result.message}')
        
        return out
    
    def slice_fit_result_param(self, parname='height'):
        
        lst= []
        for year, group in self.valid_years:
            lst.append((year, self.out.result.params[f'g{year}_{parname}']))
            
        df = pd.DataFrame([(y, p.value) for y,p in lst], columns=['year',f'temp_{parname}'])
            
        return df
    
    def plot_yearly_params(self,  parname='height'):
        
        
        self.slice_fit_result_param(parname=parname).plot.scatter(x='year',y=f'temp_{parname}')
    
    def plot_result_fit(self, savefig=True):
        if savefig:
            fig, ax = fig, ax = plt.subplots(figsize=(12, 12))
        pd.DataFrame({'year': self.valid_years_data.date,
                      'value': self.y,
                      'fit': self.out.best_fit}).plot(x='year',y=['value', 'fit'], 
                                                      title=self.valid_years_data.parameter.unique()[0])
        if savefig:
            figname = f'{self.__class__.__qualname__}_wetter_fit_{self.number_of_years}'
            plt.savefig(figname, dpi=300)
            plt.show()
    
    def to_excel(self, name):
        
        date_columns = df.select_dtypes(include=['datetime64[ns, UTC]']).columns
        for date_column in date_columns:
            df[date_column] = df[date_column].dt.date
#%%


def fit_beispiel_1jahr(DWD_parameters):
    
    DA_temp_request_1y = DwdObservationRequest(
        parameter=DWD_parameters,
        resolution=DwdObservationResolution.DAILY,
        # period=DwdObservationPeriod.RECENT,
        start_date="2020-12-25",
        end_date="2022-01-01"
    ).filter_by_station_id(station_id=(STATION_ID, ))
    
    DA_temp_station_values_1y = DA_temp_request_1y.values.all().df.dropna(axis=0)
    
    yg_1y = FitYearlyGaussians(DA_temp_station_values_1y)
    return yg_1y
    
guassian_1jahr = fit_beispiel_1jahr(DWD_parameters)



#%%  
        
def fit_beispiel_mehrere_jahre(DWD_parameters):
    
    DA_temp_request = DwdObservationRequest(
        parameter=DWD_parameters,
        resolution=DwdObservationResolution.DAILY,
        # period=DwdObservationPeriod.RECENT,
        start_date="1995-12-25",
        end_date="2022-01-01"
    ).filter_by_station_id(station_id=(STATION_ID, ))
    
    DA_temp_station_values = DA_temp_request.values.all().df.dropna(axis=0)
    
    yg = FitYearlyGaussians(DA_temp_station_values)
    
    models,pars = yg.make_composite_model()

gaussian_mehrere_jahre = fit_beispiel_mehrere_jahre(DWD_parameters)
#%%    

 # DWD_pars2 = [
 #     DwdObservationDataset.TEMPERATURE_AIR,
 #     DwdObservationDataset.TEMPERATURE_SOIL
 # ]
 