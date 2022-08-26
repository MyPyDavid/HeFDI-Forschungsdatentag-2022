# -*- coding: utf-8 -*-
"""

@ref: https://github.com/earthobservations/wetterdienst/blob/main/example/climate_observations.ipynb

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd

from wetterdienst import Settings

from wetterdienst.provider.dwd.observation import DwdObservationRequest, \
    DwdObservationPeriod, DwdObservationResolution, DwdObservationParameter, DwdObservationDataset
    

#%%
def wetterdienst_plot_deutschland(savefig=None):
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
    if savefig:
        plt.savefig('wetterdienst_plot_deutschland.png')
    
    return request

request = wetterdienst_plot_deutschland(savefig=True)
#%%

STATION_ID = 917

# 2. The usual way of retrieving data


print("Receiving historical daily climate data for Dresden-Klotzsche (1048)")
station_data = request.filter_by_station_id(station_id=[STATION_ID]).values.all().df

station_data.dropna(axis=0).head()

print("Receiving historical daily temperature and precipitation for Dresden-Klotzsche "
      "(1048).")

request = DwdObservationRequest(
    parameter=[
        DwdObservationParameter.DAILY.TEMPERATURE_AIR_MEAN_200,
        DwdObservationParameter.DAILY.TEMPERATURE_AIR_MAX_200,
        DwdObservationParameter.DAILY.TEMPERATURE_AIR_MIN_200,
        DwdObservationParameter.DAILY.PRECIPITATION_HEIGHT
    ],
    resolution=DwdObservationResolution.DAILY,
    period=DwdObservationPeriod.RECENT
).filter_by_station_id(station_id=(STATION_ID, ))

station_data = request.values.all().df

station_data.dropna(axis=0).head()

#%%
# 3. Let's create some plots

cmap = plt.get_cmap('viridis', 4)
colors = cmap.colors

PARAMETERS = ["tnk", "tmk", "txk", "rsk"]

station_data_grouped = station_data.groupby(station_data["parameter"], observed=True)

fig, axes = plt.subplots(nrows=len(PARAMETERS), tight_layout=True, figsize=(10, 40))

for (parameter, group), ax, color in zip(station_data_grouped, axes, colors):
    group.plot(x="date", y="value", label=parameter, alpha=.75, ax=ax, c=color)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle(f"Temperature and precipitation time series of Station {STATION_ID}, Germany")
plt.show()
savefig = True
if savefig:
    plt.savefig(f'temperature_precipitation_{STATION_ID}.png', dpi=300, bbox_inches='tight')

#%%
# 4. Create yearly values


station_data_yearly = []

for (year, parameter), group in station_data.groupby(
        [station_data["date"].dt.year, "parameter"], as_index=False, observed=True):
    if parameter == "rsk":
        station_data_yearly.append(group.dropna().agg({"value": np.sum}))
    else:
        station_data_yearly.append(group.dropna().agg({"value": np.mean}))

station_data_yearly = pd.concat(station_data_yearly)

station_data_yearly

#%%

# 5. Find a station

stations_rank = DwdObservationRequest(
    parameter=DwdObservationDataset.CLIMATE_SUMMARY,
    resolution=DwdObservationResolution.DAILY,
    period=DwdObservationPeriod.RECENT,
    start_date="2000-01-01",
    end_date="2010-01-01"
).filter_by_rank(
    49.8656144,
    8.6741846,
    10
).df
    