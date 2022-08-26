# -*- coding: utf-8 -*-
"""

Infos:
    https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/annual/kl/recent/KL_Jahreswerte_Beschreibung_Stationen.txt

00917 19950801 20211231            162     49.8809    8.6779 Darmstadt                                Hessen                                                                                            
00918 19871001 19950731            122     49.8453    8.6240 Darmstadt (A)                            Hessen                                                                                            
00919 19310101 19740731            169     49.8697    8.6796 Darmstadt-Botanischer Garten             Hessen                                                                                            
00920 18310101 19870930            108     49.8564    8.5929 Darmstadt (US-Air-Base)                  Hessen  


"""

import logging


from wetterdienst import Wetterdienst, Resolution, Period


from wetterdienst.provider.dwd.observation import DwdObservationDataset, DwdObservationRequest

from wetterdienst import Settings

from wetterdienst.provider.dwd.observation import (
    DwdObservationDataset,
    DwdObservationRequest,
    DwdObservationResolution,
)


log = logging.getLogger()


API = Wetterdienst(provider="dwd", network="observation")

stations = DwdObservationRequest(
   parameter=DwdObservationDataset.PRECIPITATION_MORE,
   resolution=Resolution.DAILY,
   period=Period.HISTORICAL
   )


print(stations.all().df.head())


def sql_example():
    """Retrieve temperature data by DWD and filter by sql statement."""
    Settings.tidy = True
    Settings.humanize = True
    Settings.si_units = False

    request = DwdObservationRequest(
        parameter=[DwdObservationDataset.TEMPERATURE_AIR],
        resolution=DwdObservationResolution.HOURLY,
        start_date="2000-01-01",
        end_date="2022-06-01",
    )

    stations = request.filter_by_station_id(station_id=(917,))

    sql = "SELECT * FROM data WHERE " "parameter='temperature_air_mean_200' AND value < -7.0;"
    log.info(f"Invoking SQL query '{sql}'")

    # Acquire observation values and filter with SQL.
    results_all = stations.values.all()
    
    df = stations.values.all().df
    
    
    
    
    results.filter_by_sql(sql)

    print(results.df)
    

#%%
# Find a station


def find_a_station():
    DwdObservationRequest(
    parameter=DwdObservationDataset.CLIMATE_SUMMARY,
    resolution=DwdObservationResolution.DAILY,
    period=DwdObservationPeriod.HISTORICAL,
    start_date="2000-01-01",
    end_date="2010-01-01"
    ).filter_by_rank(
        49.8656144,
        8.6741846,
        5
    ).df

