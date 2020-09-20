# cds-fin-strawberry

## Overview

This is script for the team work for Copernicus Climate Data Services
couse in Finland 2020.

The script is not fully self explanatory and may produce unexpected results.
No guarantees given.

Calculations are not necessarily exactly right, the purpose within 
time constraints was to come up with a proof of concept
for potential further development only.

If you notice anything horrendoulsy wrong - or things that could be done 
in a better way, please let me know. 
My main language has been R and there are still a few strange areas in 
Python ecosystem (and climate data handling) where I very much would like 
to learn more.

Comments are very much appreciated

## Data

Subset of [Agroclimatic indicators from 1951 to 2099 derived from climate projections](https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agroclimatic-indicators?tab=overview) at Copernicus Climate Service was used.

RCP 4.5 data from three European models was used

* *ipsl_cm5a_lr_model*: IPSL-CM5A-LR Model (IPSL, France)
* *hadgem2_es_model*: HadGEM2-ES Model (UK Met Office, UK)
* *noresm1_m_model*: NorESM1-M Model (NCC, Norway)

Details about the variables and their actual names in downloaded data are available via the [Documentation](https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agroclimatic-indicators?tab=doc) page.
Selection of data for downloading takes place on [Download Data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agroclimatic-indicators?tab=form) -page

Resulting API requests are as follows

### Annual data

> import cdsapi  
> 
> c = cdsapi.Client()  
> 
> c.retrieve(  
>     'sis-agroclimatic-indicators',  
>     {  
>         'origin': '*replace_this_with_model_name*',  
>         'variable': 'growing_season_length',  
>         'experiment': 'rcp4_5',  
>         'temporal_aggregation': 'annual',  
>         'period': [  
>             '201101_204012', '204101_207012', '207101_209912',  
>         ],  
>         'format': 'zip',  
>     },  
>     'download.zip')  

### Seasonal data

> import cdsapi
> 
> c = cdsapi.Client()
> 
> c.retrieve(  
>     'sis-agroclimatic-indicators',  
>     {  
>         'origin': '*replace_this_with_model_name*',  
>         'variable': [  
>             'cold_spell_duration_index', 'maximum_number_of_consecutive_dry_days', 'maximum_number_of_consecutive_frost_days', 'maximum_number_of_consecutive_summer_days', 'maximum_number_of_consecutive_wet_days', 'warm_and_wet_days', 'warm_spell_duration_index',  
>         ],  
>         'experiment': 'rcp4_5',  
>         'temporal_aggregation': 'season',  
>         'period': [  
>             '201101_204012', '204101_207012', '207101_209912',  
>         ],  
>         'format': 'zip',  
>     },  
>     'download.zip')  


### Decadal data

> import cdsapi  
> 
> c = cdsapi.Client()  
> 
> c.retrieve(  
>     'sis-agroclimatic-indicators',  
>     {  
>         'origin': '*replace_this_with_model_name*',  
>         'variable': [  
>             'mean_of_daily_maximum_temperature', 'mean_of_daily_mean_temperature',  
>         ],  
>         'experiment': 'rcp4_5',  
>         'temporal_aggregation': '10_day',  
>         'period': [  
>             '201101_204012', '204101_207012', '207101_209912',  
>         ],  
>         'format': 'zip',  
>     },  
>     'download.zip')  

### Toolbox request

Toolbox request can be written based on the above API requests and the format of the toolbox request below

> import cdstoolbox as ct  
> 
> @ct.application(title='Download data')  
> @ct.output.download()  
> def download_application():  
>     data = ct.catalogue.retrieve(  
>         'sis-agroclimatic-indicators',  
>         {  
>             'origin': 'noresm1_m_model',  
>             'variable': 'growing_season_length',  
>             'experiment': 'rcp4_5',  
>             'temporal_aggregation': 'annual',  
>             'period': [  
>                 '201101_204012', '204101_207012', '207101_209912',  
>             ],  
>         }  
>     )  
>     return data  

## Organization of data

The scripts assume that the zip files are extracted into
subfolders of a main data folder for the excercise.
Subfoders have the name of the model.

Preprocessed (cached) datasets will be created into the maindata folder level.

E.g.

> D:\MyData\climate\strawberries>tree  
> Folder PATH listing for volume Data  
> Volume serial number is XXXXXXXXX  
> D:.  
> ├───HadGEM2-ES  
> ├───IPSL-CM5A-LR  
> └───NorESM1-M  
> 
> D:\MyData\climate\strawberries>  
