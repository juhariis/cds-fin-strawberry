# Functions for the strawberry 'study'
#
# netCDF4 processing scripts are largely based on examples in 
# - https://medium.com/@neetinayak/combine-many-netcdf-files-into-a-single-file-with-python-469ba476fc14
# - https://iescoders.com/reading-netcdf4-data-in-python/

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import netCDF4
import xarray
import os


def check_model_files(data_folder, 
                      model, 
                      filter_txt = '',
                      show_detail = False):
    """
    Overview on netCDF files of a model
    """
    file_list = os.listdir(data_folder + model)
    if filter_txt != '':
        file_list = [k for k in file_list if filter_txt in k]
    print(file_list)
    
    for afile in file_list:
        print('\n\n' + afile + '==================================================\n')
        rootgrp = netCDF4.Dataset(os.path.join(data_folder + model, afile), "r", format="NETCDF4")
        #print(rootgrp.data_model)
        print(rootgrp)
        print(rootgrp.dimensions)

        if show_detail:
            for akey in rootgrp.variables.keys():
                print('\n' + akey + ' *************************************')
                v = rootgrp.variables[akey] 
                print(v) 

        rootgrp.close()


def get_var_details(ds, 
                    show_all = True):
    print(ds)
    if show_all:
        for akey in ds.variables.keys():
            print('\n' + akey + ' *************************************')
            v = ds.variables[akey] 
            print(v) 

            
def get_netCDF4_file(data_folder, 
                     model, 
                     avar, 
                     data_aggregation = 'season', 
                     timerange = '', 
                     print_debug = 0):
    """
    Get original or combined data sets
    
    At the moment easiest way to get netCDF4 dataset is to read it from  combined file.
    Note that metadata shown comes from the first of the earlier combined files.
    There probably is an easier / native way to do this, but not found it yet.
    
    Note: when finished with the file, use .close() in ther calling function
    """
    if timerange == '':
        if data_aggregation != '':
            data_aggregation = '_' + data_aggregation
        afile = data_folder + model + data_aggregation + '_' + avar + '.nc'
    else:
        afile = (
            data_folder + model + '/' + 
            avar.lower() + '_C3S-glob-agric_' +
            model.lower() + '_rcp4p5_' +
            data_aggregation + '_' +
            timerange +
            '_v1.nc')
            
    if print_debug > 0: print(afile)
    f = netCDF4.Dataset(
        afile, 
        "r", 
        format="NETCDF4"
    )
    if print_debug > 1: print(f.variables.keys())
    for akey in f.variables.keys():
        if print_debug > 2: print('\n' + akey + ' *************************************')
        v = f.variables[akey] 
        if print_debug > 2: print(v) 
        
    if print_debug > 1: print('\n**************************************')
    if print_debug > 1: print(f.variables.keys())
    
    return f


def combine_to_netCDF4(folder, model, srctype, var):
    """
    Combine the unzipped files in subfolders into netCDF4 files in data root folder 
    
    Note: time dimension can have a problem for which a workaround is used
    based on https://github.com/pydata/xarray/issues/2921
    """
        
    ds_cdd = xarray.open_mfdataset(
        folder + model + '/' + var + '*' + srctype + '*nc',
        combine = 'by_coords', 
        concat_dim="time"
    )
    
    print('\nbefore -------------------------------------------------------')
    get_var_details(ds_cdd)
        
    # https://github.com/pydata/xarray/issues/2921
    
    # make sure the encoding is really empty
    assert not ds_cdd.time.encoding

    # assign encoding, such that they are equal
    ds_cdd.time.encoding.update(ds_cdd.time_bounds.encoding)
    
    print('\nafter -------------------------------------------------------')
    get_var_details(ds_cdd)
        
    out_file = (
        folder + 
        model +
        '_' +
        srctype +
        '_' +
        var +
        '.nc'
    )

    print(out_file)
    
    ds_cdd.to_netcdf(out_file)
    ds_cdd.close()    


def getclosest_ij(lats, lons, latpt, lonpt):
    """
    function to find the index of the point closest
    (in squared distance) to give lat/lon value
    """
    # find squared distance of every point on grid
    lats_dist = (lats - latpt)**2
    lons_dist = (lons - lonpt)**2
    return lats_dist.argmin(), lons_dist.argmin() 

def geo_avg_ts(v, 
               iy1_min, 
               iy2_min, 
               ix1_min, 
               ix2_min, 
               t0 = 0, 
               t1 = 1e9, 
               dt = 1):
    """
    Getting average over geographical area, returning time series
    Assumes rectangular area!!
    """
    vt = []
    for it in range(t0, t1, dt):
        varsum = 0
        n = 0
        for iy in range(min(iy1_min, iy2_min), max(iy1_min, iy2_min)):
            for ix in range(min(ix1_min, ix2_min), max(ix1_min, ix2_min)):
                #print('(%.d, %.d, %d): %7.4f %s, (%.2f, %.2f)' % (iy, ix, it, v[it, iy, ix], v.units, latvals[iy], lonvals[ix]))            
                varsum = varsum + v[it, iy, ix]
                n = n + 1
        vt.append(varsum/n)
    return vt


# First version of script for one point only

plt.rcParams['figure.figsize'] = [12, 6]

def do_model_variable(data_folder, models, thevars, geo_lat, geo_lon, geo_name, timerange = ''):
    """
    make analysis with variables and models listed for one single point
    """
    
    #print(models)
    #print(vars)
    #print('*******')
    
    for avar in thevars: 
        
        seasons = ['MAM', 'JJA', 'SON', 'DJF']
        subplots = [(0, 0), (0, 1), (1, 0), (1, 1)]
        title_txt = ''
        location = ''
        
        fig, axs = plt.subplots(2, 2)
        
        #fig=plt.figure(figsize=(24,16), dpi= 100, facecolor='w', edgecolor='k')

        for iseason in range(0, len(seasons)):
            
            season_dict = dict()
            
            season_v = []
            season_y = []
            
            for amodel in models:
                
                f = get_netCDF4_file(data_folder, amodel,  avar, 'season', timerange, 0)
                
                if location == '':
                    lat, lon = f.variables['lat'], f.variables['lon']
                    latvals = lat[:]
                    lonvals = lon[:] 
                    iy_min, ix_min = getclosest_ij(latvals, lonvals, geo_lat, geo_lon)
                    location = '%s %.2f N, %.2f E' % (geo_name, geo_lat, geo_lon)
                    v = f.variables[avar]
                    v_units = v.units
                    v_long_name = v.long_name
                    
                season_set = v[iseason::4,iy_min,ix_min].flatten()
                
                season_v.append(season_set.flatten())
                season_y.append(list(range(2011, 2011 + len(season_set))))
                
                season_dict[amodel] = season_set
                axs[subplots[iseason]].scatter(range(2011, 2011 + len(season_set)), season_set, label = amodel, alpha = 0.3)
                f.close()
                    
            if iseason in [0, 2]:
                axs[subplots[iseason]].set_ylabel(v_units)
            if iseason in [2, 3]:
                axs[subplots[iseason]].set_xlabel('year')
            axs[subplots[iseason]].set_title('%s' %  (seasons[iseason]), x = 0.1, y = 0.85)
            #axs[subplots[iseason]].legend()
            
            
            #data_season = pd.DataFrame({'y':season_v, 'x':season_y})
            sns.regplot(
                x = [item for sublist in season_y for item in sublist], 
                y = [item for sublist in season_v for item in sublist],
                ax = axs[subplots[iseason]],
                x_jitter = 0.45
            )

            
        fig.suptitle('%s - %s \n%s' %  (avar, v_long_name, location))
        plt.show()
                                

def get_pkl_name(data_folder,
                 time_aggregation, 
                 region):
    return(data_folder + 'df_' + time_aggregation + '_' + region + '.pkl')


def get_region_data(geo_region,
                    time_aggregation,
                    model_list,
                    var_list,
                    time_range,
                    data_folder,
                    cache_tag,
                    read_cache):

    cache_file_name = get_pkl_name(data_folder, time_aggregation + cache_tag, geo_region['tag'])
    if read_cache & os.path.exists(cache_file_name):
        df = pd.read_pickle(cache_file_name)
        print('read from cache ' + cache_file_name)
    else:
        df = geo_avg_models(
            data_folder, 
            model_list, 
            var_list, 
            time_aggregation,
            get_region_title(geo_region),
            geo_region['gp1']['lat_lon'], 
            geo_region['gp2']['lat_lon'], 
            time_range
        )
        df.to_pickle(cache_file_name)
        print('saved to cache ' + cache_file_name)

    return df            
            
            
            
def geo_avg_models(data_folder, 
                   models, 
                   thevars, 
                   data_aggregation, 
                   geo_name, 
                   gp1, 
                   gp2, 
                   timerange):
    """
    Returning geo averaged dataframe 
    """
    
    df_out = None
    
    from datetime import date, timedelta
    
    geo_name_txt = geo_name + ' [lat %.2f, lon %.2f x lat %.2f, lon %.2f]' % (gp1[0], gp1[1], gp2[0], gp2[1])
    
    for avar in thevars: 
        
        print(avar)
        
        geo_set = False
        
        v_model_v_ts = []
        v_model_t = []
        v_model_days_from1860 = []
        v_model_model = []
        var_id = []
        var_id_long = []
        var_id_units = []
        
        for amodel in models:
        
            #print('%s - %s' %(avar, amodel))
            
            f = get_netCDF4_file(data_folder, amodel,  avar, data_aggregation, timerange, 0)
            v = f.variables[avar]
            
            if not geo_set:
                
                # set common stuff for the variable
                
                v_time = f.variables['time']
                timevals = v_time[:] 
                
                t0 = 0
                t1 = len(timevals)
                dt = 1
                                
                v_units = v.units
                v_long_name = v.long_name
                
                # assuming lat/lon the same for all models
                
                lat, lon = f.variables['lat'], f.variables['lon']
                latvals = lat[:]
                lonvals = lon[:]
                                
                iy1_min, ix1_min = getclosest_ij(latvals, lonvals, gp1[0], gp1[1])
                iy2_min, ix2_min = getclosest_ij(latvals, lonvals, gp2[0], gp2[1])
                
                geo_set = True
                
   
            #print('%s - %s, %s' %(avar, v_long_name, amodel))
    
            v_ts = geo_avg_ts(v, iy1_min, iy2_min, ix1_min, ix2_min, t0, t1, dt)

            f.close()
            
            v_model_v_ts.append(v_ts)
            v_model_t.append(list(range(0, len(v_ts))))
            v_model_days_from1860.append(timevals[list(range(t0, t1, dt))])
            v_model_model.append(np.repeat(amodel, len(v_ts), axis=0))
            var_id.append(np.repeat(avar, len(v_ts), axis=0))
            var_id_long.append(np.repeat(v_long_name, len(v_ts), axis=0))
            var_id_units.append(np.repeat(v_units, len(v_ts), axis=0))
        
        avar_df = pd.DataFrame({
            'var_id': [item for sublist in var_id for item in sublist],
            'var_name': [item for sublist in var_id_long for item in sublist],
            'var_unit': [item for sublist in var_id_units for item in sublist],
            'var': [item for sublist in v_model_v_ts for item in sublist],
            'time': [item for sublist in v_model_t for item in sublist],
            'days_from1860': [item for sublist in v_model_days_from1860 for item in sublist],
            'model': [item for sublist in v_model_model for item in sublist]
        })
        
        avar_df['days_from1860_td'] = pd.to_timedelta(avar_df['days_from1860'], 'day')        
        avar_df['datetime'] = avar_df['days_from1860_td'] + pd.to_datetime(date(1860, 1, 1))
        
        avar_df['datetime_int'] =  avar_df['datetime'].values.astype(float)/10**9
        avar_df['datetime_date'] =  pd.to_datetime(avar_df['datetime_int']*10**9)
        
        
        if df_out is None:
            df_out = avar_df.copy()
        else:            
            df_out = pd.concat([df_out, avar_df])

        #print(avar_df.info())
        #print(avar_df.head(10))

        """
        
        To be removed
        
        if do_plot:
            
            ax = plt.gca()
            sns.regplot(
                x = 'datetime_int',
                y = 'var',
                data = avar_df,
                ax=ax
            )
            xticks = ax.get_xticks()
            ax.set_xticks(ax.get_xticks().tolist()) # REMOVE IN THE FUTURE - PLACED TO AVOID WARNING - IT IS A BUG FROM MATPLOTLIB 3.3.1
            xticks_dates = [pd.to_datetime(x*10**9).strftime('%Y-%m') for x in xticks]
            ax.set_xticklabels(xticks_dates)
            plt.title(avar + ' - ' + v_long_name + ' - ' + tag + '\n' + geo_name_txt)
            plt.xlabel('Models: ' + ', '.join(models))
            plt.ylabel(avar + ' ' + v_units)
            plt.show()

            #print(xticks_dates)
        """
        
    return df_out

def add_ids(df, region, srctype):
    df_out = df.copy()
    df_out['region'] = region
    df_out['srctype'] = srctype
    return df_out

def get_region_title(geostr):
    return geostr['name'] + ' (' + geostr['gp1']['name'] + '-' +  geostr['gp2']['name'] + ')'


def get_pred_interval_sm(y, X, dfx, pi = 0.95):
    
    import statsmodels.api as sm
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    from statsmodels.stats.outliers_influence import summary_table

    df = dfx.copy()
    Y = df[y]
    X = df[X]
    X = sm.add_constant(X)
    
    re = sm.OLS(Y, X).fit()
    
    print(re.summary())
    
    st, data, ss2 = summary_table(re, alpha=1-pi)
    fittedvalues = data[:, 2]
    predict_mean_se  = data[:, 3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
    predict_ci_low, predict_ci_upp = data[:, 6:8].T
    
    return predict_ci_low, fittedvalues, predict_ci_upp


def plot_max_temp_trend(dfx, geostr, txt = '', hline = None, timeadjust = 0):
    """
    NOTE!! There is a quick fix which only works for the 2011-2099 set, needs fixing for
    the standard case
    """
    geo_name_txt = (
        get_region_title(geostr) + 
        ' [lat %.2f, lon %.2f x lat %.2f, lon %.2f]' % 
        (
            geostr['gp1']['lat_lon'][0], 
            geostr['gp1']['lat_lon'][1], 
            geostr['gp2']['lat_lon'][0], 
            geostr['gp2']['lat_lon'][1]
        )
    )
    
    df = dfx.copy()
    
    title_txt = df['var_name'].iloc[0] + '\n' + geo_name_txt + txt
    
    model_list = df['model'].unique()
    
    if df['var_unit'].iloc[0] == 'K':
        df['var'] = df['var'] - 273
        df['var_unit'] = 'C'
        
    #print(df.head(1))
    #print(df.tail(1))
    lwr, pred, upr = get_pred_interval_sm('var', 'datetime_int', df)

    ax = plt.gca()
    sns.regplot(
        x = 'datetime_int',
        y = 'var',
        data = df, 
        scatter_kws={'alpha':0.2},
        ax=ax
    )
    xticks = ax.get_xticks()
    ax.set_xticks(ax.get_xticks().tolist()) # REMOVE IN THE FUTURE - PLACED TO AVOID WARNING - IT IS A BUG FROM MATPLOTLIB 3.3.1
    xticks_dates = [pd.to_datetime((x + timeadjust*1e+09)*10**9).strftime('%Y') for x in xticks]
    ax.set_xticklabels(xticks_dates)
    plt.xlabel('RCP4.5 experiments of models: ' + ', '.join(model_list))
    plt.title(title_txt)
    plt.ylabel(df['var_id'].iloc[0] + ' [' + df['var_unit'].iloc[0] + ']')
    
    # ref line
    if hline is not None:
        plt.hlines(hline, xmin = xticks[0], xmax = xticks[-1], linestyles='dotted', color='red')
    
    # prediction intervals
    plt.plot(df['datetime_int'], lwr, linestyle = 'dashed', color = 'gray')
    plt.plot(df['datetime_int'], upr, linestyle = 'dashed', color = 'gray')
    
    xmargin = (df['datetime_int'].max() - df['datetime_int'].min()) * 0.025
    plt.xlim(df['datetime_int'].min(), df['datetime_int'].max())
    
    plt.ylim(15, 32)
    plt.show()
    
    
def plot_max_temp_trend_dec(dfx, geostr, txt = '', hline = None, timeadjust = 0):
    """
    NOTE!! There is a quick fix which only works for the 2011-2099 set, needs fixing for
    the standard case
    """
    geo_name_txt = (
        get_region_title(geostr) + 
        ' [lat %.2f, lon %.2f x lat %.2f, lon %.2f]' % 
        (
            geostr['gp1']['lat_lon'][0], 
            geostr['gp1']['lat_lon'][1], 
            geostr['gp2']['lat_lon'][0], 
            geostr['gp2']['lat_lon'][1]
        )
    )
    
    df = dfx.copy()
    
    title_txt = df['var_name'].iloc[0] + '\n' + geo_name_txt + txt
    
    model_list = df['model'].unique()
    
    if df['var_unit'].iloc[0] == 'K':
        df['var'] = df['var'] - 273
        df['var_unit'] = 'C'
        
    #print(df.head(1))
    #print(df.tail(1))
    lwr, pred, upr = get_pred_interval_sm('var', 'date_decimal', df)

    ax = plt.gca()
    sns.regplot(
        x = 'date_decimal',
        y = 'var',
        data = df, 
        scatter_kws={'alpha':0.2},
        ax=ax
    )
    plt.xlabel('RCP4.5 experiments of models: ' + ', '.join(model_list))
    plt.title(title_txt)
    plt.ylabel(df['var_id'].iloc[0] + ' [' + df['var_unit'].iloc[0] + ']')
    
    # ref line
    if hline is not None:
        plt.hlines(hline, xmin = df['date_decimal'].min(), xmax = df['date_decimal'].max(), linestyles='dotted', color='red')
    
    # prediction intervals
    plt.plot(df['date_decimal'], lwr, linestyle = 'dashed', color = 'gray')
    plt.plot(df['date_decimal'], upr, linestyle = 'dashed', color = 'gray')
    
    xmargin = (df['date_decimal'].max() - df['date_decimal'].min()) * 0.025
    plt.xlim(df['date_decimal'].min() - xmargin, df['date_decimal'].max() + xmargin)
    
    plt.ylim(15, 32)
    plt.show()
    

def plot_gdd_trend(df, 
                     xvar, 
                     yvar, 
                     geostr, 
                     var_name, 
                     var_unit, 
                     txt = '', 
                     hline = None):
    
    geo_name_txt = (
        get_region_title(geostr) + 
        ' [lat %.2f, lon %.2f x lat %.2f, lon %.2f]' % 
        (
            geostr['gp1']['lat_lon'][0], 
            geostr['gp1']['lat_lon'][1], 
            geostr['gp2']['lat_lon'][0], 
            geostr['gp2']['lat_lon'][1]
        )
    )
    
    title_txt = var_name + '\n' + geo_name_txt + txt
    
    model_list = df['model'].unique()
            
    #print(df.head())

    lwr, pred, upr = get_pred_interval_sm(yvar, xvar, df)

    ax = plt.gca()
    sns.regplot(
        x = xvar,
        y = yvar,
        data = df, 
        scatter_kws={'alpha':0.2},
        ax=ax
    )
    xticks = ax.get_xticks()
    plt.xlabel('RCP4.5 experiments of models: ' + ', '.join(model_list))
    plt.title(title_txt)
    plt.ylabel(yvar + ' [' + var_unit + ']')
    if hline is not None:
        plt.hlines(hline, xmin = xticks[0], xmax = xticks[-1], linestyles='dashed', color='red')
    
    # prediction intervals
    plt.plot(df[xvar], lwr, linestyle = 'dashed', color = 'gray')
    plt.plot(df[xvar], upr, linestyle = 'dashed', color = 'gray')
    
    xmargin = (df[xvar].max() - df[xvar].min()) * 0.025
    plt.xlim(df[xvar].min() - xmargin, df[xvar].max() + xmargin)
    plt.ylim(1000, 1900)
    plt.show()
    
    
def plot_gdd_histogram(dfx, 
                     xvar, 
                     yvar, 
                     geostr, 
                     var_name, 
                     var_unit, 
                     txt = '', 
                     vlineh = None,
                     vline = None):

    import math as mth

    #print(yvar)
    
    df = dfx.copy()
    df['decade'] = [30*mth.floor(x/30) for x in (df[xvar])]

    print('mean')
    print(df.groupby('decade')[yvar].mean())
    print('sd')
    print(df.groupby('decade')[yvar].std())
    print('median')
    print(df.groupby('decade')[yvar].median())
    
    #print(df.head())

    geo_name_txt = (
        get_region_title(geostr) + 
        ' [lat %.2f, lon %.2f x lat %.2f, lon %.2f]' % 
        (
            geostr['gp1']['lat_lon'][0], 
            geostr['gp1']['lat_lon'][1], 
            geostr['gp2']['lat_lon'][0], 
            geostr['gp2']['lat_lon'][1]
        )
    )
    
    title_txt = var_name + '\n' + geo_name_txt + txt
    
    model_list = df['model'].unique()
            
    #ax = plt.gca()
    sns.displot(x = yvar, data = df, hue = 'decade', kind = 'kde')
    plt.xlabel('RCP4.5 experiments of models: ' + ', '.join(model_list))
    plt.title(title_txt)
    plt.ylabel('')  
    if vline is not None:
        plt.vlines(vline, ymin = 0, ymax = vlineh, linestyles = 'dashed', color = 'red')
    plt.vlines(df.groupby('decade')[yvar].mean(), ymin = 0, ymax = vlineh)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()
    
def show_region_on_map(geo_coord):
    """
    Show the region on map for illustration
    Coorner coordinate points shown only
    """
    import plotly
    import chart_studio.plotly as py
    import plotly.graph_objs as go
    from plotly.offline import plot, iplot, init_notebook_mode
    
    init_notebook_mode()
    
    cases = []
    cases.append(
        go.Scattergeo(
            lon = [
                geo_coord['gp1']['lat_lon'][1], 
                geo_coord['gp1']['lat_lon'][1], 
                geo_coord['gp2']['lat_lon'][1], 
                geo_coord['gp2']['lat_lon'][1]
            ],
            lat = [
                geo_coord['gp1']['lat_lon'][0], 
                geo_coord['gp2']['lat_lon'][0], 
                geo_coord['gp1']['lat_lon'][0], 
                geo_coord['gp2']['lat_lon'][0]
            ],
            marker = dict(
                size = 2,
                color = 'red',
                opacity = 0.9,
                line = dict(width = 2)
            ),
        )
    )
    cases[0]['mode'] = 'markers'

    layout = go.Layout(
        title = geo_coord['name'],
        geo = dict(
            resolution = 50,
            scope = 'europe',
            showframe = True,
            showcoastlines = True,
            showland = True,
            landcolor = "rgb(229, 229, 229)",
            countrycolor = "rgb(255, 255, 255)" ,
            coastlinecolor = "rgb(255, 255, 255)",
            lonaxis = dict( range= [ 19.0, 32 ] ),
            lataxis = dict( range= [ 60.0, 70.0 ] ),         
        ),
        legend = dict(
               traceorder = 'reversed'
        )
    )

    fig = go.Figure(
        layout=layout, 
        data=cases
    )

    fig.update_layout(
        autosize=False,
        width=300,
        height=500,
        margin=dict(
            l=3,
            r=3,
            b=10,
            t=30,
            pad=4
        ),
        paper_bgcolor="LightGray",
    )

    iplot(fig, validate=False, filename='iantest')
