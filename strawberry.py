import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def quick_plot(dfx, geostr, txt = '', hline = None, timeadjust = 0):
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
    
    plt.xlim(df['datetime_int'].min(), df['datetime_int'].max())
    plt.ylim(15, 32)
    plt.show()