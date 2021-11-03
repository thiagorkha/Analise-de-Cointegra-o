import base64
from io import StringIO
from io import BytesIO
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as mplt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from scipy.stats import linregress
import streamlit as st



def get_market_data(tickers, period, interval):
    """
    https://github.com/ranaroussi/yfinance/issues/363
    """
    data = yf.download(
        tickers = " ".join(tickers),
        period = period,
        #start=datetime.today() - timedelta(days = period),
        #end= datetime.today (),
        interval = interval,
        #group_by = 'ticker',
        #auto_adjust = True,
        #prepost = False,
        treads = False,
        #proxy = None
    )
    return data


def coint_model(series_x, series_y):
    try:
        X = sm.add_constant(series_x.values)
        mod = sm.OLS(series_y, X)
        results = mod.fit()
        adfTest = adfuller(results.resid, autolag='AIC')
        lin = linregress(series_x,series_y)
        lin = lin[0]
        return {
            'OLS': results,
            'ADF': adfTest,
            'Lin': lin,
        }
    except:
        raise

def half_life(ts):
    lagged = ts.shift(1).fillna(method="bfill")
    delta = ts-lagged
    X = sm.add_constant(lagged.values)
    ar_res = sm.OLS(delta, X).fit()
    half_life = -1*np.log(2)/ar_res.params['x1']
    return half_life, ar_res 

def asBase64(my_plt):
    _buffer = BytesIO()
    my_plt.savefig(_buffer, format='png', bbox_inches='tight')
    _buffer.seek(0)
    return base64.encodestring(_buffer.read())

def fp_savefig(my_plt):
    _buffer = BytesIO()
    my_plt.savefig(_buffer, format='png', bbox_inches='tight')
    _buffer.seek(0)
    return _buffer

def _get_residuals_plot(ols):
    # TODO: descobrir qual é correto
    stddev = ols.resid.std()
    xmin = ols.resid.index.min()
    xmax = ols.resid.index.max()
    mplt.figure(figsize=(15,7))
    # limpa o canvas
    mplt.clf()
    mplt.cla()
    #mplt.close()
    mplt.plot(ols.resid, color='k')
    mplt.xticks(rotation=90)

    mplt.hlines([0], xmin, xmax, color='whitesmoke')
    mplt.hlines([-1*stddev, 1*stddev], xmin, xmax, color='gainsboro')
    mplt.hlines([-1.96*stddev, 1.96*stddev], xmin, xmax, color='orange')
    mplt.hlines([-3*stddev, 3*stddev], xmin, xmax, color='red')
    
    return mplt.show()


def st_get_residuals_plot(ols):
    # TODO: descobrir qual é correto
    stddev = ols.resid.std()
    xmin = ols.resid.index.min()
    xmax = ols.resid.index.max()
    mplt.figure(figsize=(15,7))
    # limpa o canvas
    mplt.clf()
    mplt.cla()
    #mplt.close()
    mplt.plot(ols.resid, color='k')
    mplt.xticks(rotation=90)

    mplt.hlines([0], xmin, xmax, color='whitesmoke')
    mplt.hlines([-1*stddev, 1*stddev], xmin, xmax, color='gainsboro')
    mplt.hlines([-1.96*stddev, 1.96*stddev], xmin, xmax, color='orange')
    mplt.hlines([-3*stddev, 3*stddev], xmin, xmax, color='red')
    
    return st.pyplot(mplt)


def beta_rotation(series_x, series_y, window=40):
    beta_list = []
    try:
        for i in range(0, len(series_x)-window):
            slice_x = series_x[i:i+window]
            slice_y = series_y[i:i+window]

            X = sm.add_constant(slice_x.values)
            mod = sm.OLS(slice_y, X)
            results = mod.fit()
            beta = results.params.x1
            beta_list.append(beta)
    except:
        raise

    return beta_list

def beta_rotation1(series_x, series_y, window=40):
       
    get_market_data([series_x, series_y], '2y', '1d')
    data = get_market_data([series_x, series_y], '2y', '1d')
    
    market_data = data[-400:]

    market_data = market_data.dropna()
    
    series_x = market_data['Close'][series_x]
    series_y = market_data['Close'][series_y]

    beta_list = []
    try:
        for i in range(0, len(series_x)-window):
            slice_x = series_x[i:i+window]
            slice_y = series_y[i:i+window]

            X = sm.add_constant(slice_x.values)
            mod = sm.OLS(slice_y, X)
            results = mod.fit()
            beta = results.params.x1
            beta_list.append(beta)
    except:
        raise

    return beta_list    

def get_beta_plot(beta_list):
    # limpa o canvas
    mplt.figure(figsize=(20,5))
    mplt.clf()
    mplt.cla()
    #mplt.close()
    try:
        mplt.plot(beta_list, color='limegreen')
    except ValueError:
        mplt.plot([], color='limegreen')
    mplt.xticks(rotation=90)
    return asBase64(mplt)

def st_get_beta_plot(beta_list):
    # limpa o canvas
    mplt.figure(figsize=(20,5))
    mplt.clf()
    mplt.cla()
    #mplt.close()
    
    try:
        mplt.plot(beta_list, color='limegreen')
    except ValueError:
        mplt.plot([], color='limegreen')
    mplt.xticks(rotation=90)
    
    return st.pyplot(mplt)


def coint_model1(series_x, series_y, periodo):
   
    get_market_data([series_x, series_y], '2y', '1d')
    data = get_market_data([series_x, series_y], '2y', '1d')
    
    market_data = data[-periodo:]

        
    series_x = market_data['Close'][series_x]
    series_y = market_data['Close'][series_y]

    series_x, series_y = clean_timeseries(series_x, series_y)

    x2 = series_x
    y2 = series_y

    try:
        X = sm.add_constant(series_x.values)
        mod = sm.OLS(series_y, X)
        results = mod.fit()
        adfTest = adfuller(results.resid, autolag='AIC')
        lin = linregress(x2,y2)
        lin = lin[0]
        return {
            'OLS': results,
            'ADF': adfTest,
            'Lin': lin,
        }
    except:
        raise



def drop_nan(a):
    return a[~np.isnan(a)]

def clean_timeseries(x, y):
    x, y = drop_nan(x), drop_nan(y),
    intersc = set.intersection(set(x.index), set(y.index))
    newx = x[intersc].sort_index()
    newy = y[intersc].sort_index()
    return newx, newy




