import streamlit as st
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
import ibov
from funcoes import coint_model, get_market_data, half_life, get_beta_plot, _get_residuals_plot, beta_rotation, asBase64, fp_savefig, st_get_residuals_plot, st_get_beta_plot,beta_rotation1, clean_timeseries, drop_nan, coint_model1


st.title("Análise de Cointegração")

sidebar = st.sidebar.header('Seleção de Ações')
seriesy = st.sidebar.selectbox('Dependente', ibov.CARTEIRA_IBOV ) + '.SA'
qtdc = st.sidebar.number_input('Quantidade da dependente',
       min_value = 0,
       max_value = 1000,
       value = 0,
       step = 1
)
seriesx = st.sidebar.selectbox('Independente', ibov.CARTEIRA_IBOV ) + '.SA'
periodo1 = st.sidebar.slider("Periodo", 100, 260, 260, 20)

#period = '2y'
#interval = '1d'

#data = get_market_data([series_x, series_y], period, interval)
#market_data = data[periodo1:]
#market_data = market_data.dropna()
#series_x = market_data['Close'][series_x]
#series_y = market_data['Close'][series_y]

if st.sidebar.button('Calcular'):

  data = get_market_data([seriesx, seriesy], '2y', '1d')
  market_data = data[-periodo1:]
  
  series_x = market_data['Close'][seriesx]
  series_y = market_data['Close'][seriesy]

  series_x, series_y = clean_timeseries(series_x, series_y)

  
  coint = coint_model(series_x, series_y)
  Adfr = coint['ADF']
  residuo = (coint['OLS']).resid
  stddev = (coint['OLS']).resid.std()
  media = (coint['OLS']).resid.median()
  stdmax = media + (stddev * 1.96)
  stdmin = media - (stddev * 1.96)
  lin = coint['Lin']
  qcd = qtdc
  qcin = qcd * lin
  beta_rot = beta_rotation(series_x, series_y, window=40)
  half_life, _ = half_life(coint['OLS'].resid)
  vl = qcd * series_y.iloc[-1]
  vl2 = qcin * series_x.iloc[-1]

  ADF = Adfr[0]

  if (ADF < -3.45):
    adfperc = '99%'
  elif (ADF < -2.87):
    adfperc = '95%'
  elif (ADF < -2.57):
    adfperc = '90%'
  else:
    adfperc = '0%'    

  st.write('Teste ADF:', adfperc)
  st.write('Residuo:',residuo.iloc[-1])
  if (residuo.iloc[-1] > 0):
    st.write('Desvio max:', stdmax)
    st.write('Vender',qcd, seriesy,'e Comprar',qcin, seriesx) 
  elif (residuo.iloc[-1] < 0) :  
    st.write('Desvio min:', stdmin)
    st.write('Comprar',qcd, seriesy,'e Vender',qcin, seriesx)
  st.write('Meia vida: ', half_life)
  st.write('Coef. Ang.:', lin)
  st.write('R$',vl)

  
  graf = st_get_residuals_plot(coint['OLS'])
  graf1 = st_get_beta_plot(beta_rot)   

  st.subheader('Periodos Cointegrados')
  periodos = list(range(20, 280 ,20))
  for periodo in periodos:
    coint = coint_model1(seriesx, seriesy, periodo)
    adfr = coint['ADF']
    if (adfr[0] < -3): 
      #st.write(x, y)
      st.write('ADF:', adfr[0],
      'Periodo:', periodo)
 




