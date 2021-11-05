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
from funcoes import coint_model, get_market_data, half_life, get_beta_plot, _get_residuals_plot, beta_rotation, asBase64, fp_savefig, st_get_residuals_plot, st_get_beta_plot,beta_rotation1, clean_timeseries, drop_nan, coint_model1, gera_pares, download_hquotes, coint_model2

pag = st.sidebar.selectbox('Escolha uma Opção',['Análise','Buscar pares'])


if pag == 'Buscar pares':
  st.title('Buscar pares')
  
  #if st.sidebar.button('Gerar Pares'):


  #if st.sidebar.button('Atualizar Dados'):
 

  if st.sidebar.button('Buscar Pares'):
    
    ibrx_tickers = [ "%s.SA" % s for s in ibov.CARTEIRA_IBOV]    
    pares = gera_pares(ibrx_tickers)
    pares1 = list(pares)

    
    ibrx_tickers = [ "%s.SA" % s for s in ibov.CARTEIRA_IBOV]
    q = download_hquotes(ibrx_tickers)
    qu = q.drop(columns=['Volume','Open','Low', 'High', 'Adj Close'])
    quo = qu.reset_index(0)
    quot = quo.drop(columns=['Date'])
    quote = quot
   
    periodos = [-220, -240, -260]
    for par in pares1:
      for periodo in periodos:
        coint = coint_model2(par[0], par[1], periodo, quote)
        Adfr = coint['ADF']
        hl, _ = half_life((coint['OLS']).resid)
        residuo = (coint['OLS']).resid
        stddev = (coint['OLS']).resid.std()
        media = (coint['OLS']).resid.median()
        stdmax = media + (stddev * 1.96)
        stdmin = media - (stddev * 1.96)

          
        if (Adfr[0] < -3.5) and (residuo.iloc[-1] > stdmax) and (coint['Lin'] > 0): 
          lst = [[par[1], par[0], Adfr[0], hl, periodo, Adfr[3], residuo.iloc[-1], stdmax, coint['Lin']]]
          df = pd.DataFrame(lst, columns = ['Dependente', 'Independente', 'ADF', 'Meia vida', 'Periodo', 'Periodo analisado', 'Residuo', 'Desvio', 'Coef. Ang.'] )
          st.dataframe(df)  
            
        elif (Adfr[0] < -3.5) and (residuo.iloc[-1] < stdmin) and (coint['Lin'] > 0):
          lst = [[par[1], par[0], Adfr[0], hl, periodo, Adfr[3], residuo.iloc[-1], stdmin, coint['Lin']]]
          df = pd.DataFrame(lst, columns = ['Dependente', 'Independente', 'ADF', 'Meia vida', 'Periodo', 'Periodo analisado', 'Residuo', 'Desvio', 'Coef. Ang.'] )
          st.dataframe(df)
      



if pag == 'Análise':
  st.title("Análise de Cointegração")
  sidebar = st.sidebar.header('Seleção de Ações')
  seriesy = st.sidebar.selectbox('Dependente', ibov.CARTEIRA_IBOV ) + '.SA'
  qtdc = st.sidebar.slider('Quantidade da Dependente',1,1000,100,1)
  seriesx = st.sidebar.selectbox('Independente', ibov.CARTEIRA_IBOV ) + '.SA'
  periodo1 = st.sidebar.slider("Periodo", 100, 260, 260, 20)

 
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

    if (residuo.iloc[-1] > 0):
      std = f'{stdmax: .2f}'
    elif (residuo.iloc[-1] < 0) :
      std = f'{stdmin: .2f}'

    res = [[adfperc, f'{(residuo.iloc[-1]): .2f}', std, f'{half_life: .2f}', f'{lin: .2f}']]  
    cjt = pd.DataFrame(res, columns = ['Teste ADF', 'Residuo', 'Desvio', 'Meia vida', 'Coef. Ang.'])

    st.dataframe(cjt)


    if (residuo.iloc[-1] > 0):
      st.write('Vender',f'{qcd: .0f}', seriesy, f'No valor de R${vl: .2f}') 
      st.write('Comprar',f'{qcin: .0f}', seriesx, f'No valor de R${vl2: .2f}')
    elif (residuo.iloc[-1] < 0) :
      st.write('Comprar',f'{qcd: .0f}', seriesy, 'No valor de R$',f'{vl: .2f}')
      st.write('Vender',f'{qcin: .0f}', seriesx, f'No valor de R${vl2: .2f}')

    st.subheader('Gráfico do Residuo')  
    graf = st_get_residuals_plot(coint['OLS'])
    st.write('Beta Rotation:', f'{((beta_rot[0]) * 10): .2f}', "%")
    st.subheader("Gráfico do Beta Rotation")
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
  




