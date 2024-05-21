
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import random
import cvxopt as opt
from cvxopt import blas, solvers
import seaborn as sns

start_date = '2004-01-01'
end_date = '2024-01-01' 

tickers = []
# ESTADOS UNIDOS
Dow_Jones = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT',
    'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WBA', 'WMT', 'DIS']

# DAX; ALEMANIA
Dax = [
    'ADS.XE', 'AI.PA', 'ALV.XE', 'BAS.XE', 'BAYN.XE', 'BMW.XE', 'BRT.XE', 'CON.XE', 'DAI.XE',
    'DBK.XE', 'DPW.XE', 'DTE.XE', 'EON.XE', 'FME.XE', 'HEN3.XE', 'IFX.XE', 'LIN.XE', 'MRK.XE',
    'MUV2.XE', 'RWE.XE', 'SAP.XE', 'SIE.XE', 'VOW.XE']

# IBEX 35: ESPAÑA
Ibex = ['ANA.MC', 'ACX.MC', 'ACS.MC', 'AENA.MC', 'AIR.MC', 'ALM.MC', 'MTS.MC', 'BKIA.MC',
    'BBVA.MC', 'CABK.MC', 'CLNX.MC', 'CIE.MC', 'COL.MC', 'DIA.MC', 'ENG.MC', 'ELE.MC',
    'FER.MC', 'FLDR.MC', 'GRF.MC', 'IAG.MC', 'IBE.MC', 'ITX.MC', 'IDR.MC', 'INM.MC',
    'MAP.MC', 'MEL.MC', 'NTGY.MC', 'REE.MC', 'REP.MC', 'SAN.MC', 'SGRE.MC', 'TEF.MC','VIS.MC',]

# CAC 40; FRANCIA
Cac = ['AI.PA', 'AIR.PA', 'ALSO.PA', 'MT.AS', 'BNP.PA', 'BOUY.PA', 'CAP.PA', 'CA.PA', 'CO.PA',
    'CDI.PA', 'CNP.PA', 'CBK.XE', 'CRDI.PA', 'DANO.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA', 'RMS.PA', 
    'KER.PA', 'OR.PA', 'BNP.PA', 'LYB.PA', 'ML.PA', 'ORA.PA', 'PEUG.PA', 'PUB.PA', 'RNO.PA',
    'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'SGO.PA', 'STLA.PA', 'TTE.PA', 'URW.PA',
    'VIE.PA', 'VIN.PA', 'VIV.PA',]
tickers += Dow_Jones + Dax + Ibex + Cac

elementos_a_eliminar = ['RWE.XE', 'HEN3.XE', 'CRDI.PA', 'STLA.PA', 'SAP.XE', 'DTE.XE', 'DAI.XE', 'CBK.XE', 'BKIA.MC', 'BAS.XE', 'DPW.XE', 
              'FME.XE', 'ALSO.PA', 'CNP.PA', 'INM.MC', 'EON.XE', 'SGRE.MC', 'BMW.XE', 'ALV.XE', 'REE.MC', 'FLDR.MC', 'VOW.XE', 
              'MUV2.XE', 'BRT.XE', 'BAYN.XE', 'DBK.XE', 'SIE.XE', 'LYB.PA', 'MRK.XE', 'ADS.XE', 'VIN.PA', 'CON.XE', 'IFX.XE', 'LIN.XE',
              'DANO.PA', 'BOUY.PA']
for i in elementos_a_eliminar:
    tickers.remove(i)

# 86

stocks_data = yf.download(tickers, start=start_date, end=end_date)
print(stocks_data)
rtos = stocks_data['Adj Close'].pct_change()
rtos = rtos.fillna(0)
print(rtos)


rtos_esperados = rtos.mean()
print(rtos_esperados)
risk = rtos.cov()
#print(risk)

RE_diarios = rtos_esperados.sort_values(ascending =False).dropna()
RE = RE_diarios.head(100)
#print(RE)

optimal_portfolio_by_returns = RE.head(100).keys()
#print(optimal_portfolio_by_returns)
#print(rtos[optimal_portfolio_by_returns].corr())


## CALCULAR RIESGO:
RISK_E = stocks_data['Adj Close'].std().sort_values(ascending = True)
RISK_E2 = RISK_E.mean()
RISK_E1 = RISK_E[optimal_portfolio_by_returns]
#print(RISK_E1)

aa  = RISK_E1.to_numpy().round(2)
#print(aa)


cartera_seleccionada = ['AAPL', 'SAF.PA','ITX.MC', 'VIV.PA', 'CIE.MC']
selec = RE_diarios[cartera_seleccionada]
print(selec)

'''Appl_por = 0.463
Cie_por = 0.175
Itx_por = 0.227
Saf_por = 0.0056
Viv_por = 0.129


RE_total_cartera = (selec['AAPL'] * Appl_por) + (selec['SAF.PA'] * Saf_por) + (selec['ITX.MC'] * Itx_por) + (selec['VIV.PA'] * Viv_por) + (selec['CIE.MC'] * Cie_por) 

print('El retorno total de la cartera en estos 20 años ha sido de:'+ RE_total_cartera*100*260*20)'''
