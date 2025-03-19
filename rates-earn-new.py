import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data, label):
    print(label + ' analysis of residuals normality')
    print('Skewness:', round(stats.skew(data), 3))
    print('Kurtosis:', round(stats.kurtosis(data), 3))
    print('Shapiro-Wilk p = ', round(100*stats.shapiro(data)[1], 1), '%')
    print('Jarque-Bera p = ', round(100*stats.jarque_bera(data)[1], 1), '%')
    print('Autocorrelation function analysis for ' + label)
    L1orig = sum(abs(acf(data, nlags = 5)[1:]))
    print('\nL1 norm original residuals ', round(L1orig, 3), label, '\n')
    L1abs = sum(abs(acf(abs(data), nlags = 5)[1:]))
    print('L1 norm absolute residuals ', round(L1abs, 3), label, '\n')

df = pd.read_excel("data.xlsx", sheet_name = 'data')
vol = df["Volatility"].values[1:]
N = len(vol)
price = df['Price'].values
dividend = df['Dividends'].values
earnings = df['Earnings'].values
cpi = df['CPI'].values
inflation = np.diff(np.log(cpi))
nearngr = np.diff(np.log(earnings))
rearngr = nearngr - inflation
earnyield = earnings/price
ngrowth = nearngr/vol
rgrowth = rearngr/vol
spreads = {'BAA-AAA': df['BAA'] - df['AAA'], 'AAA-Long': df['AAA'] - df['Long'], 'Long-Short': df['Long'] - df['Short'], 'const' : np.ones(N+1)}
DForig = pd.DataFrame(spreads).iloc[:-1]

print('Regression without normalization of factors')
RegNGrowth = OLS(ngrowth, DForig).fit()
print('Nominal Earnings Growth')
print(RegNGrowth.summary())
resngrowth = RegNGrowth.resid
plots(resngrowth, 'Nominal Growth Original Spreads')
analysis(resngrowth, 'Nominal Growth Original Spreads')

RegRGrowth = OLS(rgrowth, DForig).fit()
print('Real Earnings Growth')
print(RegRGrowth.summary())
resrgrowth = RegRGrowth.resid
plots(resrgrowth, 'Real Growth Original Spreads')
analysis(resrgrowth, 'Real Growth Original Spreads')

DFnorm = pd.DataFrame({})
print('Regression with normalization of factors')
for key in spreads:
    DFnorm[key] = spreads[key][:-1]/vol
DFnorm['vol'] = np.ones(N)

RegNGrowth = OLS(ngrowth, DFnorm).fit()
print('Nominal Earnings Growth')
print(RegNGrowth.summary())
resngrowth = RegNGrowth.resid
plots(resngrowth, 'Nominal Growth Norm Spreads')
analysis(resngrowth, 'Nominal Growth Norm Spreads')

RegRGrowth = OLS(rgrowth, DFnorm).fit()
print('Real Earnings Growth')
print(RegRGrowth.summary())
resrgrowth = RegRGrowth.resid
plots(resrgrowth, 'Real Growth Norm Spreads')
analysis(resrgrowth, 'Real Growth Norm Spreads')