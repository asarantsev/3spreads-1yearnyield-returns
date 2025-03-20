import numpy as np
import pandas as pd
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.api import OLS

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
    print('Skewness:', round(scipy.stats.skew(data), 3))
    print('Kurtosis:', round(scipy.stats.kurtosis(data), 3))
    print('Shapiro-Wilk p = ', round(100*scipy.stats.shapiro(data)[1], 1), '%')
    print('Jarque-Bera p = ', round(100*scipy.stats.jarque_bera(data)[1], 1), '%')
    print('Autocorrelation function analysis for ' + label)
    L1orig = sum(abs(acf(data, nlags = 5)[1:]))
    print('\nL1 norm original residuals ', round(L1orig, 3), label, '\n')
    L1abs = sum(abs(acf(abs(data), nlags = 5)[1:]))
    print('L1 norm absolute residuals ', round(L1abs, 3), label, '\n')
    
df = pd.read_excel("data.xlsx", sheet_name = 'data')
vol = df["Volatility"].values[1:]
spreads = pd.DataFrame({})
spreads['BAA-AAA'] = df['BAA'] - df['AAA']
spreads['AAA-Long'] = df['AAA'] - df['Long']
spreads['Long-Short'] = df['Long']-df['Short']

for key in spreads:
    plt.plot(range(1927, 2025), spreads[key], label = key)
plt.xlabel('Years')
plt.ylabel('Spreads')
plt.title('Spread Plot')
plt.legend(bbox_to_anchor=(0.05, 0.25), loc='upper left')
plt.savefig('spreads.png')
plt.close()

DFreg = pd.DataFrame({'const' : 1/vol, 'vol' : 1})

for key in spreads:
    DFreg[key] = spreads[key].iloc[:-1]/vol

for key in spreads:
    print('Regression with normalization', key)
    Reg = OLS(list(np.diff(spreads[key])/vol), DFreg).fit()
    print(Reg.summary())
    resid = Reg.resid
    plots(resid, key)
    analysis(resid, key)