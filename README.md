# pyVaRES_functions

In this repo you can find some functions I've written, related to the topics of Value at Risk and Expected Shortfall.

The **pyVaRES.py** file contains the functions I've written:
- VaR_basic = function to compute fixed or rolling VaR given input parameters such as underlying distribution, alpha level, etc...
- ES_basic = function to compute fixed or rolling ES given input parameters such as underlying distribution, alpha level, etc...
- garch11_vol_fixedspec = function to fit a garch(1,1) model and obtain volatility forecasts given the specific characteristics of the given returns distribution.
- pf_gaussianVaRES = function to compute portfolio VaR and ES.
- pf_factVaRES = function to compute portfolio VaR and ES using risk-factors mapping technique.
- VaR_validation = function to validate a given VaR model
- VEV = function to compute VaR-Equivalent-Volatility

In the **pyVaR_Examples** notebook you can find some examples on how the library can be used and what it can do. The dataset used in this notebook is contained in the Dataneeded folder.
