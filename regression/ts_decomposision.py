from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data, model='multiplicative')
trend    = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid