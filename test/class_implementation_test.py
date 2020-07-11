import pandas as pd
import numpy as np

import nostradamus.preprocessing as ns_preprocessing
import nostradamus.models.arimanostra as models


data= pd.read_excel(r"superstore.xls")

# Get information
furniture= data[data['Category'] == 'Furniture']
furniture= furniture[["Order Date", "Sales"]]

# Transform into Datetime object
furniture["Order Date"]= pd.to_datetime(furniture["Order Date"], format= "%Y-%m-%d")
# Order by date
furniture= furniture.sort_values(["Order Date"]).reset_index(drop= True)
# Index
furniture= furniture.set_index("Order Date")

# Sales mean / max / min per month (3 series)
furniture_mean= furniture["Sales"].resample('MS').mean().values
furniture_max= furniture["Sales"].resample('MS').max().values
furniture_min= furniture["Sales"].resample('MS').min().values


## tunnelSnake testing
#test= ns_preprocessing.tunnelSnake(furniture_min, 3, 0.5)
#print(test.fit_transform())
#test.plot()

# ARIMANOSTRA test model
arima_model= models.ArimaNostra(furniture_mean, [2, 1, 1], 24, 3)
arima_fit= arima_model.forecast()["mean"]
print(arima_fit)

arima_model.plot(arima_fit)