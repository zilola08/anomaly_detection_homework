# Задача по большей части на research
# Этот датасет представляет собой данные собранные с 1 сервера за длительное время, нужно узнать какие timestamp в этом датасет являются аномальными, алгоритм без тренера

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# adtk -> Anomaly Detection for Time-series
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import *

df = pd.read_csv("dataset.csv", sep=",", index_col=0)
data = df.copy()
# print(data.head())
print(data.columns)
# print(len(data))
# print(data.dtypes)

# Convert timestamp from object to datetime data type:
data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.set_index("timestamp")

# Check correlation coefs between variables
# print(data.corr())
#                       ram_value_percentage  cpu_usage_percent  io_usage_percent
# ram_value_percentage              1.000000           0.009487          0.014771
# cpu_usage_percent                 0.009487           1.000000          0.534394
# io_usage_percent                  0.014771           0.534394          1.000000

# RAM usage has almost no correlation with the other two variables, hence it has no relationship with the other two variables.
# Drop RAM usage
data = data.drop("ram_value_percentage", axis=1)

# Normalizing data
# Min-max normalization
data=(data-data.min())/(data.max()-data.min())

# Mean normalization
# data=(data-data.mean())/data.std()
# print(data)

# Quantile detector
# This method looks at the variables separately. It is only useful if we want to know when datapoints in one variable are going put of the quantiles as anomalies.
quantile_detector = QuantileAD(low=0.00001, high=0.9999)
quantile_anomalies = quantile_detector.fit_detect(data)
plot(data, anomaly = quantile_anomalies, anomaly_color="red", anomaly_tag="marker")
plt.title("Quantile Anomaly detector")
plt.show()


# MinClusterDetector 
# It treats multivariate time series as independent points in a high-dimensional space, divides them into clusters, and identifies values in the smallest cluster as anomalous. This may help capturing outliers in high-dimensional space.

from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans
data1 = data.copy()
min_cluster_detector = MinClusterDetector(KMeans(random_state=0, n_init='auto'))
mincluster_anomalies = min_cluster_detector.fit_detect(data1)
plot(data1, anomaly=mincluster_anomalies, anomaly_color='red', anomaly_tag="marker", anomaly_alpha=0.3, curve_group='all');
plt.title("MinCluster Anomaly Detector")
plt.show()

data1 = data1.assign(mincluster_anomalies=mincluster_anomalies)
mincluster_anomalies_count = 0
mincluster_anomalies_array = []
for i in range(len(data)):
  if mincluster_anomalies[i] == True:
    mincluster_anomalies_count +=1
    # mincluster_anomalies.append(data["timestamp"][i])

mincluster_anomalies_perc = (mincluster_anomalies_count/len(data1))*100
# print('mincluster anomalies: ',mincluster_anomalies)
print('mincluster anomalies count: ',mincluster_anomalies_count)
print('mincluster anomalies percentage: ',mincluster_anomalies_perc)

# Isolation Forest
# Contamination: It's the most important parameter of IF. The amount of contamination of the data set, i.e. the proportion of outliers in the data set. It defines the threshold on the scores of the samples. By default, it is 'auto' (0.1), but we can also set a value.
# HERE WE NEED TO KNOW WHAT PERCENTAGE OF DATA IS GOING TO BE ANOMALIES BEFOREHAND AND SPECIFY IT IN THE FUNCTION!
# Here I just picked the contamination level by trying different numbers. There must be better ways to find out the best contamination levels, but I could not implement them, because I couldn`t understand how they work.`

import numpy as np
from sklearn.ensemble import IsolationForest
contamination = 0.0008

data3 = data.copy()
iso_forest = IsolationForest(contamination=contamination, random_state=42)
iso_output = iso_forest.fit_predict(data3)
iso_output = np.where(iso_output<0, 1, 0)
# unique, counts = np.unique(iso_output, return_counts=True)
# print(np.asarray((unique, counts)).T)
iso_anomalies = pd.DataFrame(iso_output, index=data3.index.copy())
plot(data3, anomaly=iso_anomalies, anomaly_color='red', anomaly_alpha=0.3, anomaly_tag="marker", curve_group='all')
plt.title("Isolation Forest Anomaly Detection")
plt.show()

data3 = data3.assign(iso_anomalies=iso_anomalies)
data3["iso_anomalies"] = data3["iso_anomalies"]
iso_anomaly_count = 0
iso_anomalies = []
for i in range(len(data3)):
  if data3["iso_anomalies"][i] == 1:
    iso_anomaly_count +=1
    iso_anomalies.append(data3["iso_anomalies"][i])
iso_anomalies_perc = (iso_anomaly_count/len(data3))*100
# print('isolation forest anomalies: ', iso_anomalies)
print('isolation forest anomalies count: ', iso_anomaly_count)
print('isolation forest anomalies percentage: ',iso_anomalies_perc)


