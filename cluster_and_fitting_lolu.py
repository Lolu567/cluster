#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:28:22 2023

@author: Adeolu Adelana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import err_ranges as err
from sklearn.cluster import KMeans
from scipy import interpolate



def data_extractor(url, columns_to_delete, rows_to_skip, indicator):
    '''
    

    Parameters
    ----------
    url : String
        A string of the url for the dataset.
    columns_to_delete : list
        a list of strings which is the columns that are not needed.
    rows_to_skip : integer
        determins the number of columns to skip as identified.
    indicator : string
        a string which is the indicator to be considered.

    Returns
    -------
    df1 : dataframe
        The original dataframe.
    df2 : string
        The transposed dataframe.

    '''
  
    df = pd.read_csv(url, skiprows=rows_to_skip)
    df = df.loc[df['Indicator Name'] == indicator]
  
    #dropping columns that are not needed. Also, rows with NA were dropped
    df = df.drop(columns_to_delete, axis=1)

    #this extracts a dataframe with countries as column
    df1 = df
    
    #this section extract a dataframe with year as columns
    df2 = df.transpose()

    #removed the original headers after a transpose and dropped the row
    #used as a header
    df2 = df2.rename(columns=df2.iloc[0])
    df2 = df2.drop(index=df2.index[0], axis=0)
    df2['Year'] = df2.index
    #df2 = df2.rename(columns={"index":"Year"})
    
    return df1, df2


cluster_year = ["1980", "2020"]
url = 'API_19_DS2_en_csv_v2_4700503.csv'
columns_to_delete = ['Country Code', 'Indicator Name', 'Indicator Code']
rows_to_skip = 3
year = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', 
        '2017', '2018', '2019', '2020', '2021']
df1, df2 = data_extractor(url, columns_to_delete, rows_to_skip, 
                          indicator='Population growth (annual %)')

df_clustering = df1.loc[df1.index, ["Country Name", "1980", "2020"]]

#extract data of interest
x = df_clustering
x = x.dropna().values
plt.figure()
plt.scatter(x[:,1], x[:,2], label='data')
plt.xlabel('2020')
plt.ylabel('1980')
plt.title('Original data')
plt.legend()
plt.savefig('ade-raw_data.png')
plt.show()


#determine the elbow which is to be used as the cluster
SSE = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x[:,1:2])
    SSE.append(kmeans.inertia_)
    
plt.figure(figsize=(15, 11))
plt.plot(range(1, 11), SSE, label='cluster')
plt.xlabel('Number of clusters', fontsize=30)
plt.ylabel('SSE', fontsize=30)
plt.title('Elbow Method', fontsize=30)
plt.tick_params(axis='both', labelsize=30)
plt.legend(fontsize=30)
plt.savefig('ade-number of cluster.png')
plt.show()



# Create a model based on 2 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)

# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(x[:,1:2])

df_clustered = df_clustering[cluster_year]
df_clustered = df_clustered.dropna()
df_clustered['classification'] = km_clusters.tolist()
print(df_clustered)

def centriods(data, axis, clusters):
    return [data.loc[data["classification"] == c, axis].mean() 
            for c in range(clusters)]

centriod_x = centriods(df_clustered, cluster_year[0], 3)
centriod_y = centriods(df_clustered, cluster_year[1], 3)

n_array = np.array([centriod_x, centriod_y])
df_centriod = pd.DataFrame(n_array)
df_centriod = df_centriod.T
print(df_centriod)

def plot_clusters(samples, clusters):
    '''
    

    Parameters
    ----------
    samples : array
        an arrary of the data to be used for the clustering.
    clusters : array
        an array of the custers.

    Returns
    -------
    None.

    '''
    col_dic = {0:'blue',1:'green',2:'cyan'}
    mrk_dic = {0:'*',1:'+',2:'.'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][1], samples[sample][2], 
                    color = colors[sample], marker=markers[sample], s=50)
    plt.scatter(df_centriod[0], df_centriod[1], c='black', marker='x', s=100)
    plt.xlabel(cluster_year[1])
    plt.ylabel(cluster_year[0])
    plt.title('Cluster of Population Growth')
    plt.savefig('ade-clustered_data.png')
    plt.show()

plot_clusters(x, km_clusters)


# define the true objective function
def objective(x, a, b, c, d):
    '''
        This is the model that would be used to determin the line of best fit
    
    '''
    x = x - 2008.0
    return a*x**3 + b*x**2 + c*x + d

#get dataset
df_fitting = df2.loc[year, ["Year", "Cameroon"]].apply(pd.to_numeric, 
                                                       errors='coerce')
x = df_fitting.dropna().to_numpy()

# choose the input and output variables
x, y = x[:, 0], x[:, 1]

# curve fit
popt, _ = opt.curve_fit(objective, x, y)


# summarize the parameter values
a, b, c, d = popt
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))

param, covar = opt.curve_fit(objective, x, y)

sigma = np.sqrt(np.diag(covar))
low, up = err.err_ranges(x, objective, popt, sigma)

print(low, up)
print('covar = ', covar)

# plot input vs output
plt.figure(figsize=(15, 11))
plt.plot(x, y)

# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(min(x), max(x) + 1, 1)


# calculate the output for the range
y_line = objective(x_line, a, b, c, d)
# Use interp1d to create a function for interpolating y values
interp_func = interpolate.interp1d(x_line.flatten(), y_line)

#initialize a list to add the population growth predictions
prediction = [] 
predict = float(format(interp_func(2019), '.3f'))
prediction.append(predict)

# print out the value of the prediction
print('prediction', prediction)



# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
plt.scatter(2019, prediction, marker=('x'), s=200, color='black', 
            label=f"pop growth of {2019} growth is {predict}.")
#plt.savefig('curve_fit.png')
plt.xlabel('years', fontsize=30)
plt.ylabel('population growth', fontsize=30)
plt.title('curve fit', fontsize=30)
plt.legend(fontsize=30)
plt.savefig('ade-curve_fit.png')
plt.show()

plt.figure(figsize=(15, 11))
plt.plot(x, y, label='pop growth')
plt.plot(x_line, y_line, '--', color='red', label='curve fit')
plt.fill_between(x, low, up, alpha=0.7, color='blue', label='error range')
plt.xlabel('years', fontsize=30)
plt.ylabel('population growth', fontsize=30)
plt.title('curve fit with error range', fontsize=30)
plt.legend(fontsize=30)
plt.savefig('ade-with error ranges.png')
plt.show()
