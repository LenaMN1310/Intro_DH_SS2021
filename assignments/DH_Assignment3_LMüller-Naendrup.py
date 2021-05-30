#!/usr/bin/env python
# coding: utf-8

# COVID'S IMPACT ON AIRPORT TRAVEL

# I chose and retrieved this dataset from the kaggle website, as it seemed as a suitable dataset for some first data anaylses and visualisation. Through my analysis, I attempt to show how different airports across the globe were affected differently by the covid pandemic's measure in terms of their airpassengeres travel records. 
# 
# For now I mainly played around with the dataset and stuck closely to some of the notebooks provided on the kaggle website. I hope this is not a problem for now. The notebooks really helped me to try out and play around with some data visualisation packages and data analysis functions that were relatively unknown to me before. In the upcoming weeks, I will dive further into the dataest and try to run sum visualizations and analyses myself. 
# 
# Notebooks and dataset from: https://www.kaggle.com/austinpack/covid-impact-on-airport-traffic

# In[57]:


from datetime import date
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from shapely.geometry import Point, Polygon
from shapely.geometry import MultiPolygon
import re


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Reading Dataframe from Excel

# In[3]:


covid_airtraffic_df = pd.read_excel(os.path.join("../Intro_DH/covid_airtraffic_data.xlsx"))


# First Data Examination

# In[4]:


covid_airtraffic_df.head(9)


# In[5]:


covid_airtraffic_df.tail(4)


# In[6]:


covid_airtraffic_df.info()


# PREPROCESSING

# 1. Creating Weekdays

# In[18]:


covid_airtraffic_df["Weekday"] = covid_airtraffic_df["Date"].map(lambda x: x.weekday())

week_list = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

covid_airtraffic_df["Weekday"] = [week_list[idx] for idx in covid_airtraffic_df["Weekday"]]


# In[30]:


def cut_long(point):
    
    long, _ = point[6:-1].split(" ")
    return float(long)

def cut_lat(point):
    _ , lat = point[6:-1].split(" ")
    return float(lat)


# In[32]:


covid_airtraffic_df["long"] = covid_airtraffic_df["Centroid"].map(cut_long)

covid_airtraffic_df["lat"] = covid_airtraffic_df["Centroid"].map(cut_lat)


# 2. Dropping some irrelevant columns

# In[34]:


drop_columns = [col for col in covid_airtraffic_df.columns if not col in["AggregationMethod", "Version", "Centroid"]]

covid_airtraffic_df = covid_airtraffic_df[drop_columns]


# Displaying the DF with the new structure

# In[35]:


covid_airtraffic_df.info()


# In[36]:


#info shows that I removed Aggregation Method, Version and Centroid column (replaced with long and lat)


# 3. Deleting duplicate airport names

# In[39]:


covid_airtraffic_new = covid_airtraffic_df[~ covid_airtraffic_df[["AirportName"]].duplicated()].reset_index(drop=True)
covid_airtraffic_new


# FIRST DESCRIPTIVE STATISTICS

# In[43]:


covid_airtraffic_new.describe()


# In[44]:


covid_airtraffic_new = covid_airtraffic_new.round(1)


# In[45]:


covid_airtraffic_new.describe().round(1)


# FIRST VISUALISATIONS USING FOLIUM

# In[46]:


def visualize_airport_map(df, zoom):
    lat_map = 30.038557
    lon_map = 31.231781
    
    f = folium.Figure(width=1000, height=500)
    m = folium.Map([lat_map, lon_map], zoom_start=zoom).add_to(f)
    
    for i in range(0, len(df)):
        folium.Marker(location=[df["lat"][i], df["long"][i]], icon=folium.Icon(icon_color ="Red", icon="plane", prefix="fa")).add_to(m)
        
        return m


# In[47]:


airport_impact_map = visualize_airport_map(covid_airtraffic_new,1)


# In[48]:


airport_impact_map


# In[ ]:


#I dont know why the map only shows one singel airport as opposed to the whole list


# Further Analysis: Which airports have the most records in the dataset?

# In[49]:


plt.figure(figsize = (10,5))

g = sns.countplot(data = covid_airtraffic_df, x = "AirportName", order = covid_airtraffic_df["AirportName"].value_counts().index)

g.set_xticklabels(g.get_xticklabels(), rotation = 90)

g.set_title("Airport Records")


# Next Step: Checking Records for each month

# In[51]:


df_month = pd.DataFrame(covid_airtraffic_df["Date"].map(lambda d: d.month).value_counts())

df_month = df_month.reset_index()
df_month = df_month.rename(columns ={"Date":"count", "index":"month"})

g = sns.barplot(data = df_month.reset_index(), y = "count", x = "month")

g.set_xticklabels(g.get_xticklabels(), rotation = 90)
g.set_title("Monthly Labels")


# Next Step: Weekly Records

# In[53]:


df_week = pd.DataFrame(covid_airtraffic_df["Weekday"].value_counts())

g = df_week.plot.pie(y = "Weekday", figsize = (7,7))

g.set_title("Weekly Records")


# MAPPING AND VISUALISATIONS USING GEOPANDAS

# In[58]:


gdf = gpd.GeoDataFrame(covid_airtraffic_new, geometry = gpd.points_from_xy(covid_airtraffic_new.long, covid_airtraffic_new.lat))


# In[59]:


gdf.head()


# In[60]:


world_map = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

world_map.head()


# 1. Example: Basic World Map Geopandas

# In[61]:


g = world_map.plot(color = "white", edgecolor = "gray")

g.set_title("World Map")


# 2. Example: World Map with markers from DF

# In[62]:


ax = world_map.plot(color = "white", edgecolor = "gray", figsize = (15,10))

g =gdf.plot(ax = ax, marker = "*", color = "green", markersize = 50)

g.set_title("Map with Markers")

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




