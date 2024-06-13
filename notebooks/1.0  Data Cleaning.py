#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Detection Project
# ##### By Yordanos Simegnew Muche

# #### 1. Importing The Necessary Libraries

# In[1]:


# importing the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
pd.set_option("display.max_columns",None)


# ##### 2. Loading The Dataset and Data Exploration

# In[2]:


# Loading the Dataset
df = pd.read_csv("C:\\Users\\yozil\\Desktop\\My projects\\3.0 brest cancer detection project\\data.csv")


# In[3]:


# displaying sample records
df.sample(3)


# In[4]:


# shape of the dataset
df.shape


# In[5]:


# General information about the dataset
df.info()


# In[6]:


# columns
df.columns


# In[7]:


# number of categorical columns
len(df.select_dtypes("object").columns)


# In[8]:


# we only have one categorical column, let's see this categorical column
df.select_dtypes("object").columns


# In[9]:


# number of numerical columns
len(df.select_dtypes(["int64","float64"]).columns)


# In[10]:


# we have 32 numerical columns and let's see them
df.select_dtypes(["int64","float64"]).columns


# In[11]:


# statistical summary in numerical columns
df.describe()


# #### 3. Data Cleaning

# ##### 3.1 Standardizing column names

# In[12]:


# here we make all column names to be in title case format
def title_maker(column_name):
    return column_name.str.title()


# In[13]:


# now let's apply our title maker function to our column names
df.columns = title_maker(df.columns)


# In[14]:


df.columns


# ##### 3.2 Standardizing Values in the Text columns

# In[15]:


# let's make sure all the text entries are standardized by making them all title case.
def text_maker(df):
    return df.str.title()


# In[16]:


# Now let's apply our text maker function to the text columns.
df[df.select_dtypes("object").columns] =df.select_dtypes("object").apply(text_maker)


# ##### 3.3 Removing Unecessary Space from Values in the text columns(if any) 

# In[17]:


# Now let's make sure there is no unecessary space in the values of text columns
def space_remover(text_column):
    return text_column.str.strip()


# In[18]:


# let's apply the space remover function to our text columns.
df[df.select_dtypes("object").columns] = df.select_dtypes("object").apply(space_remover)


# ##### 3.4 Removing Duplicated Records (if any)

# In[19]:


# first let's check the existance of duplicated records
df.duplicated().any()


# In[20]:


df.duplicated().sum()


# The above result show us there is no duplicated record in our dataset.

# ##### 3.5 Handling Missing Values(if any) 

# In[21]:


# first let's check the existance of missing values in our dataset
df.isnull().any()


# In[22]:


# we have missing value in our last column which is (unnamed: 32), let's see how many missing values are there in this column.
df.isnull().sum()


# In[23]:


# we have 569 null values in this column, we handle this missing values by removing this column from our dataset.
df.drop("Unnamed: 32", axis =1, inplace = True)
df.sample(3)


# In[24]:


# now let's check the shape of our dataset
df.shape


# #### 3.5 Handling an outliers

# In[25]:


# for the case of this project we assign outliers as values above and below 4 standard deviation from the mean
def outlier_limits(col):
    mean = col.mean()
    std = col.std()
    upper_limit = mean + 4 * std
    lower_limit = mean - 4 * std
    return upper_limit, lower_limit


# In[26]:


df.select_dtypes(["int64","float64"]).apply(outlier_limits)


# In[27]:


# now let's see the outlier records
def outlier_records(dataframe):
    outliers_df = pd.DataFrame()
    for col in dataframe.columns:
        mean = dataframe[col].mean()
        std = dataframe[col].std()
        upper_limit = mean + 4 * std
        lower_limit = mean - 4 * std
        outliers_rec = dataframe[(dataframe[col] > upper_limit) | (dataframe[col] < lower_limit)]
        outliers_df = pd.concat([outliers_df, outliers_rec])
    return outliers_df.drop_duplicates()
        


# In[28]:


# first let's see how many outlier records we have.
len(outlier_records(df.select_dtypes(["int64","float64"]),))


# In[29]:


# we have 43 outlier records in the data set let's see them
outlier_records(df.drop("Id",axis = 1).select_dtypes(["int64","float64"]))


# In[30]:


# this are the outlier records in our dataframe, and we handle this outliers by removing them from our dataset.
outliers_index = outlier_records(df.drop("Id",axis = 1).select_dtypes(["int64","float64"])).index
outliers_index


# In[31]:


# Now let's remove this outliers
df.drop(outliers_index,inplace = True)


# In[32]:


# shape of new dataframe
df.shape


# In[33]:


df.info()


# In[34]:


# Now let's fix the index
df.reset_index(inplace = True)


# In[35]:


# sample records
df.sample(3)


# In[36]:


# now let's drop the index columns
df.drop("index", axis = 1, inplace = True)


# In[37]:


# general information
df.info()


# In[38]:


# displaying sample records
df.sample(3)


# In[39]:


# Now let's export our cleaned data
df.to_csv("C:\\Users\\yozil\\Desktop\\My projects\\3.0 brest cancer detection project\\breast cancer cleaned data.csv",index = False)


# In[40]:


pd.read_csv("C:\\Users\\yozil\\Desktop\\My projects\\3.0 brest cancer detection project\\breast cancer cleaned data.csv")


# # Here we are done with the DATA CLEANING task
