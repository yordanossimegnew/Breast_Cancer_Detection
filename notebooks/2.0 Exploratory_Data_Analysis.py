#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)


# In[3]:


# Loading the Data Frame
df = pd.read_csv("C:\\Users\\yozil\\Desktop\\My projects\\3.0 Breast_Cancer_Detection\\Breast_Cancer_Detection\\data\\processed\\breast cancer cleaned data.csv")


# In[4]:


df.sample(2)


# In[5]:


df.info()


# In[6]:


df.describe(
)


# #### Univariate Analysis

# In[7]:


# univariate analysis on diagnosis_variable
plt.figure(figsize=(6,3))
sns.countplot(data = df, x = "Diagnosis")
plt.title("Count of Diagnosis variable")
plt.savefig("C:\\Users\\yozil\\Desktop\\My projects\\3.0 Breast_Cancer_Detection\\Breast_Cancer_Detection\\models\\1.0 Count_of_Diagnosis_Variable.jpg")
plt.show()


# As we can see from the above plot:
# 1. in our dataset we have more of benign groups than those of malignants.

# In[8]:


# Univariate analysis on radius mean variable
plt.figure(figsize=(6,3))
sns.distplot(df.Radius_Mean)
plt.title("Distribution_of_Radius_Mean_Variable")
plt.savefig("C:\\Users\\yozil\\Desktop\\My projects\\3.0 Breast_Cancer_Detection\\Breast_Cancer_Detection\\models\\2.0 Distribution_of_Radius_Mean_Variable.jpg")
plt.show()


# As we can see from the above distplot:
# 1. slightly more records have a "Radius RMean" value less than that of the mean.
# 2. most of the records have a "Radius_Mean" value between 10 and 20

# In[9]:


# Univariate analysis on Texture mean variable
plt.figure(figsize=(6,3))
sns.distplot(df.Texture_Mean)
plt.title("Distribution_of_Texture_Mean_Variable")
plt.savefig("C:\\Users\\yozil\\Desktop\\My projects\\3.0 Breast_Cancer_Detection\\Breast_Cancer_Detection\\models\\3.0 Distribution_of_Texture_Mean_Variable.jpg")
plt.show()


# As we can see from the texture mean variable distribution plot above:
# 1. The distrubution plot  shows a bell-shaped curve, which is characteristic of a normal distribution.
# 2. The distribution appears to be fairly symmetric around the center.
# 3. The data points are spread between approximately 5 and 35, with the highest density around the center.

# In[10]:


# Univariate analysis on Perimeter mean variable
plt.figure(figsize=(6,3))
sns.distplot(df.Perimeter_Mean)
plt.title("Distribution_of_Perimeter_Mean_Variable")
plt.savefig("C:\\Users\\yozil\\Desktop\\My projects\\3.0 Breast_Cancer_Detection\\Breast_Cancer_Detection\\models\\4.0 Distribution_of_Periemeter_Mean_Variable.jpg")
plt.show()


# As we can see from the above distribution plot of Perimeter mean variable
# 1. The histogram shows a somewhat bell-shaped curve but with some deviations.
# 2. The distribution appears to be slightly skewed to the right (positive skewness), as the tail on the right side is longer than the left.
# 3. The data points are spread between approximately 25 and 175, with the highest density around the center.

# In[11]:


# Univariate analysis on Area mean variable
plt.figure(figsize=(6,3))
sns.distplot(df.Area_Mean)
plt.title("Distribution_of_Area_Mean_Variable")
plt.savefig("C:\\Users\\yozil\\Desktop\\My projects\\3.0 Breast_Cancer_Detection\\Breast_Cancer_Detection\\models\\5.0 Distribution_of_Area_Mean_Variable.jpg")
plt.show()


# As we can see from the above Distribution Plot of Area_Mean variable
# 1. The histogram shows a right-skewed distribution rather than a bell-shaped curve.
# 2. The distribution is not symmetric; it has a longer tail on the right side.
# 3. The data points are spread between approximately 0 and 2000, with the highest density around lower values.

# In[12]:


# Univariate analysis on Smoothness mean variable
plt.figure(figsize=(6,3))
sns.distplot(df.Smoothness_Mean)
plt.title("Distribution_of_Smoothness_Mean_Variable")
plt.savefig("C:\\Users\\yozil\\Desktop\\My projects\\3.0 Breast_Cancer_Detection\\Breast_Cancer_Detection\\models\\6.0 Distribution_of_Smoothness_Mean_Variable.jpg")
plt.show()


# As we can see from the above smoothness mean distribution plot
# 1. The distribution appears to be approximately normal, centered around a mean smoothness value of approximately 0.10.
# 2. The smoothness values range roughly from 0.04 to 0.14, indicating the spread or variability in the data.
# 

# In[13]:


# Univariate analysis on Compactness mean variable
plt.figure(figsize=(6,3))
sns.distplot(df.Compactness_Mean)
plt.title("Distribution_of_Compactness_Mean_Variable")
plt.savefig("C:\\Users\\yozil\\Desktop\\My projects\\3.0 Breast_Cancer_Detection\\Breast_Cancer_Detection\\models\\7.0 Distribution_of_Compactness_Mean_Variable.jpg")
plt.show()


# As we can see from the distribution Compactness distribution plot above:
# 1. The distribution has a peak around the 0.05 to 0.07 range, suggesting that the most frequent values for compactness mean are in this range.
# 2. The compactness values range from close to 0 to about 0.35, indicating a wide variability in the data.
# 3. The distribution is right-skewed (positively skewed), with a longer tail extending to the right. This indicates that while most values are clustered around the lower end, there are a few higher values stretching the distribution to the right.

# In[14]:


# Univariate analysis on Concavity mean variable
plt.figure(figsize=(6,3))
sns.distplot(df.Concavity_Mean)
plt.title("Distribution_of_Concavity_Mean_Variable")
plt.savefig("C:\\Users\\yozil\\Desktop\\My projects\\3.0 Breast_Cancer_Detection\\Breast_Cancer_Detection\\models\\8.0 Distribution_of_Concavity_Mean_Variable.jpg")
plt.show()


# As we can see from the above concavity mean distribution plot:
# 1. The distribution has a peak around 0.02 to 0.04, suggesting that the most frequent values for concavity mean are in this range.
# 2. The concavity values range from close to 0 to about 0.4, indicating a wide variability in the data.
# 3. The distribution is right-skewed (positively skewed), with a longer tail extending to the right. This indicates that while most values are clustered around the lower end, there are a few higher values stretching the distribution to the right.

# In[16]:


# Univariate analysis on Concave Points_Mean variable
plt.figure(figsize=(6,3))
sns.distplot(df["Concave Points_Mean"])
plt.title("Distribution_of_Concave Points_Mean_Variable")
plt.savefig("C:\\Users\\yozil\\Desktop\\My projects\\3.0 Breast_Cancer_Detection\\Breast_Cancer_Detection\\models\\9.0 Distribution_of_Concave Points_Mean_Variable.jpg")
plt.show()


# In[8]:


df.columns


# In[ ]:




