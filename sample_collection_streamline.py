# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:42:10 2022

@author: Shreni Singh
"""

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

import numpy as np
import seaborn as sns
#loading dataset
df = pd.read_excel(r"C:\Users\Shreni Singh\Desktop\medical_sample_streamline\sample_data.xlsx")
df.head()
df.info()
df.shape
df.dtypes
data=df.drop(['Patient_ID','Patient_Gender','Agent_ID','Test_Booking_Date','Test_Booking_Time_HH_MM','Mode_Of_Transport','Scheduled_Sample_Collection_Time_HH_MM','Sample_Collection_Date'],axis=1)
data.head()
data.info()
duplicate = data.duplicated()
duplicate
sum(duplicate)
data = data.drop_duplicates() 
duplicate = data.duplicated()
sum(duplicate)

# let's find outliers 
sns.boxplot(data.Patient_Age)         # outliers


# Detection of outliers (find limits for salary based on IQR)
IQR = data['Patient_Age'].quantile(0.75) - data['Patient_Age'].quantile(0.25)
lower_limit = data['Patient_Age'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Patient_Age'].quantile(0.75) + (IQR * 1.5)
lower_limit
upper_limit
############### 3. Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])
df_t = winsor.fit_transform(data[['Patient_Age']])
sns.boxplot(df_t.Patient_Age);plt.title('Boxplot');plt.show()
data['Patient_Age']=df_t
sns.boxplot(data.Patient_Age)  # no outliers

sns.boxplot(data.Cut_off_time_HH_MM)       # outliers

# Detection of outliers (find limits for salary based on IQR)
IQR = data['Cut_off_time_HH_MM'].quantile(0.75) - data['Cut_off_time_HH_MM'].quantile(0.25)
lower_limit = data['Cut_off_time_HH_MM'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Cut_off_time_HH_MM'].quantile(0.75) + (IQR * 1.5)
lower_limit
upper_limit
############### 3. Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Cut_off_time_HH_MM'])
df_t = winsor.fit_transform(data[['Cut_off_time_HH_MM']])
sns.boxplot(df_t.Cut_off_time_HH_MM);plt.title('Boxplot');plt.show()
data['Cut_off_time_HH_MM']=df_t
sns.boxplot(data.Cut_off_time_HH_MM)     # no outliers

sns.boxplot(data.Time_Taken_To_Reach_Patient_MM)          # outliers

# Detection of outliers (find limits for salary based on IQR)
IQR = data['Time_Taken_To_Reach_Patient_MM'].quantile(0.75) - data['Time_Taken_To_Reach_Patient_MM'].quantile(0.25)
lower_limit = data['Time_Taken_To_Reach_Patient_MM'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Time_Taken_To_Reach_Patient_MM'].quantile(0.75) + (IQR * 1.5)
lower_limit
upper_limit
############### 3. Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Time_Taken_To_Reach_Patient_MM'])
df_t = winsor.fit_transform(data[['Time_Taken_To_Reach_Patient_MM']])
sns.boxplot(df_t.Time_Taken_To_Reach_Patient_MM);plt.title('Boxplot');plt.show()
data['Time_Taken_To_Reach_Patient_MM']=df_t
sns.boxplot(data.Time_Taken_To_Reach_Patient_MM)       # no outliers


sns.boxplot(data.Time_For_Sample_Collection_MM)               #  outliers

# Detection of outliers (find limits for salary based on IQR)
IQR = data['Time_For_Sample_Collection_MM'].quantile(0.75) - data['Time_For_Sample_Collection_MM'].quantile(0.25)
lower_limit = data['Time_For_Sample_Collection_MM'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Time_For_Sample_Collection_MM'].quantile(0.75) + (IQR * 1.5)
lower_limit
upper_limit
############### 3. Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Time_For_Sample_Collection_MM'])
df_t = winsor.fit_transform(data[['Time_For_Sample_Collection_MM']])
sns.boxplot(df_t.Time_For_Sample_Collection_MM);plt.title('Boxplot');plt.show()
data['Time_For_Sample_Collection_MM']=df_t
sns.boxplot(data.Time_For_Sample_Collection_MM)       # no outliers

sns.boxplot(data.Lab_Location_KM)           # outliers present

# Detection of outliers (find limits for salary based on IQR)
IQR = data['Lab_Location_KM'].quantile(0.75) - data['Lab_Location_KM'].quantile(0.25)
lower_limit = data['Lab_Location_KM'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Lab_Location_KM'].quantile(0.75) + (IQR * 1.5)
lower_limit
upper_limit
############### 3. Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Lab_Location_KM'])
df_t = winsor.fit_transform(data[['Lab_Location_KM']])
sns.boxplot(df_t.Lab_Location_KM);plt.title('Boxplot');plt.show()
data['Lab_Location_KM']=df_t
sns.boxplot(data.Lab_Location_KM)       # no outliers
                         
sns.boxplot(data.Time_Taken_To_Reach_Lab_MM)                   #  outliers

# Detection of outliers (find limits for salary based on IQR)
IQR = data['Time_Taken_To_Reach_Lab_MM'].quantile(0.75) - data['Time_Taken_To_Reach_Lab_MM'].quantile(0.25)
lower_limit = data['Time_Taken_To_Reach_Lab_MM'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Time_Taken_To_Reach_Lab_MM'].quantile(0.75) + (IQR * 1.5)
lower_limit
upper_limit
############### 3. Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Time_Taken_To_Reach_Lab_MM'])
df_t = winsor.fit_transform(data[['Time_Taken_To_Reach_Lab_MM']])
sns.boxplot(df_t.Time_Taken_To_Reach_Lab_MM);plt.title('Boxplot');plt.show()
data['Time_Taken_To_Reach_Lab_MM']=df_t
sns.boxplot(data.Time_Taken_To_Reach_Lab_MM)       # no outliers


# visualization for correlation
sns.heatmap(data.corr(),cbar=True,cmap='Blues');


# label encoding for visualization purposes

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
datalabel = data.copy()


datalabel['Test_Name'] = le.fit_transform(datalabel['Test_Name'])
datalabel['Sample'] = le.fit_transform(datalabel['Sample'])
datalabel['Way_Of_Storage_Of_Sample'] = le.fit_transform(datalabel['Way_Of_Storage_Of_Sample'])
datalabel['Cut_off_Schedule'] = le.fit_transform(datalabel['Cut_off_Schedule'])
datalabel['Traffic_Conditions'] = le.fit_transform(datalabel['Traffic_Conditions'])

datalabel.info()

plt.figure(figsize=(15, 7))
plt.subplot(2,2,1)
sns.distplot(datalabel['Test_Name'])

plt.figure(figsize=(15, 7))
plt.subplot(2,2,1)
sns.distplot(datalabel['Cut_off_Schedule'])

plt.figure(figsize=(15, 7))
plt.subplot(2,2,1)
sns.distplot(datalabel['Traffic_Conditions'])

plt.figure(figsize=(15, 7))
plt.subplot(2,2,1)
sns.distplot(datalabel['Time_Taken_To_Reach_Lab_MM'])

plt.figure(figsize=(15, 7))
plt.subplot(2,2,1)
plt.hist(datalabel['Reached_On_Time'])

data.columns
data.head()

# rearranging the columns
data = data.iloc[: , [0, 1, 2, 3, 4, 6, 5,7, 8 , 9 , 10 , 11 , 12]]
data.head()
# separating the predictors and target variable
X = data.drop('Reached_On_Time', axis=1)
Y = data['Reached_On_Time']

# resampling the variables 
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import RandomOverSampler

os = RandomOverSampler(random_state=42)

X.shape, Y.shape
x_res, y_res = os.fit_resample(X, Y)
x_res.shape, y_res.shape

# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.20, random_state=32)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

ohe = OneHotEncoder()

ohe.fit(X[['Test_Name', 'Sample', 'Way_Of_Storage_Of_Sample', 'Cut_off_Schedule', 'Traffic_Conditions']])


column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['Test_Name', 'Sample', 'Way_Of_Storage_Of_Sample', 'Cut_off_Schedule', 'Traffic_Conditions']), 
                                                    remainder='passthrough')


# building model using Random Forest

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

rf_cl = RandomForestClassifier(random_state=42)

pipe = make_pipeline(column_trans, rf_cl)

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)
# saving the model in a pickle file
import pickle

rf_model = 'rf_cl_model.pkl'
pickle.dump(pipe, open(rf_model, 'wb'))



















