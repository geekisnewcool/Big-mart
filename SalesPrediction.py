import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('C:/Users/bhupe/Desktop/Big Mart')
#impoting the datasets
train_dataset=pd.read_csv('train.csv')
test_dataset=pd.read_csv('test.csv')
#dependent variable
y=train_dataset.iloc[:,[11]].values
#independent variables
x=train_dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
x.Outlet_Size=x.Outlet_Size.fillna(value='Medium')
x1=test_dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
x1.Outlet_Size=x1.Outlet_Size.fillna(value='Medium')
#filling missing values in weight column
x['new_weight']=x['Item_Weight'].groupby([x['Item_Type']]).apply(lambda z: z.fillna(z.mean()))
x1['new_weight']=x1['Item_Weight'].groupby([x1['Item_Type']]).apply(lambda z: z.fillna(z.mean()))
z=x.iloc[:,[1,2,4,6,7,8,9,10]]
z1=x1.iloc[:,[1,2,4,6,7,8,9,10]]

#replacing values in item type_fat_content
z.Item_Fat_Content=z.Item_Fat_Content.replace(['Low Fat', 'low fat'], 'LF')
z.Item_Fat_Content=z.Item_Fat_Content.replace(['Regular'], 'reg')
z1.Item_Fat_Content=z1.Item_Fat_Content.replace(['Low Fat', 'low fat'], 'LF')
z1.Item_Fat_Content=z1.Item_Fat_Content.replace(['Regular'], 'reg')

train_length = len(train_dataset)

dataset=pd.concat(objs=[z,z1],axis=0)
dataset = pd.get_dummies(dataset)
train=dataset.iloc[0:train_length,:].values
test=dataset.iloc[train_length:14204,:].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
train[:,2]=labelencoder.fit_transform(train[:,2])
test[:,2]=labelencoder.fit_transform(test[:,2])

onehotencoder=OneHotEncoder(categorical_features=[2])
train=onehotencoder.fit_transform(train).toarray()
test=onehotencoder.fit_transform(test).toarray()

from sklearn.preprocessing import StandardScaler
fs=StandardScaler()
fs.fit_transform(train)
fs.fit_transform(test)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,criterion='mse')
regressor=regressor.fit(train,y)

y_final=regressor.predict(test)

new_dataset=test_dataset.drop(['Item_Weight', 'Item_Visibility', 'Item_Fat_Content', 'Item_Type', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'], axis=1)
new_dataset['Item_Outlet_Sales']=y_final

new_dataset.to_csv('C:/Users/bhupe/Desktop/Big Mart/Prt.csv',index=False)


