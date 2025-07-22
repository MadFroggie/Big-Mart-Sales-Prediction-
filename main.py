import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')

train_df['source'] = 'train'
test_df['source'] = 'test'
data = pd.concat([train_df, test_df], ignore_index=True)

item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
data.loc[data['Item_Weight'].isnull(), 'Item_Weight'] = data.loc[data['Item_Weight'].isnull(), 'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x, 'Item_Weight'])

outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
data.loc[data['Outlet_Size'].isnull(), 'Outlet_Size'] = data.loc[data['Outlet_Size'].isnull(), 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
data['Outlet_Age'] = 2013 - data['Outlet_Establishment_Year']
data['Item_Category'] = data['Item_Identifier'].apply(lambda x: x[:2])
data['Item_Category'] = data['Item_Category'].replace({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})

item_vis_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
data.loc[data['Item_Visibility'] == 0, 'Item_Visibility'] = data.loc[data['Item_Visibility'] == 0, 'Item_Identifier'].apply(lambda x: item_vis_avg.loc[x, 'Item_Visibility'])

le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])

cat_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Category']
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Category'])
data.drop(['Outlet_Establishment_Year', 'Item_Type'], axis=1, inplace=True)

train = data[data['source'] == 'train']
test = data[data['source'] == 'test']
train.drop('source', axis=1, inplace=True)
test.drop(['source', 'Item_Outlet_Sales'], axis=1, inplace=True)

target = 'Item_Outlet_Sales'
features = [col for col in train.columns if col not in [target, 'Item_Identifier', 'Outlet_Identifier']]

model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=100, n_jobs=-1, random_state=42)
model.fit(train[features], train[target])
test[target] = model.predict(test[features])

submission = test[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']]
submission.to_csv('submission.csv', index=False)
print("submission.csv created successfully.")
