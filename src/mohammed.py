#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# preprocessing 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from feature_engine.encoding import RareLabelEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from category_encoders.ordinal import OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression

# modeling
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


#%%
# load data
flights = pd.read_csv('../data/cleaned_data_mohammed.csv')
flights['Datetime_Dep'] = pd.to_datetime(flights['Datetime_Dep'])
flights['Datetime_Arrival'] = pd.to_datetime(flights['Datetime_Arrival'])
flights['Total_Stops'] = flights['Total_Stops'].astype(str)
flights.drop('Route', axis=1, inplace=True)

# create train-test split
train, test = train_test_split(flights, test_size=0.3)

X_train = train.drop(columns='Price')
y_train = train['Price']




#%%
# Target Engineering
tt = TransformedTargetRegressor(transformer=PowerTransformer('box-cox'))

# Numeric feature engineering
nzv = VarianceThreshold(threshold=0.1)
yj = PowerTransformer(method='yeo-johnson')
scaler = StandardScaler()


### lump the infrequent categories
rare_encoder = RareLabelEncoder(tol=0.01, replace_with="other")
# rare_encoder.fit_transform(X_train)['airline'].unique()

# Categorical feature engineering
dummy_encoder = OneHotEncoder(drop='first')

# ordinal feature engineering
col = X_train['Total_Stops']
dict = {'0':0, '1':1, '2':2, '3':3, '4':4}
category_mapping = [{'col': 'Total_Stops', 'mapping': dict}]

ordinal_encoder = OrdinalEncoder(cols=['Total_Stops'], mapping=category_mapping)



# %%
# preprocessing

# preprocessing target label
# tt_fit = tt.fit(X_train, y_train)
# tt_fit.

# combine all steps into a preprocessing pipeline
preprocessor = ColumnTransformer(
  remainder="passthrough",
  transformers=[
  # ("trans_target", tt, ),
  ("nzv_encode", nzv, selector(dtype_include="number")),
  ("yeo-johnson",yj, selector(dtype_include="number")),
  ("std_encode", scaler, selector(dtype_include="number")),
  ('ordinal_encode', ordinal_encoder, ['Total_Stops']),
  ("rare_encode", rare_encoder, selector(dtype_include="object")),
  ("dummy_encode", dummy_encoder, selector(dtype_include="object")),
  ])

lm = LinearRegression()

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', lm)
])

# define loss function
loss = 'neg_root_mean_squared_error'

cv = RepeatedKFold(n_splits=5, n_repeats=5)


results = cross_val_score(model_pipeline, X_train, y_train, cv = cv, scoring=loss)




# %%
X_train_trans = preprocessor.fit_transform(X_train_reduced)




# %%

hyper_grid_pca = {'n_components': range(1, 20)}














