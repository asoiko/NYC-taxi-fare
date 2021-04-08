import pickle

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# read data
train = pd.read_csv('data/train.csv', nrows=1_000_000)
test = pd.read_csv('data/test.csv')


# add new feature absolute diff coordinate
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train)
add_travel_vector_features(test)

# drop NaN
train.dropna(how='any', axis='rows', inplace=True)

# filters data 
train = train[(train['abs_diff_latitude'] < 5.0) & (train['abs_diff_longitude'] < 5.0)]
train = train[train['fare_amount'] > 0]

# drop outliers
train.drop(train[train['passenger_count'] > 6].index, axis=0, inplace=True)
train.drop(train[train['passenger_count'] == 0].index, axis=0, inplace=True)

# to datetime
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

# add date features 
data = [train, test]
for x in data:
    x["year"] = x["pickup_datetime"].dt.year
    x["month"] = x["pickup_datetime"].dt.month
    x["day_of_month"] = x["pickup_datetime"].dt.day
    x["day_of_week"] = x["pickup_datetime"].dt.dayofweek
    x["hour"] = x["pickup_datetime"].dt.hour


# get distance with coordinate
def get_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for i in data:
        R = 6371

        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])

        delta_phi = np.radians(i[lat2] - i[lat1])
        delta_lambda = np.radians(i[long2] - i[long2])

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = (R * c)
        i['distance_km'] = d

    return d


get_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
train.drop(train[(train['pickup_latitude'] == 0) & (train['pickup_longitude'] == 0)].index, axis=0, inplace=True)

# fitting

features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
            'abs_diff_longitude', 'abs_diff_latitude', 'year', 'month', 'day_of_month', 'day_of_week', 'hour', \
            'distance_km']

X = train[features]
y = train['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# lgbm 
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'nthread': 4,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'max_depth': -1,
    'subsample': 0.8,
    'bagging_fraction': 1,
    'max_bin': 5000,
    'bagging_freq': 20,
    'colsample_bytree': 0.6,
    'metric': 'rmse',
    'min_split_gain': 0.5,
    'min_child_weight': 1,
    'min_child_samples': 10,
    'scale_pos_weight': 1,
    'seed': 0,
    'force_col_wise': True
}

train_set = lgbm.Dataset(X_train, y_train, silent=False)
test_set = lgbm.Dataset(X_test, y_test, silent=False)
model = lgbm.train(params, train_set, num_boost_round=10000, early_stopping_rounds=500, verbose_eval=500,
                   valid_sets=test_set)

pred_lgbm_submit = model.predict(test[features], num_iteration=model.best_iteration)

# create submission file
submit = pd.DataFrame({
    "key": test.key,
    "fare_amount": pred_lgbm_submit.round(2)}, columns=['key', 'fare_amount'])
submit.to_csv('data/sample_submission.csv', index=False)

# save model
pickle_out = open('models/lgbm_reg.pickle', 'wb')
pickle.dump(model, pickle_out)
