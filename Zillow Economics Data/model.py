import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

def model_train(data,split):

    features_considered = data.columns
    features_considered = [x for x in features_considered if x not in ('year', 'Date')]
    features = data[features_considered]
    features.index = data['Date']
    # features_cat = features[features.RegionName != 'UnitedStates']

    encoder = LabelEncoder()
    region_le = features['RegionName']
    region_labels = encoder.fit_transform(region_le)
    features['RegionLabels'] = region_labels
    region_ohe = OneHotEncoder()
    region_features_arr = region_ohe.fit_transform(features[['RegionLabels']]).toarray()
    region_features_labels = list(encoder.classes_)
    region_features = pd.DataFrame(region_features_arr, columns=region_features_labels)
    region_features = region_features.drop(['Wyoming'], axis='columns')
    region_features.index = features.index
    features1 = pd.concat([features, region_features], axis='columns')
    features = features1.drop(['RegionName'], axis='columns')
    features = features.drop(['RegionLabels'], axis='columns')
    features.dropna(subset=['Sale_Prices'], inplace=True)

    df_train = features[:split]
    df_test = features[split:]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 30,
        "min_child_samples": 100,
        "learning_rate": 0.1,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.5,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    lgb_train = lgb.Dataset(df_train.loc[:,df_train.columns != "Sale_Prices"], df_train.loc[:,"Sale_Prices"], categorical_feature='auto')
    lgb_eval = lgb.Dataset(df_test.loc[:,df_test.columns != "Sale_Prices"], df_test.loc[:,"Sale_Prices"], reference=lgb_train, categorical_feature='auto')
    gbm = lgb.train(params, lgb_train, num_boost_round=10000, valid_sets=[lgb_eval], early_stopping_rounds=1000,verbose_eval=100)

    predicted_revenue = gbm.predict(df_test.loc[:, df_test.columns != "Sale_Prices"],
                                    num_iteration=gbm.best_iteration)
    predicted_revenue[predicted_revenue < 0] = 0
    df_test["predicted"] = predicted_revenue

    return gbm,df_train,df_test
