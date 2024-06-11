# BUSINESS SCIENCE LEARNING LABS ----
# LAB 59: CUSTOMER LIFETIME VALUE ----
# CUSTOMER LIFETIME VALUE WITH MACHINE LEARNING ----
# **** ----

# CONDA ENV USED: lab_59_customer_ltv_py

# LIBRARIES ----
import pandas as pd
import numpy as np
import joblib 

import plydata.cat_tools as cat
import plotnine as pn

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

pn.options.dpi = 300


# 1.0 DATA PREPARATION ----

cdnow_raw_df = pd.read_csv(
    "data/CDNOW_master.txt", 
    sep   = "\s+",
    names = ["customer_id", "date", "quantity", "price"]
)

cdnow_raw_df.info()

cdnow_df = cdnow_raw_df \
    .assign(
        date = lambda x: x['date'].astype(str)
    ) \
    .assign(
        date = lambda x: pd.to_datetime(x['date'])
    ) \
    .dropna()

cdnow_df.info()

# 2.0 COHORT ANALYSIS ----
# - Only the customers that have joined at the specific business day

# Get Range of Initial Purchases ----
cdnow_first_purchase_tbl = cdnow_df \
    .sort_values(['customer_id', 'date']) \
    .groupby('customer_id') \
    .first()

cdnow_first_purchase_tbl

cdnow_first_purchase_tbl['date'].min()

cdnow_first_purchase_tbl['date'].max()

# Visualize: All purchases within cohort

cdnow_df \
    .reset_index() \
    .set_index('date') \
    [['price']] \
    .resample(
        rule = "MS"
    ) \
    .sum() \
    .plot()

# Visualize: Individual Customer Purchases

ids = cdnow_df['customer_id'].unique()
ids_selected = ids[0:10]

cdnow_cust_id_subset_df = cdnow_df \
    [cdnow_df['customer_id'].isin(ids_selected)] \
    .groupby(['customer_id', 'date']) \
    .sum() \
    .reset_index()
pn.ggplot(
    data=cdnow_cust_id_subset_df,
    mapping=pn.aes(x='date', y='price', group='customer_id')
) + \
pn.geom_line() + \
pn.geom_point() + \
pn.facet_wrap('customer_id') + \
pn.scale_x_date(
    date_breaks="1 year",
    date_labels="%Y"
)



# 3.0 MACHINE LEARNING ----
#  Frame the problem:
#  - What will the customers spend in the next 90-Days? (Regression)
#  - What is the probability of a customer to make a purchase in next 90-days? (Classification)


# 3.1 TIME SPLITTING (STAGE 1) ----

n_days   = 90
max_date = cdnow_df['date'].max() 
cutoff   = max_date - pd.to_timedelta(n_days, unit = "d")

temporal_in_df = cdnow_df \
    [cdnow_df['date'] <= cutoff]

temporal_out_df = cdnow_df \
    [cdnow_df['date'] > cutoff]


# 3.2 FEATURE ENGINEERING (RFM) ----
#   - Most challenging part
#   - 2-Stage Process
#   - Need to frame the problem
#   - Need to think about what features to include

# Make Targets from out data ----
# Make Recency (Date) Features from in data ----
# Make Targets from out data ----
targets_df = temporal_out_df \
    .drop(['quantity', 'date'], axis=1) \
    .groupby('customer_id') \
    .sum() \
    .rename(columns={'price': 'spend_90_total'}) \
    .assign(spend_90_flag=1)

max_date = temporal_in_df['date'].max()
if temporal_in_df.empty:
    recency_features_df = pd.DataFrame(columns=["recency"])  # Create an empty DataFrame
else:
    recency_features_df = temporal_in_df \
        [['customer_id', 'date']] \
        .groupby('customer_id') \
        .apply(
            lambda x: (x['date'].max() - max_date) / pd.to_timedelta(1, "day")
        ) \
        .to_frame() \
        .set_axis(["recency"], axis=1)


# Make Frequency (Count) Features from in data ----

frequency_features_df = temporal_in_df \
    [['customer_id', 'date']] \
    .groupby('customer_id') \
    .count() \
    .set_axis(['frequency'], axis=1)

# Make Price (Monetary) Features from in data ----

price_features_df = temporal_in_df \
    .groupby('customer_id') \
    .aggregate(
        {
            'price': ["sum", "mean"]
        }
    ) \
    .set_axis(['price_sum', 'price_mean'], axis = 1)

# 3.3 COMBINE FEATURES ----

features_df = pd.concat(
    [recency_features_df, frequency_features_df, price_features_df], axis = 1
) \
    .merge(
        targets_df, 
        left_index  = True, 
        right_index = True, 
        how         = "left"
    ) \
    .fillna(0)

# 4.0 MACHINE LEARNING -----

X = features_df[['recency', 'frequency', 'price_sum', 'price_mean']]

# 4.1 NEXT 90-DAY SPEND PREDICTION ----

y_spend = features_df['spend_90_total']

xgb_reg_spec = XGBRegressor(
    objective="reg:squarederror",   
    random_state=123
)

xgb_reg_model = GridSearchCV(
    estimator=xgb_reg_spec, 
    param_grid=dict(
        learning_rate = [0.01, 0.1, 0.3, 0.5]
    ),
    scoring = 'neg_mean_absolute_error',
    refit   = True,
    cv      = 5
)

xgb_reg_model.fit(X, y_spend)

predictions_reg = xgb_reg_model.predict(X)


# 4.2 NEXT 90-DAY SPEND PROBABILITY ----

y_prob = features_df['spend_90_flag']

xgb_clf_spec = XGBClassifier(
    objective    = "binary:logistic",   
    random_state = 123
)

xgb_clf_model = GridSearchCV(
    estimator=xgb_clf_spec, 
    param_grid=dict(
        learning_rate = [0.01, 0.1, 0.3, 0.5]
    ),
    scoring = 'roc_auc',
    refit   = True,
    cv      = 5
)

xgb_clf_model.fit(X, y_prob)

predictions_clf = xgb_clf_model.predict_proba(X)

# 4.3 FEATURE IMPORTANCE (GLOBAL) ----

# Importance | Spend Amount Model
imp_spend_amount_dict = xgb_reg_model \
    .best_estimator_ \
    .get_booster() \
    .get_score
