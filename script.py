# Importing Modules
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import uniform, randint
import xgboost as xgb


# Loading dataset

# Initializing train dataset
train_file = "train.csv"
train_df = pd.read_csv(train_file)
print("Full train dataset shape is {}".format(train_df.shape))

# Dropping ID column from train dataset
train_df = train_df.drop('Id', axis=1)

# Initializing test dataset
test_file_path = "test.csv"
test_df = pd.read_csv(test_file_path)

# Stores test IDs
test_ids = test_df['Id']

# Dropping ID column from test dataset
test_df = test_df.drop('Id', axis=1)


# Data Cleaning Function
def clean_data(train_df, test_df):

    for df in [train_df, test_df]:
        
        # Filling LotsFrontage missing data with mean value
        if 'LotFrontage' in train_df.columns:
            df['LotFrontage'] = df['LotFrontage'].fillna(train_df['LotFrontage'].mean())
    
        # Filling BsmtFinType2 missing data with mode value (due to skewness)
        if 'BsmtFinType2' in df.columns:
            df['BsmtFinType2'] = df['BsmtFinType2'].fillna(train_df['BsmtFinType2'].mode()[0])
    
        # Filling BsmtFinType1 missing data with mode value (due to skewness)
        if 'BsmtFinType1' in df.columns:
            df['BsmtFinType1'] = df['BsmtFinType1'].fillna(train_df['BsmtFinType1'].mode()[0])
    
        # Removing Alley
        if 'Alley' in df.columns: 
            df.drop(['Alley'], axis = 1, inplace = True)
    
        # Filling MasVnrType missing data with modal value
        if 'MasVnrType' in df.columns:
            df['MasVnrType'] = df['MasVnrType'].fillna(train_df['MasVnrType'].mode()[0])
    
        # Filling MasVnrArea missing data with modal value
        if 'MasVnrArea' in df.columns:
            df['MasVnrArea'] = df['MasVnrArea'].fillna(train_df['MasVnrArea'].mode()[0])
    
        # Filling GarageQual missing data with modal value
        if 'GarageQual' in df.columns:
            df['GarageQual'] = df['GarageQual'].fillna(train_df['GarageQual'].mode()[0])
    
        # Filling missing data in BsmtCond with modal values
        if 'BsmtCond' in df.columns:
            df['BsmtCond'] = df['BsmtCond'].fillna(train_df['BsmtCond'].mode()[0])
    
        # Filling missing data in Electrical with modal values
        if 'Electrical' in df.columns:
            df['Electrical'] = df['Electrical'].fillna(train_df['Electrical'].mode()[0])
    
        # Filling missing data in BsmtQual with modal values
        if 'BsmtQual' in df.columns:
            df['BsmtQual'] = df['BsmtQual'].fillna(train_df['BsmtQual'].mode()[0])
    
        # Removing GarageYrBlt
        if 'GarageYrBlt' in df.columns:
            df.drop(['GarageYrBlt'], axis = 1, inplace = True)
    
        # Filling missing data in FireplaceQu with modal values
        if 'FireplaceQu' in df.columns:
            df['FireplaceQu'] = df['FireplaceQu'].fillna(train_df['FireplaceQu'].mode()[0])
    
        # Filling missing data in GarageType with modal values
        if 'GarageType' in df.columns:
            df['GarageType'] = df['GarageType'].fillna(train_df['GarageType'].mode()[0])
    
        # Filling missing data in GarageFinish with modal values
        if 'GarageFinish' in df.columns:
            df['GarageFinish'] = df['GarageFinish'].fillna(train_df['GarageFinish'].mode()[0])
    
        # Filling missing data in GarageCond with modal values
        if 'GarageCond' in df.columns:
            df['GarageCond'] = df['GarageCond'].fillna(train_df['GarageCond'].mode()[0])
    
        # Filling missing data in GarageCond with modal values
        if 'BsmtExposure' in df.columns:
            df['BsmtExposure'] = df['BsmtExposure'].fillna(train_df['BsmtExposure'].mode()[0])
    
        df.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
    
        # For Test Dataset:
        
        # Filling missing data in BsmtHalfBath with modal values
        if 'BsmtHalfBath' in df.columns:
            df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(train_df['BsmtHalfBath'].mode()[0])
    
        # Filling missing data in BsmtFullBath with modal values
        if 'BsmtFullBath' in df.columns:
            df['BsmtFullBath'] = df['BsmtFullBath'].fillna(train_df['BsmtFullBath'].mode()[0])
    
        # Filling missing data in BsmtFinSF1 with modal values
        if 'BsmtFinSF1' in df.columns:
            df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(train_df['BsmtFinSF1'].mode()[0])
    
        # Filling missing data in BsmtFinSF2 with modal values
        if 'BsmtFinSF2' in df.columns:
            df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(train_df['BsmtFinSF2'].mode()[0])
    
        # Filling missing data in TotalBsmtSF with modal values
        if 'TotalBsmtSF' in df.columns:
            df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(train_df['TotalBsmtSF'].mode()[0])
    
        # Filling missing data in BsmtUnfSF with modal values
        if 'BsmtUnfSF' in df.columns:
            df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(train_df['BsmtUnfSF'].mode()[0])
    
        # Filling missing data in GarageCars with modal values
        if 'GarageCars' in df.columns:
            df['GarageCars'] = df['GarageCars'].fillna(train_df['GarageCars'].mode()[0])
    
        # Filling missing data in GarageArea with modal values
        if 'GarageArea' in df.columns:
            df['GarageArea'] = df['GarageArea'].fillna(train_df['GarageArea'].mode()[0])
    
    return train_df, test_df

#  Cleaning training and testing dataset
train_processed, test_processed = clean_data(train_df, test_df)


# Data Preprocessing Function
# Handling Categorical Feature
categorical_columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
                      'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                      'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                      'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                      'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                      'PavedDrive', 'SaleType', 'SaleCondition']


label_encoders = {}
le = LabelEncoder()
scaler = StandardScaler()

# LabelEncoder converts categorical columns to numerical representation
for field in categorical_columns:
    if field != 'SalePrice':
        
        # Fit on combined data to ensure consistent encoding
        combined_values = pd.concat([train_processed[field], test_processed[field]], axis=0)
        le.fit(combined_values)
        
        train_processed[field] = le.transform(train_processed[field])
        test_processed[field] = le.transform(test_processed[field])
        
        label_encoders[field] = le

# Feature Scaling: Transforms dataset so they have a mean of 0 and a standard deviation of 1

scaler.fit(train_processed[categorical_columns])
train_processed[categorical_columns] = scaler.transform(train_processed[categorical_columns])
test_processed[categorical_columns] = scaler.transform(test_processed[categorical_columns])
        
        

print("Data preprocessing completed...")



## Using Scikit-learn
# Prepare descriptive features and target features
desc_feat = train_processed.drop('SalePrice', axis=1)
targ_feat = train_processed['SalePrice']

# Function for train-test-split (80:20)
d_train, d_val, t_train, t_val = train_test_split(desc_feat, targ_feat, test_size=0.2, random_state=23)

def rmse_score(t_true, t_pred):
    """Calculates Root Mean Squared Error (RMSE)"""
    return math.sqrt(mean_squared_error(t_true, t_pred))

# Training Bottom Line Models

# Random Forest Bottomline Model
rf_bl = RandomForestRegressor()
rf_bl.fit(d_train, t_train)
rf_bl_preds = rf_bl.predict(d_val)
rf_bl_rmse = rmse_score(t_val, rf_bl_preds)

# K-NN Bottomline Model
knn_bl = KNeighborsRegressor()
knn_bl.fit(d_train, t_train)
knn_val_preds = knn_bl.predict(d_val)
knn_bl_rmse = rmse_score(t_val, knn_val_preds)

# SVM Bottomline Model
svm_bl = SVR()
svm_bl.fit(d_train, t_train)
svm_bl_preds = svm_bl.predict(d_val)
svm_bl_rmse = rmse_score(t_val, svm_bl_preds)

# XGBoost Bottomline Model
xgb_bl = xgb.XGBRegressor(objective='reg:squarederror')
xgb_bl.fit(d_train, t_train)
xgb_bl_preds = xgb_bl.predict(d_val)
xgb_bl_rmse = np.sqrt(mean_squared_error(t_val, xgb_bl_preds))

bottomline_models = {
    'Random Forest': rf_bl_rmse,
    'k-NN': knn_bl_rmse,
    'SVM': svm_bl_rmse,
    'XGBoost': xgb_bl_rmse
}


# Training Fine Tuned Models

# Random Forest Fine-tuned Model
# Random Forest Hyperparameter tuning
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# Random Forest Hyperparameter tuning (For RandomizedSearchCV)
rf_params_rs = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'bootstrap': [True, False],
    'max_features': ['sqrt', 'log2', None]
}

# Setting up RandomizedSearch Cross Validation
rf_grid = RandomizedSearchCV(
    RandomForestRegressor(random_state=3), 
    rf_params_rs,
    n_iter= 50,
    cv=3,
    verbose= 1,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Training the fine tuned model
rf_grid.fit(d_train, t_train)
# Finding the best params
rf_best = rf_grid.best_estimator_
# Setting up and training fine tuned model
rf_tuned_preds = rf_best.predict(d_val)
rf_tuned_rmse = rmse_score(t_val, rf_tuned_preds)


# k-NN Fine-tuned Model
# k-NN Hyperparameter tuning
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# k-NN Hyperparameter tuning (For RandomizedSearchCV)
knn_params_rs = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2, 3],  # For Minkowski distance
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Setting up RandomizedSearch Cross Validation
knn_grid = RandomizedSearchCV(
    KNeighborsRegressor(), 
    knn_params_rs, 
    n_iter= 30,
    cv=3,
    verbose= 1,
    random_state= 3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Training the fine tuned model
knn_grid.fit(d_train, t_train)
# Finding the best params
knn_best = knn_grid.best_estimator_
# Setting up and training fine tuned model
knn_tuned_preds = knn_best.predict(d_val)
knn_tuned_rmse = rmse_score(t_val, knn_tuned_preds)


# SVM Fine-tuned Model
# SVM Hyperparameter tuning
svm_params = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

# SVM Hyperparameter tuning (For RandomizedSearchCV)
svm_params_rs = {
    'kernel': ['rbf'],
    'C': uniform(0.1, 5),
    'epsilon': uniform(0.01, 0.2),
    'gamma': ['scale', 'auto'],
}

# Setting up RandomizedSearch Cross Validation
svm_grid = RandomizedSearchCV(
    SVR(), 
    svm_params_rs,
    n_iter= 10,
    random_state= 3,
    cv=2,
    verbose= 1,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Training the fine tuned model
svm_grid.fit(d_train, t_train)
# Finding the best params
svm_best = svm_grid.best_estimator_
# Setting up and training fine tuned model
svm_tuned_preds = svm_best.predict(d_val)
svm_tuned_rmse = rmse_score(t_val, svm_tuned_preds)


# XGBoost Fine-tuned Model

# XGBoost learning parameter tuning
lr = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
}

xgb_lr = xgb.XGBRegressor(
    n_estimators=100,
    random_state=3,
    objective='reg:squarederror'
)

# Learning Rate tuning (For RandomizedSearchCV)
xgb_params_rs = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'min_child_weight': randint(1, 10),
    'subsample': uniform(0.6, 0.4),  # Between 0.6 and 1.0
    'colsample_bytree': uniform(0.6, 0.4),
    'learning_rate': uniform(0.01, 0.29),  # Between 0.01 and 0.3
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

# SVM Hyperparameter tuning (For RandomizedSearchCV)
xgb_gridcv = RandomizedSearchCV(
    xgb.XGBRegressor(
        random_state= 3,
        objective='reg:squarederror',
        tree_method='hist'
    ),
    xgb_params_rs,
    n_iter= 50,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

# Training the fine tuned model
xgb_gridcv.fit(d_train, t_train)
# Finding the best params
xgb_best = xgb_gridcv.best_estimator_
# Setting up and training fine tuned model
xgb_tuned_preds = xgb_best.predict(d_val)
xgb_tuned_rmse = rmse_score(t_val, xgb_tuned_preds)

tuned_models = {
    'Random Forest': rf_tuned_rmse,
    'k-NN': knn_tuned_rmse,
    'SVM': svm_tuned_rmse,
    'XGBoost': xgb_tuned_rmse
    
}


## Using Scikit-learn
train_set = pd.concat([d_train, t_train], axis= 1)
test_set = pd.concat([d_val, t_val], axis= 1)

# Preparing TensorFlow datasets
ds_train = tfdf.keras.pd_dataframe_to_tf_dataset(train_set, label="SalePrice", task= tfdf.keras.Task.REGRESSION)
ds_val   = tfdf.keras.pd_dataframe_to_tf_dataset(test_set,   label="SalePrice", task= tfdf.keras.Task.REGRESSION)


# Training Bottom Line Models

# Random Forest Bottomline Model
rf_bltf  = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
# Training the bottomline model
rf_bltf.fit(ds_train)
# Producing prediction
rf_bltf_pred = np.squeeze(rf_bltf.predict(ds_val))
# Saves true values of target feature
y_true = test_set['SalePrice'].to_numpy()
# Calculating RMSE
rf_bltf_rmse = np.sqrt(mean_squared_error(y_true, rf_bltf_pred))

# XGBoost Bottomline Model
gb_bltf  = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION)
# Training the bottomline model
gb_bltf.fit(ds_train)
# Producing prediction
gb_bltf_pred = np.squeeze(gb_bltf.predict(ds_val))
# Calculating RMSE
gb_bltf_rmse = np.sqrt(mean_squared_error(y_true, gb_bltf_pred))

bottomline_models_tf = {
    'Random Forest (TF)': rf_bltf_rmse,
    'Gradient Boosting (TF)': gb_bltf_rmse
}


# Training Fine Tuned Models

# Random Forest Fine-tuned Model
rf_tunedtf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION, num_trees=500, max_depth=16, min_examples=2)
# Training the fine tuned model
rf_tunedtf.fit(ds_train)
# Producing prediction
rf_tunedtf_pred = np.squeeze(rf_tunedtf.predict(ds_val))
# Calculating RMSE
rf_tunedtf_rmse = np.sqrt(mean_squared_error(y_true, rf_tunedtf_pred))

# XGBoost Fine-tuned Model
gb_tunedtf = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION, num_trees=1_000, max_depth=8, shrinkage=0.05, subsample=0.7)
# Training the fine tuned model
gb_tunedtf.fit(ds_train)
# Producing prediction
gb_tunedtf_pred = np.squeeze(gb_tunedtf.predict(ds_val))
# Calculating RMSE
gb_tunedtf_rmse = np.sqrt(mean_squared_error(y_true, gb_tunedtf_pred))

tuned_models_tf = {
    'Random Forest (TF)': rf_tunedtf_rmse,
    'Gradient Boosting (TF)': gb_tunedtf_rmse
}


# Evaluating the best model
all_results = {**bottomline_models, **tuned_models, **bottomline_models_tf, **tuned_models_tf}
best_model_name = min(all_results, key=all_results.get)
best_rmse = all_results[best_model_name]

print(f"\nBest Model: {best_model_name} with RMSE: {best_rmse:.2f}")

# Generates test predictions with best model
if best_model_name == 'Random Forest':
    if rf_tuned_rmse < rf_bl_rmse:
        best_model = rf_best
        print("Using fine-tuned Random Forest")
    else:
        best_model = rf_bl
        print("Using baseline Random Forest")
elif best_model_name == 'k-NN':
    if knn_tuned_rmse < knn_bl_rmse:
        best_model = knn_best
        print("Using fine-tuned k-NN")
    else:
        best_model = knn_bl
        print("Using baseline k-NN")
elif best_model_name == 'XGBoost':
    if xgb_tuned_rmse < xgb_bl_rmse:
        best_model = xgb_best
        print("Using fine-tuned XGBoost")
    else:
        best_model = xgb_bl  # Fixed: was knn_bl
        print("Using baseline XGBoost")
elif best_model_name == 'Random Forest (TF)':
    if rf_tunedtf_rmse < rf_bltf_rmse:
        best_model = rf_tunedtf
        print("Using fine-tuned TF Random Forest")
    else:
        best_model = rf_bltf
        print("Using baseline TF Random Forest")
elif best_model_name == 'Gradient Boosting (TF)':
    if gb_tunedtf_rmse < gb_bltf_rmse:
        best_model = gb_tunedtf
        print("Using fine-tuned TF Gradient Boosting")
    else:
        best_model = gb_bltf
        print("Using baseline TF Gradient Boosting")
else:  # SVM
    if svm_tuned_rmse < svm_bl_rmse:
        best_model = svm_best
        print("Using fine-tuned SVM")
    else:
        best_model = svm_bl
        print("Using baseline SVM")

# Generates final predictions
if 'TF' in best_model_name:
    # For TensorFlow Decision Forests models, we need a tf.data.Dataset
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_processed, task=tfdf.keras.Task.REGRESSION)
    test_predictions = np.squeeze(best_model.predict(test_ds))
else:
    # For scikit-learn models
    test_predictions = best_model.predict(test_processed)


# Generating Prediction From The Best Model
# Creates submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})

# Saves submission
submission.to_csv('house_prices_submission.csv', index=False)
print(f"\nPredictions saved to 'house_prices_submission.csv'")
print(f"Best model used: {best_model}")


# Visualizes All Model Scores
comparison_data = []

for model in ['Random Forest', 'k-NN', 'SVM', 'XGBoost']:
    comparison_data.append({
        'Model': model,
        'Type': 'Bottomline',
        'RMSE': bottomline_models[model]
    })
    comparison_data.append({
        'Model': model,
        'Type': 'Fine-tuned',
        'RMSE': tuned_models[model]
    })

# Creating a comparison dataframe
comparison_df = pd.DataFrame(comparison_data)

for model in ['Random Forest (TF)', 'Gradient Boosting (TF)']:
    comparison_data.append({
        'Model': model,
        'Type': 'Bottomline',
        'RMSE': bottomline_models_tf[model]
    })
    comparison_data.append({
        'Model': model,
        'Type': 'Fine-tuned',
        'RMSE': tuned_models_tf[model]
    })

# Creating a comparison dataframe
comparison_df = pd.DataFrame(comparison_data)
print(comparison_df)