# House Price Prediction

Built predictive bottomline and fine tuned models to predict house prices using the Ames housing dataset from Kaggle's House Prices competition.
Link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## Data Exploration & Analysis

- Dataset Loading: Loaded training (1,460 houses) and test (1,459 houses) datasets

- Exploratory Data Analysis:
  - Examined 79 features describing house characteristics
  - Analyzed price distribution using histograms and statistical summaries
  - Visualized numerical feature distributions
  - Identified missing values patterns
  - Analyzed feature correlations with house prices

## Data Preprocessing
  
  - Missing Value Handling: Created a comprehensive preprocess_data() function that:
    
      - Fills LotFrontage with mean values
      - Handles categorical missing values with mode
      - Removes problematic columns (Alley, GarageYrBlt, PoolQC, Fence, MiscFeature)
        
  - Feature Engineering:
    
      - Label encoded 40+ categorical features
      - Ensured consistent encoding between training and test sets
        
  - Data Splitting: 80/20 train-validation split

## Model Implementation

- Scikit-learn Models
  
  - Implemented both baseline and fine-tuned versions of:
  - Random Forest Regressor
  - K-Nearest Neighbors Regressor
  - Support Vector Machine (SVR)
  - XGBoost Regressor
    
- TensorFlow Decision Forests
  - Random Forest Model (baseline & fine-tuned)
  - Gradient Boosted Trees Model (baseline & fine-tuned)
 
## Hyperparameter Tuning

- Used GridSearchCV for scikit-learn models
- Tuned parameters like:
  - Number of estimators, max depth, min samples split (Random Forest)
  - Number of neighbors, weights, distance metrics (k-NN)
  - Kernel parameters, C, gamma (SVM)
  - Learning rate, tree parameters (XGBoost)
 
## Model Evaluation & Comparison

- Evaluation Metric:
  - Root Mean Squared Error (RMSE)
- Performance Tracking:
  - Compared baseline vs fine-tuned models
- Visualization:
  - Created comparative bar plots showing model performance
- Best Model Selection:
  - Automatically identified the best performing model

## Results

- Gradient Boosting (TF) Fine-tuned: ~20,475 RMSE

## Final Prediction

- Generated predictions on the test dataset using the best model
- Created a submission file (house_prices_submission.csv) in Kaggle format
- Handled both scikit-learn and TensorFlow model prediction formats

![image](D1.png)
![image](D2.png)
![image](D3.png)
![image](D5.png)
![image](D4.png)

