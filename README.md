# House Price Prediction

Built predictive bottomline and fine tuned models to predict house prices using the Ames housing dataset from Kaggle's House Prices competition.
Link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## What I Did

- Cleaned the raw CSVs:

- filled missing numbers with the column mean

- filled missing text with the mode

- Turned every non-numeric column into numbers with label-encoding.

- Split the data 80 / 20 into train and validation sets.

- Trained baseline and fine-tuned Random Forest, K-Nearest Neighbors, Support Vector Machine and Gradient Boosting models.

- Wrote an rmse_score() helper to measure error; fixed a small naming bug that was causing a NameError.

- Logged the baseline score (≈ 20,221 RMSE).

- Added fully-commented code for a tuned XGBoost model (learning-rate, depth, subsample, etc.).

## The Data

This dataset has 79 features describing houses. Fields like square footage, number of bedrooms, neighborhood, condition ratings, garage size, etc. Pretty comprehensive for house characteristics.

Training set: 1,460 houses  
Test set: 1,459 houses

## Files

- `train.csv` - training data with house prices
- `test.csv` - test data without prices (for predictions)
- `data_description.txt` - explains what each feature means
- Main analysis code in the notebook

## Results So Far

- Random Forest baseline: ~20,221 RMSE
- XGBoost can achieve lower scores with proper tuning
