# Boston Neighbourhood House Price Prediction
Regression model to predict median home values in Boston neighbourhoods from 13 socioeconomic and environmental features — built using 8 tuned regression models and CRISP-DM methodology.

## Project Overview
Accurate property valuation is critical for buyers, sellers, mortgage lenders, and urban planners. This project builds a regression pipeline on the Boston Housing dataset, applying careful feature transformation and comparing 8 regression models to find the best predictor of home prices.
Problem TypeRegressionDatasetHousing.csv — 506 samples, 13 featuresTargetMEDV — Median Home Value in $1,000sMethodologyCRISP-DM

## Dataset Features
FeatureDescriptionCRIMPer capita crime rateZNProportion of residential land zoned for large lotsINDUSProportion of non-retail business acresCHASCharles River proximity (binary)NOXNitric oxide concentrationRMAverage number of rooms per dwellingAGEProportion of owner-occupied units built before 1940DISWeighted distances to employment centresRADAccessibility index to radial highwaysTAXProperty tax rate per $10,000PTRATIOPupil-teacher ratioBProportion of Black residents (historical)LSTATPercentage lower-status populationMEDVTarget — Median home value in $1,000s

## Business Understanding
A real estate analytics firm wants to estimate median home values across Boston neighbourhoods to support automated property valuation for mortgage applications, help buyers identify undervalued areas, and assist urban planners in understanding how factors like crime rate and school quality affect housing prices.

## Pipeline (CRISP-DM)
Data Cleaning

Removed 5 duplicate rows
Filled missing values with column median — robust to outliers

Skewness Treatment

Log1p transformation applied to CRIM, ZN, DIS and B (reflected then log1p)
MEDV target also log-transformed — predictions reversed using np.expm1()
Cube root transformation applied to CRIM and ZN where log1p was insufficient

Feature Engineering

Columns grouped into continuous, discrete and categorical for targeted analysis
Features (X) and target (y) separated after all transformations
Train/test split 80/20 with random_state=42
StandardScaler applied — fit on train only, transform on test

## Modelling — 8 regression models compared
ModelDetailLinear RegressionBaselinePolynomial RegressionDegree=2 — best modelLasso RegressionL1 regularisation, GridSearchCV on alphaRidge RegressionL2 regularisation, GridSearchCV on alphaElasticNetL1+L2 combined, GridSearchCV on alpha and l1_ratioDecision TreeNon-linear baselineRandom ForestBagging ensembleKNN Regressorn_neighbors=5

## Results

Best Model: Polynomial Linear Regression (degree=2)
Metric: R² Score on train, CV and test sets
Final prediction reversed from log scale using np.expm1() to return actual price in $1,000s


## Sample Prediction
Input: CRIM=0.1, ZN=18, INDUS=2.3, CHAS=0, NOX=0.45,
       RM=6.5, AGE=65, DIS=4.5, RAD=1, TAX=300,
       PTRATIO=15.3, B=390, LSTAT=5

Predicted Median Home Value: ~$28,400

## Tech Stack

Language: Python
Libraries: pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn
Models: Linear Regression, Polynomial Regression, Lasso, Ridge, ElasticNet, Decision Tree, Random Forest, KNN
Techniques: CRISP-DM, Log and Cube Root Transformation, Polynomial Features, GridSearchCV, Cross-Validation, StandardScaler
