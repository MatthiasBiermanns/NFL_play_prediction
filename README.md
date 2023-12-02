# NFL_play_prediction

Student Project in Data Mining of the University Mannheim to predict the space gained by different plays of the offense

## Preprocessing

## Model 

### Examined Models

- Linear Regression
- Polynomial Regression
- Random Forest
- XGBoost
- Artificial Neural Network - MLP Regression

### Evaluation Procedure

The evaluation was applied under those agreed upon criteria:
- random state: 42
- train-test-split: 80% / 20 %

For each model the given procedure was applied each for pass- and run-play-prediction.

1. Proof of concept by training one simple model with a subset of the overall data. This model has no impact on the evaluation.
2. Evaluating the best hyper-parameters with nested cv approach.
    - `k_fold = 3` due to the training time
    - `score = 'neg_mean_squared_error'`
    - `data_fraction >= 0.25` depending on how time-consuming the model training is and how many hyper-parameter combinations are evaluated 
    - `outlier_removal__strict_columns = []` 
3. Analyzation of the results & best parameters.
4. Training the model with the best parameters and a `k_fold = 3` and the full dataset
5. Evaluate the model results:
    - show decision tree
    - feature importance analysis 
    - etc.