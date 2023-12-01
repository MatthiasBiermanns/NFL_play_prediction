import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger

class OutlierRemoval(BaseEstimator, TransformerMixin):
    def __init__(self, strict_removal = False) -> None:
        self.strict_removal = strict_removal


    def fit(self, X, y=None):
        _X = X.copy()
        _Y = y.copy()
        _X, _Y = self.outlier_removal(_X, _Y, factor_iqr= 1.5 if self.strict_removal else 3.0 )
        return _X, _Y
    
    def fit_transform(self, X, y=None): # called when training
        _X = X.copy()
        _X = self.outlier_removal(_X, y, factor_iqr= 1.5 if self.strict_removal else 3.0 )
        return _X
    
    def transform(self, X, y=None): # called when testing
        # no implementation, as outliers should not be removed for testing
        _X = X.copy()
        return _X
    
    def outlier_removal(self, training_df, target, factor_iqr: float = 3.0):
        for column in training_df.columns:
            # parse columns to a numeric data type
            try:
                training_df[column] = pd.to_numeric(training_df[column])
                # check whether these are of type boolean and parse them if so
                is_boolean = all(
                    value in [float(0), float(1)] for value in training_df[column]
                )
                if is_boolean:
                    training_df[column] = training_df[column].astype(bool)

                # else remove outliers using the inter quartile range
                else:
                    quantile_value = 0.25
                    q1 = training_df[column].quantile(quantile_value)
                    q3 = training_df[column].quantile(1 - quantile_value)
                    iqr = q3 - q1
                    lower_bound = q1 - factor_iqr * iqr
                    upper_bound = q3 + factor_iqr * iqr
                    
                    # remove outliers
                    training_df = training_df.loc[
                        ~(
                            (training_df[column] < lower_bound)
                            | (training_df[column] > upper_bound)
                        )
                    ]
                    missing_rows = target.index.difference(training_df.index)
                    target = target.drop(index=missing_rows)

            # if parsing the column doesn't work the column is a string and can be handled as such
            except ValueError:
                training_df[column] = training_df[column].apply(str)

        # parse back all columns to type object
        for column in training_df.columns:
            training_df[column] = training_df[column].astype(object)

        logger.info('Outlier removed successfully')
        return training_df, target