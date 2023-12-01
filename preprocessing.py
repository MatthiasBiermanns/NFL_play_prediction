"""
author: Thilo
use this script as follows:
- import: from preprocessing import NFLPreprocessing
- usage: i) create an instance of the NFLPreprocessing class by providing a list of csv files and optionally the size of the test set
        ii) this instance contains the following attributes (marked with * are especially important for further usage such as model development):

        combined_df = dataframes of all years combined into one large dataframe
        *run_df = complete run dataframe
        *pass_df = complete pass dataframe
        encoder = one hot encoding
        minmax_scaler = MinMax scaler
        standardizer = Standardizer
        prepro = Column Transformer of the previous three

        iii) the instance contains the following methods (marked with * are especially important for further usage such as model development):

        make_combined_df(self, csv_file_list: list): combines the dataframes from the list into one
        drop_irrelevant_observations(self): drops observations which are not relevant for our case
        insert_missing_values(self): inserts missing values for roof attribute
        drop_irrelevant_features(self): drops irrelevant features, such as ids and names
        clear_nas(self): removes remaining NAs
        split_into_run_and_pass_dataframes(self): splits combined_df into run and passing dataframe depending on the play_type attribute
        outlier_sampler_iqr(self, X, y): removes outliers on the provided training_df according to the initially provided iqr factor
        make_preprocessor(self): makes Column Transformer of the previous three
        * make_preprocessing_pipeline(self): returns a Pipeline object by providing the steps stored in self.prepro
        * get_prepro_feature_names_from_pipeline(self) -> list: returns the feature names from the ColumnTransformer containing the preprocessing pipeline steps
        * get_dataframe_from_preprocessing_pipeline(self,
        pipeline: sklearn.pipeline.Pipeline,
        datafrme_to_be_transformed: pd.DataFrame,
    ) -> pd.DataFrame: returns a pandas Dataframe with the provided pipeline applied to the provided dataframe

"""


from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import json
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from loguru import logger
import sklearn
import re
from imblearn import FunctionSampler
import imblearn.pipeline

class AbstractNFLPreprocessing(ABC):
    """abstract class to create layout for NFLPreprocessing class"""

    def __init__(self, csv_file_list: list, factor_iqr: float = 3.0) -> None:
        super().__init__()
        logger.info("--- Executing Preprocessing Steps ---")

        # initialization
        self.combined_df = None
        self.run_df = None
        self.pass_df = None
        self.prepro = None

        self.encoder = OneHotEncoder(drop="first")
        self.outlier_remover = FunctionSampler(func=self.outlier_sampler_iqr, validate=False)
        self.minmax_scaler = MinMaxScaler()
        self.standardizer = StandardScaler()

        self.factor_iqr = factor_iqr


        # apply preprocessing steps
        self.make_combined_df(csv_file_list)
        self.drop_irrelevant_observations()
        self.insert_missing_values()
        self.transform_columns()
        self.drop_irrelevant_features()
        self.split_into_run_and_pass_dataframes()
        self.clear_nas(self.run_df)
        self.clear_nas(self.pass_df)
        logger.info("Preparing pipeline")
        self.prepro = self.make_preprocessor()
        logger.info("Successfully prepared pipeline")
        logger.info("--- Successfully Loaded Preprocessing Steps ---")

    @abstractmethod
    def make_combined_df(self, csv_file_list):
        pass

    @abstractmethod
    def drop_irrelevant_observations(self):
        pass

    @abstractmethod
    def insert_missing_values(self):
        pass

    @abstractmethod
    def transform_columns(self):
        pass

    @abstractmethod
    def drop_irrelevant_features(self):
        pass

    @abstractmethod
    def clear_nas(self):
        pass

    @abstractmethod
    def split_into_run_and_pass_dataframes(self):
        pass

    @abstractmethod
    def make_preprocessor(self):
        pass

    @abstractmethod
    def make_preprocessing_pipeline(self):
        pass


class NFLPreprocessing(AbstractNFLPreprocessing):
    def __init__(self, file_list: list) -> None:
        super().__init__(file_list)

    def make_combined_df(self, csv_file_list: list):
        """combines dataframes from csv list into one large dataframe

        Args:
            csv_file_list (list): list of filenames of csv files
        """
        logger.info("Loading csv files")

        # load each csv file as a dataframe and collect them in a list
        dataframes = []
        for csv_file in csv_file_list:
            df = pd.read_csv(csv_file)
            dataframes.append(df)

        # combine all dataframes into a single one
        self.combined_df = pd.concat(dataframes, axis=0)
        logger.info("Successfully loaded csv files")

    def drop_irrelevant_observations(self):
        """
        drops non pass- or run plays and plays with rare or unpredictable outcomes
        """
        logger.info("Removing irrelevant observations")
        # remove non-pass and non-run plays from dataframe
        self.combined_df = self.combined_df[
            self.combined_df["play_type"].isin(["pass", "run"])
        ]
        self.combined_df = self.combined_df.reset_index(drop=True)

        # drop plays with penalties
        self.combined_df.drop(
            self.combined_df[self.combined_df["penalty"] == 1].index,
            axis=0,
            inplace=True,
        )

        # drop plays with laterals
        self.combined_df.drop(
            self.combined_df[self.combined_df["lateral_reception"] == 1].index,
            axis=0,
            inplace=True,
        )
        # drop plays with replays or challenges
        self.combined_df.drop(
            self.combined_df[self.combined_df["aborted_play"] == 1].index,
            axis=0,
            inplace=True,
        )
        # drop two point conversion plays
        self.combined_df.drop(
            self.combined_df[~self.combined_df["two_point_conv_result"].isna()].index,
            axis=0,
            inplace=True,
        )
        logger.info("Successfully deleted irrelevant observations")

    def transform_columns(self):
        logger.info("Transforming columns")
        # adjust the spread line to the view of the team with possession of the ball
        self.combined_df.loc[
            self.combined_df["posteam_type"] == "away", "spread_line"
        ] *= -1

        # transform drive start yard line
        def transform_dsyl(row):
            value = row["drive_start_yard_line"]
            if pd.notna(value) and isinstance(
                value, str
            ):  # Check if not NaN and is a string
                match = re.match(r"([A-Z]+)(\d+)", value)
                if match:
                    team, number = match.groups()
                    return int(number) if row["posteam"] == team else 100 - int(number)
                elif " " in value:
                    # Handle the case where there is a space but no match
                    return int(value.split()[1])
                else:
                    # Handle the case where there is no space (e.g., '50')
                    return int(value)
            else:
                # Handle the case where the value is NaN or not a string
                return value

        self.combined_df["drive_start_yard_line"] = self.combined_df.apply(
            transform_dsyl, axis=1
        )

        logger.info("Successfully transformed columns")

    def insert_missing_values(self):
        """inserts missing values for the roof variable.
        please refer to insertion.txt for source of values"""

        logger.info("Inserting missing values")

        # open json file
        with open("roof_insertion.json") as file:
            insertion = json.load(file)

            # update values for roof
            def update_roof(row):
                if any(game_id in row["game_id"] for game_id in insertion["open_roof"]):
                    return "open"
                if any(
                    game_id in row["game_id"] for game_id in insertion["closed_roof"]
                ):
                    return "closed"
                return row["roof"]

            # Apply the function to update 'roof'
            self.combined_df.loc[
                self.combined_df["roof"].isna(), "roof"
            ] = self.combined_df[self.combined_df["roof"].isna()].apply(
                update_roof, axis=1
            )
            logger.info("Successfully inserted missing values")

    def drop_irrelevant_features(self):
        """
        drops features irrelevant for task at hand
        """
        logger.info("Dropping irrelevant features")
        with open("drop_columns.json") as file:
            column_drop_dict = json.load(file)
            for drop_column_list in column_drop_dict.values():
                self.combined_df.drop(drop_column_list, axis=1, inplace=True)
        logger.info("Successfully dropped irrelevant features")

    def clear_nas(self, dataframe):
        logger.info("Clearing obervations with NAs")
        dataframe.dropna(inplace=True)
        logger.info("Successfully cleared observations with NAs")

    def split_into_run_and_pass_dataframes(self):
        """
        splits self.combined_df into run and pass dataframes
        """
        logger.info("Splitting into run and pass dataframes")
        self.run_df = (
            self.combined_df[self.combined_df["play_type"] == "run"]
            .drop(["play_type", "passer_id"], axis=1)
            .reset_index(drop=True)
        )
        pass_df = (
            self.combined_df[self.combined_df["play_type"] == "pass"]
            .drop("play_type", axis=1)
            .reset_index(drop=True)
        )

        # Count the number of passes per passer and game
        passer_game_counts = (
            pass_df.groupby(["passer_id", "season"]).size().reset_index(name="count")
        )

        # Filter the passers with at least 224 pass attempts in at least one season
        passers_with_min_passes = set(
            passer_game_counts[passer_game_counts["count"] >= 224]["passer_id"]
        )

        # Filter out the pass plays where the passer_id is not in the passers_with_min_passes set
        self.pass_df = pass_df[pass_df["passer_id"].isin(passers_with_min_passes)].drop(
            "passer_id", axis=1
        )
        logger.info("Successfully split into run and pass dataframes")

    def get_prepro_feature_names_from_pipeline(self) -> list:
        """only works when model has not been added to pipeline!

        Returns:
            list: list of feature names
        """

        return [
            item.replace("encoder__", "")
            .replace("standardization__", "")
            .replace("minmax__", "")
            .replace("remainder__", "")
            for item in self.prepro.get_feature_names_out()
        ]

    def get_dataframe_from_preprocessing_pipeline(
        self,
        pipeline: sklearn.pipeline.Pipeline,
        dataframe_to_be_transformed: pd.DataFrame,
    ) -> pd.DataFrame:
        """requires a pipeline without a model!
        returns dataframe processed by pipeline with feature names

        Args:
            pipeline (sklearn.pipeline.Pipeline):
            datafrme_to_be_transformed (pd.Dataframe):

        Returns:
            pd.DataFrame: preprocessed dataframe with all feature names
        """
        transformed = pipeline.transform(dataframe_to_be_transformed)
        transformed = transformed.todense()
        feature_names = self.get_prepro_feature_names_from_pipeline()
        return pd.DataFrame(transformed, columns=feature_names)
    
    def outlier_sampler_iqr(self, X, y):
        features = X.columns
        df = X.copy()
        df['Outcome'] = y

        indices = [x for x in df.index]
        out_indexlist = []

        for col in features:
            # ignore string-type features
            if is_string_dtype(X[col]):
                continue

            #Using nanpercentile instead of percentile because of nan values
            Q1 = np.nanpercentile(df[col], 25.)
            Q3 = np.nanpercentile(df[col], 75.)

            cut_off = (Q3 - Q1) * self.factor_iqr
            upper, lower = Q3 + cut_off, Q1 - cut_off

            outliers_index = df[col][(df[col] < lower) | (df[col] > upper)].index.tolist()
            #outliers = df[col][(df[col] < lower) | (df[col] > upper)].values
            out_indexlist.extend(outliers_index)

        #using set to remove duplicates
        out_indexlist = list(set(out_indexlist))

        clean_data = np.setdiff1d(indices,out_indexlist)

        return X.loc[clean_data], y.loc[clean_data]

    def make_preprocessor(self):
        """combines make_encoder(), make_standardizer(), make_minmax_scaler() into a single
        ColumnTransformer

        Returns:
            ColumnTransformer
        """
        with open("encoding_normalization.json") as file:
            encoding_normalization = json.load(file)
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "encoder",
                        self.encoder,
                        encoding_normalization["one_hot_features"],
                    ),
                    (
                        "standardization",
                        self.standardizer,
                        encoding_normalization["standardizing_features"],
                    ),
                    (
                        "minmax",
                        self.minmax_scaler,
                        encoding_normalization["minmax_features"],
                    ),
                ],
            )
        return preprocessor

    def make_preprocessing_pipeline(self):
        return imblearn.pipeline.Pipeline(steps=[
                (
                    "outlier_remover",
                    self.outlier_remover,
                ),
                (
                    "preprocessor", 
                    self.prepro
                )
            ]
        )
