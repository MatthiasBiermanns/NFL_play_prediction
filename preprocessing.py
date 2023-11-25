from abc import ABC, abstractmethod
import pandas as pd
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from loguru import logger
import scipy


class AbstractNFLPreprocessing(ABC):
    def __init__(self, csv_file_list: list, test_size: float = 0.25) -> None:
        super().__init__()
        logger.info("--- Loading Preprocessing Steps ---")
        # initialization
        self.combined_df = None
        self.run_df = None
        self.pass_df = None
        self.run_test = None
        self.run_train = None
        self.pass_test = None
        self.pass_train = None
        self.encoder = None
        self.normalizer = None
        self.pipeline = None
        self.prepro_col_transf = None

        # apply preprocessing steps
        self.make_combined_df(csv_file_list)
        self.drop_irrelevant_observations()
        self.insert_missing_values()
        self.drop_irrelevant_features()
        self.clear_nas()
        self.split_into_run_and_pass_dataframes()
        self.run_test, self.run_train = self.split_into_test_and_training_dataframes(
            self.run_df, test_size
        )
        self.pass_test, self.pass_train = self.split_into_test_and_training_dataframes(
            self.pass_df, test_size
        )
        self.encoder = self.encoding_of_categorical_features()
        self.normalizer = self.apply_normalization()
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
    def drop_irrelevant_features(self):
        pass

    @abstractmethod
    def clear_nas(self):
        pass

    @abstractmethod
    def split_into_run_and_pass_dataframes(self):
        pass

    @abstractmethod
    def split_into_test_and_training_dataframes(self, df, test_size):
        pass

    @abstractmethod
    def encoding_of_categorical_features(self):
        pass

    @abstractmethod
    def outlier_removal(self):
        pass

    @abstractmethod
    def apply_normalization(self):
        pass

    @abstractmethod
    def make_pipeline(self):
        pass


class NFLPreprocessing(AbstractNFLPreprocessing):
    def __init__(self, file_list: list) -> None:
        super().__init__(file_list)

    def make_combined_df(self, csv_file_list):
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
        logger.info("Removing irrelevant observations")
        # remove non-pass and non-run plays from dataframe
        self.combined_df.drop(
            self.combined_df[
                ~self.combined_df["play_type"].isin(["pass", "run"])
            ].index,
            axis=0,
            inplace=True,
        )

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
            self.combined_df[self.combined_df["replay_or_challenge"] == 1].index,
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

    def insert_missing_values(self):
        logger.info("Imputing missing values")
        # please refer to insertion.txt for source of values
        with open("roof_insertion.json") as file:
            insertion = json.load(file)

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
        logger.info("Dropping irrelevant features")
        with open("drop_columns.json") as file:
            column_drop_dict = json.load(file)
            for drop_column_list in column_drop_dict.values():
                self.combined_df.drop(drop_column_list, axis=1, inplace=True)
        logger.info("Successfully dropped irrelevant features")

    def clear_nas(self):
        logger.info("Claering obervations with NAs")
        self.combined_df.dropna(inplace=True)
        logger.info("Successfully cleared observations with NAs")

    def encoding_of_categorical_features(self):
        logger.info("Encoding categorical features")
        # create ColumnTransformer
        """ encoder = ColumnTransformer(
            transformers=[
                (
                    "encoder",
                    OneHotEncoder(drop="first"),
                    ["posteam", "posteam_type", "defteam", "roof"],
                )
            ],
            remainder="passthrough",  # include non-transformed columns
        ) """

        encoder = [
            (
                "encoder",
                OneHotEncoder(drop="first"),
                ["posteam", "posteam_type", "defteam", "roof"],
            )
        ]

        logger.info("Successfully encoded categorical features")
        return encoder

    def split_into_run_and_pass_dataframes(self):
        logger.info("Splitting into run and pass dataframes")
        self.run_df = (
            self.combined_df[self.combined_df["play_type"] == "run"]
            .drop("play_type", axis=1)
            .reset_index()
        )
        self.pass_df = (
            self.combined_df[self.combined_df["play_type"] == "pass"]
            .drop("play_type", axis=1)
            .reset_index()
        )
        logger.info("Successfully split into run and pass dataframes")

    def split_into_test_and_training_dataframes(self, df, test_size):
        logger.info("Splitting into train and test dataframes")
        # set seed for reproducability
        seed = 1887  # nur der HSV

        # Shuffle the DataFrame
        df = df.sample(frac=1, random_state=seed)

        # calculate size of test df
        split_size = int(test_size * len(df))

        # Split the DataFrame
        test_df = df.head(split_size)
        training_df = df.tail(len(df) - split_size)
        test_df.reset_index(inplace=True)
        training_df.reset_index(inplace=True)
        logger.info("Successfully split into train and test dataframes")
        return test_df, training_df

    def apply_normalization(self):
        logger.info("Normalizing numerical features")
        numeric_features = [
            "yardline_100",
            "game_seconds_remaining",
            "down",
            "ydstogo",
            "td_prob",
            "wpa",
        ]
        """ normalizer = ColumnTransformer(
            transformers=[
                ("standardization", StandardScaler(), ["score_differential"]),
                ("minmax", MinMaxScaler(), numeric_features),
            ],
            remainder="passthrough",  # include non-transformed columns
        ) """

        normalizer = [
            ("standardization", StandardScaler(), ["score_differential"]),
            ("minmax", MinMaxScaler(), numeric_features),
        ]
        logger.info("Successfully normalized numerical features")
        return normalizer

    def outlier_removal(self, training_df):
        logger.info("Removing outliers")
        # TODO @ Joel
        logger.info("Successfully removed outliers")
        return training_df

    def make_pipeline(self):
        preprocessing = self.encoder + self.normalizer
        self.prepro_col_transf = ColumnTransformer(
            transformers=preprocessing, remainder="passthrough"
        )
        return Pipeline([("preprocessing", self.prepro_col_transf)])

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
            for item in self.prepro_col_transf.get_feature_names_out()
        ]

    def get_dataframe_from_pipeline(
        self, dataframe_transformed_by_pipeline: scipy.sparse._csr.csr_matrix
    ) -> pd.DataFrame:
        """requires the output of the pipeline.transform(pd.DataFrame) as input
        and returns a pandas DataFrame with the feature names and the preprocessed features


        Args:
            dataframe_transformed_by_pipeline (scipy.sparse._csr.csr_matrix): output of pipeline.transform(pd.DataFrame)

        Returns:
            pd.DataFrame: preprocessed dataframe with all feature names
        """
        transformed = dataframe_transformed_by_pipeline.todense()
        feature_names = self.get_prepro_feature_names_from_pipeline()
        return pd.DataFrame(transformed, columns=feature_names)
