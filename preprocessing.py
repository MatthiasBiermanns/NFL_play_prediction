from abc import ABC, abstractmethod
import pandas as pd
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from loguru import logger
import scipy
import sklearn


class AbstractNFLPreprocessing(ABC):
    """abstract class to create layout for NFLPreprocessing class"""

    def __init__(self, csv_file_list: list, test_size: float = 0.25) -> None:
        super().__init__()
        logger.info("--- Executing Preprocessing Steps ---")

        # initialization
        self.combined_df = None
        self.run_df = None
        self.pass_df = None
        self.run_test = None
        self.run_train = None
        self.pass_test = None
        self.pass_train = None
        self.encoder = None
        self.minmax_scaler = None
        self.standardizer = None
        self.prepro = None
        self.pipeline = None

        # apply preprocessing steps
        self.make_combined_df(csv_file_list)
        self.drop_irrelevant_observations()
        self.insert_missing_values()
        self.drop_irrelevant_features()
        self.clear_nas()
        self.split_into_run_and_pass_dataframes()
        self.run_train, self.run_test = self.split_into_test_and_training_dataframes(
            self.run_df, test_size
        )
        self.pass_train, self.pass_test = self.split_into_test_and_training_dataframes(
            self.pass_df, test_size
        )
        logger.info("Preparing pipeline")
        self.encoder = self.make_encoder()
        self.minmax_scaler = self.make_minmax_scaler()
        self.standardizer = self.make_standardizer()
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
    def make_encoder(self):
        pass

    @abstractmethod
    def outlier_removal(self):
        pass

    @abstractmethod
    def make_minmax_scaler(self):
        pass

    @abstractmethod
    def make_standardizer(self):
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

    def clear_nas(self):
        logger.info("Claering obervations with NAs")
        self.combined_df.dropna(inplace=True)
        logger.info("Successfully cleared observations with NAs")

    def split_into_run_and_pass_dataframes(self):
        """
        splits self.combined_df into run and pass dataframes
        """
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

    def split_into_test_and_training_dataframes(
        self, df: pd.DataFrame, test_size: float = 0.25
    ):
        """
        splits dataframe into test and training dataset

        Args:
            df (pd.DataFrame): dataframe to be split into training and test dataframe
            test_size (float, optional): size of the test set. Defaults to 0.25.

        Returns:
            pd.DataFrame: training and test set
        """
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
        return training_df, test_df

    def outlier_removal(self, training_df, factor_iqr: float = 3.0):
        logger.info("Removing outliers")
        for column in training_df.columns:
            # parse columns to a numeric data type
            try:
                training_df[column] = pd.to_numeric(training_df)
                # check whether these are of type boolean and parse them if so
                is_boolean = all(value in [0.00, 1.00] for value in training_df[column])
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
                    training_df = training_df.loc[
                        ~(
                            (training_df[column] < lower_bound)
                            | (training_df[column] > upper_bound)
                        )
                    ]

            except ValueError:
                training_df[column] = training_df[column].apply(str)

        logger.info("Successfully removed outliers")
        return training_df

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
        datafrme_to_be_transformed: pd.DataFrame,
    ) -> pd.DataFrame:
        """requires a pipeline without a model!
        returns dataframe processed by pipeline with feature names

        Args:
            pipeline (sklearn.pipeline.Pipeline):
            datafrme_to_be_transformed (pd.Dataframe):

        Returns:
            pd.DataFrame: preprocessed dataframe with all feature names
        """
        transformed = pipeline.transform(datafrme_to_be_transformed)
        transformed = transformed.todense()
        feature_names = self.get_prepro_feature_names_from_pipeline()
        return pd.DataFrame(transformed, columns=feature_names)

    def make_encoder(self):
        # create ColumnTransformer
        encoder = Pipeline(steps=[("encoder", OneHotEncoder(drop="first"))])
        return encoder

    def make_standardizer(self):
        standardizer = Pipeline(steps=[("standardization", StandardScaler())])
        return standardizer

    def make_minmax_scaler(self):
        minmax_scaler = Pipeline(steps=[("minmax", MinMaxScaler())])
        return minmax_scaler

    def make_preprocessor(self):
        """combines make_encoder(), make_standardizer(), make_minmax_scaler() into a single
        ColumnTransformer

        Returns:
            ColumnTransformer
        """
        numeric_features = [
            "yardline_100",
            "game_seconds_remaining",
            "down",
            "ydstogo",
            "td_prob",
            "wpa",
            "score_differential",
        ]
        return ColumnTransformer(
            transformers=[
                (
                    "encoder",
                    self.encoder,
                    ["posteam", "posteam_type", "defteam", "roof"],
                ),
                ("standardization", self.standardizer, ["score_differential"]),
                ("minmax", self.minmax_scaler, numeric_features),
            ]
        )

    def make_preprocessing_pipeline(self):
        return Pipeline(steps=[("preprocessor", self.prepro)])
