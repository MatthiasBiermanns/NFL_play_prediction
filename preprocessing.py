from abc import ABC, abstractmethod
import pandas as pd


class AbstractNFLPreprocessing(ABC):
    def __init__(self, csv_file_list: list) -> None:
        super().__init__()
        self.combined_df= None
        self.make_combined_df()
        self.drop_irrelevant_observations()
        
    @abstractmethod
    def make_combined_df(self, csv_file_list):
        pass

    @abstractmethod
    def drop_useless_observations(self):
        pass

    @abstractmethod
    def impute_missing_values(self):
        pass

    @abstractmethod
    def drop_useless_features(self):
        pass

    @abstractmethod
    def clear_nas(self):
        pass

    @abstractmethod
    def split_into_run_and_pass_dataframes(self):
        pass

    @abstractmethod
    def split_into_test_and_training_dataframes(self):
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
        # load each csv file as a dataframe and collect them in a list
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        # combine all dataframes into a single one
        self.combined_df = pd.concat(dataframes, axis=0)
    
    
    
    def drop_irrelevant_observations(self, dataframe):
        # remove non-pass and non-run plays from dataframe
        self.combined_df.drop(
            dataframe[~dataframe["play_type"].isin(["pass", "run"])].index,
            axis=0,
            inplace=True,
        )
        # drop safeties
        self.combined_df.drop(self.combined_df[self.combined_df["safety"] == 1].index, axis=0, inplace=True)

        # drop plays with penalties
        self.combined_df.drop(self.combined_df[self.combined_df["penalty"] == 1].index, axis=0, inplace=True)

        # drop plays with laterals
        self.combined_df.drop(
            self.combined_df[self.combined_df["lateral_reception"] == 1].index, axis=0, inplace=True
        )
        self.combined_df.drop(
            self.combined_df[self.combined_df["lateral_rush"] == 1].index, axis=0, inplace=True
        )

        # drop plays with replays or challenges
        self.combined_df.drop(
            self.combined_df[self.combined_df["replay_or_challenge"] == 1].index, axis=0, inplace=True
        )

        # drop two point conversion plays
        self.combined_df.drop(
            self.combined_df[~self.combined_df["two_point_conv_result"].isna()].index, axis=0, inplace=True
        )








