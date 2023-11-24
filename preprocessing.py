from abc import ABC, abstractmethod
import pandas as pd
import json


class AbstractNFLPreprocessing(ABC):
    def __init__(self, csv_file_list: list) -> None:
        super().__init__()
        self.combined_df = None
        self.make_combined_df()
        self.drop_irrelevant_observations()
        self.impute_missing_values()

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
    def drop_irrelevant_features(self):
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
        self.combined_df.drop(
            self.combined_df[self.combined_df["safety"] == 1].index,
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
        self.combined_df.drop(
            self.combined_df[self.combined_df["lateral_rush"] == 1].index,
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

    def impute_missing_values(self):
        # please refer to imputation.txt for source
        closed_roof = [
            "2021_01_JAX_HOU",
            "2021_01_SEA_IND",
            "2021_02_LA_IND",
            "2021_03_CAR_HOU",
            "2021_05_NE_HOU",
            "2021_08_LA_HOU",
            "2021_09_NYJ_IND",
            "2021_10_JAX_IND",
            "2021_11_NE_ATL",
            "2021_12_NYJ_HOU",
            "2021_12_TB_IND",
            "2021_13_IND_HOU",
            "2021_13_TB_ATL",
            "2021_15_NE_IND",
            "2021_16_LAC_HOU",
            "2021_17_LV_IND",
            "2021_18_NO_ATL",
            "2021_18_TEN_HOU",
        ]

        open_roof = [
            "2021_01_PHI_ATL",
            "2021_04_WAS_ATL",
            "2021_06_HOU_IND",
            "2021_08_CAR_ATL",
            "2021_08_TEN_IND",
            "2021_14_SEA_HOU",
            "2021_16_DET_ATL",
        ]

        def update_roof(row):
            if any(game_id in row["game_id"] for game_id in open_roof):
                return "open"
            if any(game_id in row["game_id"] for game_id in closed_roof):
                return "closed"
            return row["roof"]

        # Apply the function to update 'roof'
        self.combined_df.loc[
            self.combined_df["roof"].isna(), "roof"
        ] = self.combined_df[self.combined_df["roof"].isna()].apply(update_roof, axis=1)

    def drop_irrelevant_features(self):
        with open("drop_columns.json") as file:
            column_drop_dict = json.load(file)
            for drop_column_list in column_drop_dict.values():
                self.combined_df.drop(drop_column_list, axis=1, inplace=True)
