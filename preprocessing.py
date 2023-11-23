from abc import ABC, abstractmethod
import pandas as pd


class AbstractNFLPreprocessing(ABC):
    def __init__(self, file_list: list) -> None:
        super().__init__()
        self.file_list = file_list

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
