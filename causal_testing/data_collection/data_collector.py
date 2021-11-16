from abc import ABC, abstractmethod
import pandas as pd


class DataCollector(ABC):

    def __init__(self, df=pd.DataFrame()):
        self.df = df

    @abstractmethod
    def collect_data(self, **kwargs):
        """
        Populate the dataframe with execution data.
        :return:
        """
        ...
