from causal_testing.data_collection.data_collector import DataCollector


class ExperimentalDataCollector(DataCollector):

    def __init__(self):
        super().__init__()

    def collect_data(self, **kwargs):
        pass
