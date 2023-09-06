import os


class DatasetCatalog(object):
    default_data_dir = os.path.expanduser('~/Datasets')
    DATA_DIR = os.environ.get('DATASET_HOME', default_data_dir)

    @staticmethod
    def get(name: str):
        raise RuntimeError("Dataset not available: {}".format(name))