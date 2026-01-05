import os
from typing import Optional

import mlflow
from skops.io import load as skops_load

from moneyscam.settings import inference_config


class ModelLoader:
    def __init__(self, local_path: Optional[str] = None, mlflow_uri: Optional[str] = None):
        self.local_path = local_path or inference_config.model_path
        self.mlflow_uri = mlflow_uri
        self.model = None

    def load(self):
        if self.mlflow_uri:
            self.model = mlflow.pyfunc.load_model(self.mlflow_uri)
        else:
            if not os.path.exists(self.local_path):
                raise FileNotFoundError(f"Model not found at {self.local_path}")
            self.model = skops_load(self.local_path, trusted=True)
        return self.model

    def reload(self):
        self.model = None
        return self.load()
