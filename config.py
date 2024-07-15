from typing import Any
import yaml

class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        self.configs = self._load_config()

    def _load_config(self):
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.configs
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key):
        return self.get(key)

    def __repr__(self):
        return f"ConfigLoader({self.configs})"

config = ConfigLoader("config.yaml")

def get(key: str) -> Any:
    return config[key]
