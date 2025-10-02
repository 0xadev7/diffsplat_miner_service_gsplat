import yaml, os

_CFG_PATH = os.path.join(os.path.dirname(__file__), "pipeline_config.yaml")
with open(_CFG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)