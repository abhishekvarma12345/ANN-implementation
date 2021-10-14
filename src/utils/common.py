import logging
import yaml

def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    logging.info(f"read the content successfully from {config_path}")
    return content

    