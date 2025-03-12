import os
import pandas
import yaml
import pandas as pd

from zlogger.logger import ZLogger
import configparser

path_file = "config/logging.ini"
config = configparser.ConfigParser()
config.read(path_file)
logger = ZLogger("hotel_reservation_system", config)


def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path")
        
        with open(file_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Succesfully read the YAML file")
            return config
    
    except Exception as e:
        logger.error("Error while reading YAML file")
    

def load_data(path):
    try:
        logger.info("Loading data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading the data {e}")
    