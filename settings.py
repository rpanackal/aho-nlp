import yaml
from pathlib import Path

class Settings:
    is_config_read = False

    def __init__(self):
        if not self.is_config_read:
            self.read_config_file()
        self.extract()

    @classmethod
    def read_config_file(self):
        """
        Read the configuration file config.yaml and store the content as a dictionary
        """
        self.project_path = Path(__file__).resolve().parent
        self.config_path = self.project_path / 'config.yaml'

        with open(self.config_path.as_posix(), "r") as yamlfile:
            # config.yaml read into a dictionary
            self.config_file = yaml.load(yamlfile, Loader=yaml.FullLoader)
            #print("Reading configuration successful.")
        
        self.is_config_read = True
        return self.config_file
    

    def extract(self):
        """
        Extracts the sections in config.py to object attributes.
        """
        self.hyperband = self.config_file["HYPERBAND"]
        self.experiment = self.config_file["EXPERIMENT"]
        self.reporter = self.config_file["CLI REPORTER"]
        self.hyperparamters = self.config_file["HYPERPARAMETERS"]
        self.general = self.config_file["GENERAL"]
        self.bohb = self.config_file["BOHB"]
