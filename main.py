
from train import Trainer
from ray import tune
from pathlib import Path
import torch

import argparse

from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

import yaml
from settings import Settings

import warnings
warnings.filterwarnings("ignore",category=UserWarning, module="torch")

setting = Settings()
#import logging
#logging.getLogger(tune.utils.trainable.__name__).disabled = True
#print(logging.Logger.manager.loggerDict)
#logging.disable(logging.INFO)

class Monitor:
    def __init__(self, search_alg):
        self.project_path = Path(__file__).resolve().parent
        self.model_path = self.project_path / 'models'

        self.search_alg = search_alg
        print("HPO algorithm in use: ",self.search_alg)

    def get_num_gpus(self):
        """Get the number of GPUs available per trial

        Returns:
            int: Number of GPUs per trial, restricted to 1 if gpu 
                 available, else 0
        """
        self.num_gpus = 0
        if torch.cuda.is_available():
            self.num_gpus = 1
            # if torch.cuda.device_count() > 1:
            #     ids = list(range(1, torch.cuda.device_count()))
            #     self.num_gpus = len(ids)

        return self.num_gpus

    def set_space(self):     
        """Extract the search space from the configuration file
        with right distribution/values for every tunable hyperparameter.

        Returns:
            dict: A dictionary with hyperparameter as keys and search domain
            as value.
        """
        space = {
            "lstm" : {
                    "embedding_dim" : None,
                    "hidden_dim" : None,
                    "num_layers" : None,
                    "bidirectional" : True,
                    "dropout" : None
                    },
            "batch_size" : None,
            "learning_rate": None
            }

        for hp, domain0 in setting.hyperparamters.items():
            if hp == "lstm":
                for lstm_hp, domain1 in domain0.items():
                    space["lstm"][lstm_hp] = self.get_values(domain1)
            else:
                space[hp] = self.get_values(domain0)

        self.space = space
        return self.space

    def get_values(self, domain):
        """Get value(s) or generator object for each hyperparameter
        to sample from.

        Args:
            domain ([dict]): A dictionary with keys 'grid', 'args' and 'distribution'
            made available in object of Settings class from configuration file.

        Returns:
            [int/float/GeneratorObject]: An integer, float or a generator object
            to sample from the provided list of values.
        """
        if self.search_alg == "grid":
            return self.get_grid(domain)
        else:
            return self.get_distribution(domain)

    def get_grid(self, domain):
        """Get value(s) or generator object for each hyperparameter
        to sample from, for grid search.

        Args:
            domain ([dict]): A dictionary with keys 'grid', 'args' and 'distribution'
            made available in object of Settings class from configuration file.

        Returns:
            [int/float/CategoricalObject]: An integer, float or a generator object
            to sample from the provided list of values.
        """
        if isinstance(domain["grid"], list):
            return tune.grid_search(domain["grid"])
        else:
            return domain["grid"]
        

    def get_distribution(self, domain):
        """Get value(s) or generator object for each hyperparameter
        to sample from, for Random, Hyperband or BOHB search algorithm

        Args:
            domain ([dict]): A dictionary with keys 'grid', 'args' and 'distribution'
            made available in object of Settings class from configuration file.

        Returns:
            [int/float/GeneratorObject]: An integer, float or a generator object
            to sample from the provided list of values.
        """
        distributions = {
            "uniform": tune.uniform,  # Uniform float between -5 and -1
            "quniform": tune.quniform,  # Round to increments of 0.2
            "loguniform": tune.loguniform,  # Uniform float in log space
            "qloguniform": tune.qloguniform,  # Round to increments of 0.0005
            "randn": tune.randn,  # Normal distribution with mean 10 and sd 2
            "qrandn": tune.qrandn,  # Round to increments of 0.2
            "randint": tune.randint,  # Random integer between -9 and 15
            "qrandint": tune.qrandint,  # Round to increments of 3 (includes 12)
            "choice": tune.choice,  # Choose one of these options uniformly
            "grid": tune.grid_search  # Search over all these values
        }

        if isinstance(domain["args"], list):
            if domain["distribution"] in distributions:
                if domain["distribution"] in ["choice", "grid"]:
                    dist = distributions[domain["distribution"]](domain["args"])
                else:
                    dist = distributions[domain["distribution"]](*domain["args"])
            else:
                print(f"Wrong distribution given: '{domain['distribution']}'" +
                 f" not a member of accepted distributions {list(distributions.keys())}")
                exit(0)
        # For a single argument, distribution is ignored
        else:
            dist = domain["args"]
        
        return dist

    def get_cli_reporter(self):
        """Create a CLIReporter to manage the output printed to command line.

        Returns:
            [CLIReporter]: A CLIReporter object to be passed to tune.run() as argument.
        """
        self.reporter = tune.CLIReporter(
                                    max_progress_rows=setting.reporter["max_progress_rows"],
                                    metric_columns=setting.reporter["metric_columns"],
                                    max_report_frequency=setting.reporter["max_report_frequency"],
                                    infer_limit=20,
                                    print_intermediate_tables=setting.reporter["print_intermediate_tables"],
                                    )
        return self.reporter

    def tuner(self):
        """Manage entire experiment.
        Schedulers and Algorithm objects are created and used to initialize tune.run()
        Parameters of tune.run() are set from object of Settings class.
        """
        self.set_space()

        algorithm = None
        scheduler = None
        if self.search_alg == "hyperband":
            scheduler = ASHAScheduler(
                time_attr="epoch",
                reduction_factor=setting.hyperband["reduction_factor"],
                grace_period=setting.hyperband["grace_period"],
                max_t=setting.experiment["max_epoch"]
            )
        elif self.search_alg == "bohb":
            algorithm = TuneBOHB(max_concurrent=4)
            scheduler = HyperBandForBOHB(
                time_attr="epoch",
                max_t=setting.experiment["max_epoch"],
                reduction_factor=setting.bohb["reduction_factor"])

        self.analysis = tune.run(
            Trainer,
            name=setting.experiment["name"],
            resources_per_trial={"cpu": 4, "gpu":self.get_num_gpus()},
            config= self.space,
            num_samples=setting.experiment["num_samples"] if self.search_alg != "grid" else 1,
            stop = {"epoch" : setting.experiment["max_epoch"]},
            metric=setting.experiment["metric"],
            checkpoint_freq=setting.experiment["checkpoint_freq"],
            mode=setting.experiment["mode"],
            local_dir=self.model_path.as_posix(),
            max_failures=0,
            reuse_actors=True,
            scheduler=scheduler,
            search_alg=algorithm,
            verbose=setting.experiment["verbose"],
            progress_reporter=self.get_cli_reporter(),
            log_to_file=True,
            loggers=None)

    def get_best_config_all(self):
        """Get the best configuration of all trials, taking into account not only 
        the final accuracy at end of a epoch but at every epoch of all trials.

        Returns:
            [dict]: A dictionary with hyperparameters and best performing values of them
            based on validation accuracy.
        """
        return self.analysis.get_best_config(metric="accuracy",
                                            mode="max", scope="all")

    def write_results_to_file(self):
        """Write final results and all trial info to .csv files.
        """
        # Get a dataframe for the last reported results of all of the trials
        df = self.analysis.results_df
        output_path = self.project_path / 'output/results.csv'
        df.to_csv(path_or_buf=output_path.as_posix(), mode='w')

        # Get a dict mapping {trial logdir -> dataframes} for all trials in the experiment
        all_dataframes = self.analysis.trial_dataframes
        for trial_name, df in all_dataframes.items():
            filename = "_".join(trial_name.split("/")[-1].split("=")[0].split("_")[:-1]) + ".csv"
            output_dir = self.project_path / 'output/trials'
            output_dir.mkdir(parents=True, exist_ok=True)   
            output_path = output_dir / filename

            df.to_csv(path_or_buf=output_path.as_posix(), mode='w')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--search",
                    help=" Specify search algorithm to use.",
                    default= "hyperband",
                    choices=["hyperband", "random", "grid", "bohb"]
                    )
    args = parser.parse_args()
    mon = Monitor(args.search)
    mon.tuner()
    print("\n Best config based on accuracy: ",mon.get_best_config_all())
