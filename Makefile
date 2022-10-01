help : Makefile
	@echo "Welcome to Automatic Hyperparameter Optimization demo"
	@echo
	@echo "To run the demo, use command python3 main.py"
	@echo "By default hyperband search is executed. To select the search alogirthm, use option -s"
	@echo "An Experiment is a single run of main.py file. Evaluation of each sampled configuration is a Trial"
	@echo
	@echo "Available options for command python3 main.py are : "
	@echo "    -h    --help    : Gives brief list of options."
	@echo "    -s    --search  : To specify search algorithm to use." \
		  "Accepted values in {grid,random,hyperband,bohb}, eg: python3 main.py -s grid"
	@echo 
	@echo "Alternatively use targets 'grid', 'random', 'hyperband' or 'bohb' with make command eg. make grid"
	@echo
	@echo "Use targets 'clean' and 'clean-models', to clear auto-generated python files"
	@echo "and remove checkpointed models respectively."
	@echo
	@echo "If neccessary, check target 'help-experiment' to see how to modify experiment to reduce run time."
	@echo "Running any seach in CPU may take upto several minutes, GPU recommended."
	@echo
	@echo "The hyperparameter search space of all algorithms are configured in config.yaml."
	@echo "To know more about default settings make targets 'help-grid', 'help-random', 'help-hyperband' and 'help-bohb'"
	@echo
	@echo "To set custom hyperparameter search space or custom experiment check Readme.md file"

grid: # Runs grid seach in search space defined in config.yaml, specifically 'grid' element
	python3 main.py -s grid

help-grid:
	@echo "No of configurations generated is 8. Hyperparameters tuned and their values are :"
	@echo "    learning_rate : {0.001, 0.01}                A discrete set composed form grid list [a1, a2, [a3, [...]]]"
	@echo "    dropout       : {0.3, 0.5}                   A discrete set composed form grid list [a1, a2, [a3, [...]]]"
	@echo "    embedding_dim : {50, 100}                    A discrete set composed form grid list [a1, a2, [a3, [...]]]"
	@echo
	@echo "To define custom hyperparamter search space, modify the 'grid' key in file config.yaml. More" \
		  " information in Readme.md"
	@echo
	@echo "	~Time(gpu)       : ~ 6 min in single GPU(MX 150) for 8 Trials, ~45sec per Trial for 2 epochs"
	@echo "Running grid seach in CPU may take upto several minutes, GPU recommended"

random: # Runs grid seach in search space defined in config.yaml, specifically 'distribution' and 'args'
	python3 main.py -s random

help-random:
	@echo "No of configurations sampled by default is 5. Hyperparameters tuned and their distribution are :"
	@echo
	@echo "    learning_rate : loguniform(0.001, 0.01),      uniformly sample a float in range args [lowerbound, upperbound] "
	@echo "                                                  while sampling in log space."
	@echo "    batch_size    : choice([128, 256]),           unifromly sample from args list [a1, a2, [a3, [...]]] " 
	@echo "    num_layers    : choice([2, 3]),               unifromly sample from args list [a1, a2, [a3, [...]]] "
	@echo "    embedding_dim : choice([50, 100, 200, 300]),  unifromly sample from args list [a1, a2, [a3, [...]]] "
	@echo
	@echo "To define custom hyperparamter search space, modify the 'distribution' and 'args' key in file config.yaml."
	@echo "More information in Readme.md"
	@echo
	@echo "	~Time(gpu)       : ~ 7 min in single GPU(MX 150) for 5 Trials, ~ 80sec per Trial for 2 epochs"
	@echo "Running grid seach in CPU may take upto several minutes, GPU recommended"
	

hyperband: # Runs hyperband seach in search space defined in config.yaml, specifically 'distribution' and 'args'
	python3 main.py -s hyperband

help-hyperband:
	@echo "No of configurations sampled by default is 5. Hyperparameters tuned and their distribution are :"
	@echo
	@echo "    learning_rate : loguniform(0.001, 0.01),      uniformly sample a float in range args [lowerbound, upperbound] "
	@echo "                                                  while sampling in log space."
	@echo "    batch_size    : choice([128, 256]),           unifromly sample from args list [a1, a2, [a3, [a4 ...]]] " 
	@echo "    num_layers    : choice([2, 3]),               unifromly sample from args list [a1, a2, [a3, [a4 ...]]] "
	@echo "    embedding_dim : choice([50, 100, 200, 300])   unifromly sample from args list [a1, a2, [a3, [a4 ...]]] "
	@echo
	@echo "Other default settings are :"
	@echo	  "    reduction_rate: 3 , downsamping rate"
	@echo	  "    grace_period" : 1 , min no. of epochs to run a configuration before early stopping.
	@echo
	@echo "To define custom hyperparamter search space, modify the 'distribution' and 'args' key in file config.yaml."
	@echo "More information in Readme.md"
	@echo
	@echo "	~Time(gpu)       : ~ 7 min in single GPU(MX 150) for 5 Trials, ~ 80sec per Trial for 2 epochs"
	@echo "Running hyperband seach in CPU may take upto several minutes, GPU recommended."

bohb:
	python3 main.py -s bohb

help-bohb:
	@echo "No of configurations sampled by default is 5. Hyperparameters tuned and their distribution are :"
	@echo
	@echo "    learning_rate : loguniform(0.001, 0.01),      uniformly sample a float in range args [lowerbound, upperbound] "
	@echo "                                                  while sampling in log space."
	@echo "    batch_size    : choice([128, 256]),           unifromly sample from args list [a1, a2, [a3, [a4 ...]]] " 
	@echo "    num_layers    : choice([2, 3]),               unifromly sample from args list [a1, a2, [a3, [a4 ...]]] "
	@echo "    embedding_dim : choice([50, 100, 200, 300])   unifromly sample from args list [a1, a2, [a3, [a4 ...]]] "
	@echo
	@echo "Other default settings are :"
	@echo	  "    reduction_rate: 3 , downsamping rate"
	@echo
	@echo "To define custom hyperparamter search space, modify the 'distribution' and 'args' key in file config.yaml."
	@echo "More information in Readme.md"
	@echo
	@echo "	~Time(gpu)       : ~ 8 min in single GPU(MX 150) for 5 Trials, ~ 90sec per Trial for 2 epochs"
	@echo "Running hyperband seach in CPU may take upto several minutes, GPU recommended."

help-experiment: 
	@echo "To speed up the experiment, following adjustments are available"
	@echo "    1. Decrease no. of sampled configuration, lower 'num_samples' under 'EXPERIMENT' section in config.yaml"
	@echo "    2. Decrease no. of epochs per configuration, lower 'max_epoch' under 'EXPERIMENT' section in config.yaml"
	@echo
	@echo "Hyperparameters with significant effect on run time are 'num_layers', 'embedding_dim' and 'hidden_dim'"

file-usage:
	@echo "File : preprocessing.py"
	@echo "    Files used    : datasets/raw/AG news.tar.gz (11.8MB), config.yaml(1.1KB)"
	@echo "    Files created : datasets/processed/* (30.0MB)" 
	@echo
	@echo "File : train.py"
	@echo "    Files used    : datasets/processed/* (30.0MB), preprocessing.py(12KB), modelling.py(2.2KB), config.yaml(1.1KB), embeddings/GloVe ~(4GB, in mounted volume)"
	@echo "    Files created : models/* size grows based on checkpoints created"
	@echo
	@echo "File : main.py"
	@echo "    Files used    : config.yaml(1.1KB), train.py(10KB)"
	@echo "    Files created : None"
	@echo
	@echo "File : modelling.py"
	@echo "    Files used    : None"
	@echo "    Files created : None"

clean:	# Remove auto-generated python files (pycache and .pyc files).
	sudo rm -rf __pycache__

clean-models: # Remove models checkpoints in models/.
	sudo rm -rf models/*


	