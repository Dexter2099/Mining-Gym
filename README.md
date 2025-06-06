# Mining-Gym
Mining-Gym is an open-source, configurable benchmarking environment for optimizing truck dispatch scheduling in open-pit mining using Reinforcement Learning.


## Repository Files

- `mgym_GymRun.py`: Main script for training and playing RL-based scheduling agents.
- `mgym_DefSchdRun.py`: Script to run classical/rule-based schedulers.
- `mgym_GymEnv.py`: OpenAI Gym-compatible environment setting that wraps the DES Mining site simulator.
- `mgym_DesEnv.py`: Script for the DES-based Mining Site Simulator.
- `scheduler.py`: Contains definitions for rule-based scheduling algorithms.
- `config_extend.txt`: Configuration file where you can customize simulation settings.
- `environment.yml`: Conda environment setup file to ensure reproducibility.
  
## Setup Instructions

1. **Create and activate the virtual environment:**  
conda env create -f environment.yml  

2. **To train a new RL policy network, run:**  
python mgym_GymRun.py train --num_episodes 10  
--num_episodes: Sets the number of training episodes.  
The RL algorithm is selected internally in the mgym_GymRun.py code.

3. **To run a classical scheduler (e.g., random scheduling), use:**
python mgym_DefSchdRun.py --num_episodes 10 --algo_choice 1
--num_episodes: Number of episodes to simulate.
--algo_choice: Selects the scheduler algorithm as defined in scheduler.py.
Example: 1 stands for random scheduling.

4. **To play using a pretrained model, run:**
python mgym_GymRun.py play --num_episodes 5 --model_path <path_to_saved_model.zip>
Replace <path_to_saved_model.zip> with the actual path to your saved model.

5. **Use a custom configuration file (optional):**
Both `mGym_GymRun.py` and `mGym_DefSchdRun.py` accept a `--config` argument to load parameters from an alternative file.
Example: `python mGym_GymRun.py train --num_episodes 10 --config my_config.txt`
Example: `python mGym_DefSchdRun.py --num_episodes 5 --algo_choice 1 --config my_config.txt`

6. **To change configuration data:**
You can modify the simulation settings by editing the config.extend.txt file or another file provided via `--config`. This allows you to adjust parameters such as environment details, scheduler settings, and other simulation-related options.

7. **Progress Bars**
Both training (`mGym_GymRun.py`) and classical scheduler runs (`mGym_DefSchdRun.py`) display an episode progress bar powered by `tqdm`. The package is already listed in `environment.yml`.


