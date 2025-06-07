# Mining-Gym
Mining-Gym is a configurable benchmarking environment for optimizing truck dispatch scheduling in open-pit mining using Reinforcement Learning.

## About

**Why was this written?**  
Open-pit mines operate massive truck fleets to move material between shovels, crushers and dumps. Manually dispatching these trucks, or relying on rigid rules, wastes fuel and time—especially when equipment breaks down or queues build up. Reinforcement Learning (RL) could learn smarter schedules, but researchers lacked a standard, realistic simulator for fair comparisons. Mining-Gym fills that gap.

**What exactly is Mining-Gym?**  
Mining-Gym packages a discrete-event mine simulator behind the familiar OpenAI Gym API. You can swap between classical dispatch algorithms and RL agents while using the same configuration files. Everything runs locally with minimal setup, so experimentation is as simple as calling a training script.

**How does RL "see" the mine?**  
The environment converts the state of trucks, shovels, crushers and stockpiles into an observation vector. At each step the agent chooses where a truck should go next. Rewards reflect haulage efficiency and downtime, guiding the RL policy toward efficient dispatching.

**What did they test?**  
The initial experiments compared rule-based schedules against RL agents across different mine layouts. Metrics such as throughput and queue time were used to evaluate how quickly RL policies adapt when trucks break down or resources become congested.

Bottom line: Mining-Gym turns a messy industrial problem into an RL playground you can run on a laptop. If you can train CartPole, you’re two commands away from optimising million-dollar mining fleets.


## Repository Files

- `mgym_GymRun.py`: Main script for training and playing RL-based scheduling agents.
- `mgym_DefSchdRun.py`: Script to run classical/rule-based schedulers.
- `mgym_GymEnv.py`: OpenAI Gym-compatible environment setting that wraps the DES Mining site simulator.
- `mgym_DesEnv.py`: Script for the DES-based Mining Site Simulator.
- `scheduler.py`: Contains definitions for rule-based scheduling algorithms.
- `config_extend.txt`: Configuration file where you can customize simulation settings.
- `requirements.txt`: Pip requirements file with the core dependencies.
- `environment.yml`: Optional conda environment file containing the same packages.
  
## Setup

### 1. Install dependencies

Use either `pip` or `conda` to install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate mgym
```

### 2. Launch the GUI

The recommended way to explore Mining-Gym is through the simple graphical
interface:

```bash
python mgym_gui.py
```

The GUI lets you train or play an RL model and run the classical schedulers with
a few clicks. Output appears in your terminal window.

### 3. Command-line scripts (optional)

For scripted runs you can call the underlying programs directly:

- `python mgym_GymRun.py train/play ...` – reinforcement-learning interface
- `python mgym_DefSchdRun.py ...` – classical schedulers

Use the `--help` flag on each script for available options. To render the
environment visually add `--render human`.

Both training and classical runs display progress bars via `tqdm`.



## Research

This repository implements concepts from the open-access paper ["Mining-Gym: A Configurable RL Benchmarking Environment for Truck Dispatch Scheduling"](https://doi.org/10.48550/arXiv.2503.19195). The environment and experiments closely follow the approach presented in that work.

## Prompt Engineering

This program was created completely through prompt engineering with ChatGPT Codex. The project evolved from a series of iterative prompts, including:

- Requesting a high-level design for a reinforcement learning environment to dispatch trucks in an open-pit mine.
- Generating the Python modules for the environment using the OpenAI Gym API.
- Adding rule-based scheduler implementations and configuration files.
- Writing training and inference scripts, along with a simple GUI for experimentation.

