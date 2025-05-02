# Beep – Basic Automated Pick and Place

A minimal Python package for automated picking and placing of objects. 

## Installation

TODO

## Setup

Copy the config file template and make any necessary changes.

```bash
cp config_template.yml config.yml
```

Set `run_in_simulation` to `True` in `config.yml` to run in simulation, as a sanity check. 

## Usage

On machine controlling the robot / NUC:
```
python beepp/server.py
```

On machine with GPUs:
```
python main.py
```

## Note
There is no task planner or any other high-level functionality.
The package is designed to be a simple tool with fixed sequences of actions. 
