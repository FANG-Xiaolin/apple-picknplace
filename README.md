# Beep – Basic Automated Pick and Place

A minimal Python package for automated picking and placing of objects. 

## Installation

TODO

## Sanity Check

Run in simulation with the provided pkl file for debugging / visualization.

```
python main.py --run_in_simulation --vis
```

## Setup

Copy the config file template and make any necessary hardware-specific changes.

```bash
cp config_template.yml config.yml
```

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
