# APPLE – Automated Pick and PLacE

A minimal Python package for automated picking and placing of objects. 
It uses [UncOS](https://github.com/FANG-Xiaolin/uncos) for object segmentation, 
GPT for object naming, 
and simple motion planners for manipulation.

## Setup

Copy the config file template and make any necessary changes.

```bash
cp config_template.py config.py
```

## Usage

```
python main.py
```

## Note
There is no task planner or any other high-level functionality.
The package is designed to be a simple tool with fixed sequences of actions. 
