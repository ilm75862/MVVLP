# Demand-driven Autonomous Parking

## Install
Dependencies:

- [python 3.8](https://www.python.org/downloads/release/python-3818/) <3
- [numpy](https://numpy.org/install/) <3
- `pip install pandas` for data manipulation <3
- `pip install stable-baselines3` for reinforcement learning <3
- `pip install gym` for OpenAI's gym environments <3
- `pip install transformers` for Huggingface's transformers <3
- `pip install opencv-python` for computer vision tasks <3
- `pip install ray` for parallel and distributed computing <3
- `pip install dm_tree, typer, scipy` <3
- `pip install h5py` for handling HDF5 files <3


[//]: # (- `pip install fastapi` for building APIs <3)

[//]: # (- `pip install ray` for parallel and distributed computing <3)

[//]: # (- `pip install requests` for making HTTP requests <3)

[//]: # (- `pip install gradio` for interactive web UIs <3)

[//]: # (- `pip install uvicorn` for ASGI server <3)

## Dataset
This project runs with a specific dataset to function properly. Please download the dataset from the following link and save it to a local folder:

[Download Dataset](https://doi.org/10.57760/sciencedb.12908)

Ensure that you have downloaded the entire dataset and place the data files in the correct folder as specified in the project instructions.

## Quick Start
Follow the steps below to get started quickly with the Reinforcement Learning Environment Build and Test using the provided Python scripts:
1. Ensure that you have downloaded and properly set up the dataset as described in [Dataset](#dataset).
2. Clone the repository to your local computer. 
3. Install any necessary dependencies.
### Deep Learning Agent Demo

Run the script to enter the reinforcement learning environment and use the Perfect Parking agent to obtain inputs from the deep learning agent with the corresponding perfect action labels.
```
$ python training_data_with_deep_learning.py
```
### Reinforcement Learning Agent Demo

Run the script to enter the reinforcement learning environment and quickly start training a simple DQN agent for autonomous parking task.
```
$ python RL_demo.py
```

### Test Demo
Run the script to test the agent's performance. Here, a random agent is used, but any agent can be substituted. The results will be saved as JSON and ZIP files.
```
$ python test_demo.py
```


## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file included with this repository.
