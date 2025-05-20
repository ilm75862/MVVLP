# Multi-View Benchmark Dataset for Vision-and-Language Navigation

## Install

> [!NOTE] 
> Linux with Python 3.8
---
First, clone our MVVLP GitHub repository:

```shell
git clone https://github.com/ilm75862/MVVLP.git
```
---
Then, navigate to the `MVVLP` directory and install the dependencies listed in `requirements.txt`. It's recommended to create a new virtual environment using a package manager like `conda` or `uv`, and then install the dependencies:
```shell
cd MVVLP
pip install -r requirements.txt
```
---
## Dataset
This project runs with a specific dataset to function properly. Please download the dataset from the following link and save it to the local folder `MVVLP/data/`:

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/TJIET/MVVLP)

Ensure that you have downloaded the entire dataset and place the data files in the correct folder as specified in the project instructions.

---
## Quick Evaluation in Vision-Language Parking Simulator
Built on MVVLP dataset, the Vision-Language Parking simulator replicates realistic indoor parking environments and allows for testing various AI agents' understanding and reasoning capabilities under diverse visual and linguistic conditions.

---
### MLLM Agents
The script `LLM_test.py` enables systematic evaluation of different VLM agents across multiple instruction types and camera views, providing quantitative metrics to assess performance in simulated real-world tasks.

```shell
python ./scripts/LLM_test.py [--load] [--instr_types INSTR_TYPES ...] [--models MODELS ...] [--views VIEWS ...]
```
---
#### ⚙️ Arguments

- `--instr_types`  
  Specify one or more instruction formats. Default:  
  `['raw', 'synonyms', 'long', 'short', 'abstract','test']`

- `--models`  
  List of vision-language models to evaluate. Default:  
  `['deepseek-vl-7b-chat', 'Qwen2.5-VL-7B-Instruct', 'Janus-Pro-7B']`

- `--views`  
  Camera views available from the vehicle perspective. Default:  
  `['front', 'left', 'right', 'combined', 'multi', 'side']`

---
#### 📁 Output

For each combination of model, instruction type, and view:

- Runs an experiment.
- Saves the parking results to:  
  `../results/MLLM/parking/<model>_<view>_<instr_type>_result.json`
- Saves the MLLM metrics to:  
  `../results/MLLM/metrics/metrics_results.json`
---
### Reinforcement Learning Agent

#### Train
The script `RL_train.py` facilitates the training of reinforcement learning agents within simulated environments, enabling policy optimization through trial-and-error interactions. It supports various configurations and training algorithms, providing a framework to benchmark agent learning progress and performance over time.
```
$ python ./scripts/RL_train.py
```
---
#### Test
Then you can test the RL agents and get the metrics with `RL_test.py`:
```
$ python ./scripts/RL_test.py
```
- Saves the parking results to:  
  `../results/RL/parking/<agent_name>_result.json`
- Saves the MLLM metrics to:  
  `../results/RL/metrics/metrics_result.json`
---

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file included with this repository.
