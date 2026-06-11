from avp_env.metrics.metrics import get_parking_metrics
from avp_env.metrics.utils import instru_len
from avp_env.metrics.experiment import run_experiments, load_experiments, run_dl_experiments
from avp_env.envs.avp_env import MetricsVLLMEnv
from avp_env.agents.DL_agent import DLAgent
import argparse
import os
import gc
import torch
import json

log_file_path = "../results/DL/metrics/metrics_results.json"
results = []
model = "cnnmlp"
view = "multi"
instr_type = "raw"
instruction_path = f'../data/Command/{instr_type}_command.json'
output_file = f'../results/DL/parking/{model}_{view}_{instr_type}_result.json'

output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

env = MetricsVLLMEnv(env_type=instr_type)

agent = DLAgent()

instru_num = instru_len(instruction_path)

experiments = run_dl_experiments(env, agent, instru_num, output_file)

metrics = get_parking_metrics(experiments)
log_text = (
        "=" * 40 +
        f"\nMetrics for instr_type '{instr_type}', model '{model}', view '{view}':\n" +
        f"{metrics}\n" +
        "=" * 40
)
print(log_text)

record = {
    "instr_type": instr_type,
    "model": model,
    "view": view,
    "metrics": metrics
}
results.append(record)

del agent
del env
gc.collect()
torch.cuda.empty_cache()

with open(log_file_path, "w") as f:
    json.dump(results, f, indent=2)