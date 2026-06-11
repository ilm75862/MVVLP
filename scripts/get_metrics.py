import json
from avp_env.metrics.metrics import get_parking_metrics

method = "MLLM"
model = "Janus-Pro-7B"
view = "multi"
instr_type = "human"

# parser.add_argument('--instr_types', nargs='+', type=str, default=['raw','synonyms','long','short','abstract','test'], help='List of instruction types')

output_file = f'../results/{method}/parking/{model}_{view}_{instr_type}_result.json'

with open(output_file, "r") as json_file:
    experiments = json.load(json_file)

metrics = get_parking_metrics(experiments)
log_text = (
        "=" * 40 +
        f"\nMetrics for instr_type '{instr_type}', model '{model}', view '{view}':\n" +
        f"{metrics}\n" +
        "=" * 40
)
print(log_text)
