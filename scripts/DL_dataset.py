import os
import numpy as np
import h5py

from avp_env.envs.avp_env import MetricsVLLMEnv
from avp_env.agents.rule import RulebasedAgent
from transformers import AutoTokenizer
from avp_env.metrics.utils import get_target_features


def collect_data_for_supervised_learning(env, agent):

    images = []
    instructions = []
    actions = []

    num_instructions = len(env.metrics_instructions)

    for ep in range(50):

        obs = env.reset(InsIndex=ep)
        ini_image, ini_instruction = obs
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


        instruction_tokens = tokenizer.encode(
            ini_instruction, add_special_tokens=True,
            max_length=128, padding='max_length',
            truncation=True
        )

        target_instruction = env.getTargetInstruction()
        perfect_trajectory = env.get_perfect_trajectory(target_instruction)
        target_id, target_features = get_target_features(env)

        for t in range(len(perfect_trajectory)):
            image, instruction = obs

            action = 0
            next_obs, reward, done, info = env.step(action)
            obs = next_obs

            for feature in target_features:
                if t == feature["path_id"]:
                    action = feature["loc_id"]
                    break
            action = int(action)

            images.append(image)
            instructions.append(instruction_tokens)
            actions.append(action)

            if done:
                break

    images = np.stack(images).astype(np.uint8)
    actions = np.asarray(actions, dtype=np.int64)

    return images, instructions, actions


instr_type = "raw"

env = MetricsVLLMEnv(env_type=instr_type)

agent = RulebasedAgent(
    isOptimal=True,
    isRandom=False
)

images, instructions, actions = collect_data_for_supervised_learning(
    env,
    agent,
)

os.makedirs("../data", exist_ok=True)

save_path = f"../data/Supervised_{instr_type}_dataset.h5"

str_dtype = h5py.string_dtype(encoding="utf-8")

with h5py.File(save_path, "w") as f:
    f.create_dataset(
        "images",
        data=images,
        compression="gzip",
        chunks=True
    )
    f.create_dataset(
        "instructions",
        data=np.array(instructions)
    )
    f.create_dataset(
        "actions",
        data=actions
    )

print(f"Saved dataset to {save_path}")
print("images:", images.shape)
print("instructions:", len(instructions))
print("actions:", actions.shape)
