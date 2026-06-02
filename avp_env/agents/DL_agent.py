import torch
import numpy as np
from DL_model import ResNetTransModel, CNNLSTMModel, CNNMLPModel, CNNTransformerModel, ResNetGRUModel, \
    ResNetMLPModel, \
    CNNGRUModel, ResNetLSTMModel


class DLAgent:
    def __init__(self):
        self.model_path = f"../model/CNN_MLP.pth"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化你的 DL 模型
        self.model = CNNMLPModel(12, 270, 480, 128, 3, 0.4).to(self.device)

        # 读取模型参数
        ckpt = torch.load(self.model_path, map_location=self.device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]

        ckpt = {
            k.replace("module.", ""): v
            for k, v in ckpt.items()
        }

        self.model.load_state_dict(ckpt)
        self.model.eval()

    def get_action(self, image, prompt):

        device = next(self.model.parameters()).device

        # image: numpy.ndarray, shape [H, W, C]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        else:
            image = image.float()

        # [H, W, C] -> [C, H, W]
        if image.ndim == 3:
            image = image.permute(2, 0, 1)

        # [C, H, W] -> [1, C, H, W]
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(device)

        # prompt / instruction 也要转成 Tensor
        if isinstance(prompt, np.ndarray):
            prompt = torch.from_numpy(prompt).float()
        elif not torch.is_tensor(prompt):
            prompt = torch.tensor(prompt, dtype=torch.float32)
        else:
            prompt = prompt.float()

        # [instruction_dim] -> [1, instruction_dim]
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0)

        prompt = prompt.to(device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image, prompt)

        # 如果 output 是分类 logits
        action = torch.argmax(output, dim=1).item()

        return action
