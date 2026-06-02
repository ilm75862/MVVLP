import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CNNLSTMModel(nn.Module):
    def __init__(self, img_channels, img_height, img_width, instr_size, num_actions, dropout = 0.0):
        super(CNNLSTMModel, self).__init__()
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        cnn_output_size = self._get_cnn_output_size(img_channels, img_height, img_width)

        # LSTM for instruction processing
        self.lstm = nn.LSTM(input_size=instr_size, hidden_size=128, num_layers=3, batch_first=True)

        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size + 128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        # self.softmax = torch.nn.LogSoftmax()
        self.softmax = torch.nn.LogSoftmax()

    def _get_cnn_output_size(self, img_channels, img_height, img_width):
        x = torch.randn(1, img_channels, img_height, img_width)
        x = self.cnn(x)
        return x.view(1, -1).size(1)

    def forward(self, image, instruction):
        # Process image with CNN
        img_feat = self.cnn(image)

        # Process instruction with LSTM
        instruction = instruction.unsqueeze(1)  # Adding batch dimension for LSTM
        _, (instr_feat, _) = self.lstm(instruction)
        instr_feat = instr_feat[-1]  # Get the output of the last LSTM layer

        # Concatenate features and pass through fully connected layers
        combined_feat = torch.cat((img_feat, instr_feat), dim=1)
        out = self.fc(combined_feat)
        out = self.softmax(out)

        return out

class CNNMLPModel(nn.Module):
    def __init__(self, img_channels, img_height, img_width, instr_size, num_actions, dropout = 0.0):
        super(CNNMLPModel, self).__init__()
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        cnn_output_size = self._get_cnn_output_size(img_channels, img_height, img_width)

        # MLP for instruction processing
        self.instr_fc = nn.Sequential(
            nn.Linear(instr_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size + 128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.softmax = torch.nn.LogSoftmax()

    def _get_cnn_output_size(self, img_channels, img_height, img_width):
        x = torch.randn(1, img_channels, img_height, img_width)
        x = self.cnn(x)
        return x.view(1, -1).size(1)

    def forward(self, image, instruction):
        # Process image with CNN
        img_feat = self.cnn(image)

        # Process instruction with MLP
        instr_feat = self.instr_fc(instruction)

        # Concatenate features and pass through fully connected layers
        combined_feat = torch.cat((img_feat, instr_feat), dim=1)
        out = self.fc(combined_feat)
        out = self.softmax(out)

        return out

class CNNTransformerModel(nn.Module):
    def __init__(self, img_channels, img_height, img_width, instr_size, num_actions, nhead=8, num_layers=3, dropout = 0.0):
        super(CNNTransformerModel, self).__init__()
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Flatten()
        )
        cnn_output_size = self._get_cnn_output_size(img_channels, img_height, img_width)

        # Transformer for instruction processing
        # self.instr_embedding = nn.Linear(instr_size, 128)
        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size + 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.softmax = torch.nn.LogSoftmax()

    def _get_cnn_output_size(self, img_channels, img_height, img_width):
        x = torch.randn(1, img_channels, img_height, img_width)
        x = self.cnn(x)
        return x.view(1, -1).size(1)

    def forward(self, image, instruction):
        # Process image with CNN
        img_feat = self.cnn(image)
        # if torch.isnan(img_feat).any():
        #     raise "cnn !!!!"
        # Process instruction with Transformer
        # instr_feat = self.instr_embedding(instruction).unsqueeze(0)  # Add batch dimension
        instr_feat = instruction.unsqueeze(0)  # Add batch dimension
        # if torch.isnan(instr_feat).any():
        #     raise "instr_embedding !!!!"
        instr_feat = self.transformer_encoder(instr_feat).squeeze(0)  # Remove batch dimension
        # if torch.isnan(instr_feat).any():
        #     raise "transformer_encoder !!!!"
        # Concatenate features and pass through fully connected layers
        combined_feat = torch.cat((img_feat, instr_feat), dim=1)
        out = self.fc(combined_feat)
        # if torch.isnan(out).any():
        #     raise "fc !!!!"
        out = self.softmax(out)

        return out

class ResNetGRUModel(nn.Module):
    def __init__(self, img_channels, img_height, img_width, instr_size, num_actions, dropout = 0.0):
        super(ResNetGRUModel, self).__init__()
        # ResNet for image processing
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer

        # GRU for instruction processing
        self.gru = nn.GRU(input_size=instr_size, hidden_size=128, num_layers=3, batch_first=True)

        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 128),  # 512 is the output size of ResNet18
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, image, instruction):
        # Process image with ResNet
        img_feat = self.resnet(image)

        # Process instruction with GRU
        instruction = instruction.unsqueeze(1)  # Adding batch dimension for GRU
        _, instr_feat = self.gru(instruction)
        instr_feat = instr_feat[-1]  # Get the output of the last GRU layer

        # Concatenate features and pass through fully connected layers
        combined_feat = torch.cat((img_feat, instr_feat), dim=1)
        out = self.fc(combined_feat)
        out = self.softmax(out)
        return out

class ResNetLSTMModel(nn.Module):
    def __init__(self, img_channels, img_height, img_width, instr_size, num_actions, dropout = 0.0):
        super(ResNetLSTMModel, self).__init__()
        # ResNet for image processing
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer

        # lstm for instruction processing
        self.lstm = nn.LSTM(input_size=instr_size, hidden_size=128, num_layers=3, batch_first=True)

        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 128),  # 512 is the output size of ResNet18
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, image, instruction):
        # Process image with ResNet
        img_feat = self.resnet(image)

        # Process instruction with GRU
        instruction = instruction.unsqueeze(1)  # Adding batch dimension for GRU
        _, (instr_feat, _) = self.lstm(instruction)
        instr_feat = instr_feat[-1]  # Get the output of the last GRU layer

        # Concatenate features and pass through fully connected layers
        combined_feat = torch.cat((img_feat, instr_feat), dim=1)
        out = self.fc(combined_feat)
        out = self.softmax(out)
        return out
class ResNetMLPModel(nn.Module):
    def __init__(self, img_channels, img_height, img_width, instr_size, num_actions, dropout = 0.0):
        super(ResNetMLPModel, self).__init__()
        # ResNet for image processing
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer

        # MLP for instruction processing
        self.instr_fc = nn.Sequential(
            nn.Linear(instr_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 128),  # 512 is the output size of ResNet18
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, image, instruction):
        # Process image with ResNet
        img_feat = self.resnet(image)

        # Process instruction with GRU
        # instruction = instruction.unsqueeze(1)  # Adding batch dimension for GRU
        instr_feat = self.instr_fc(instruction)
        # instr_feat = instr_feat[-1]  # Get the output of the last GRU layer

        # Concatenate features and pass through fully connected layers
        combined_feat = torch.cat((img_feat, instr_feat), dim=1)
        out = self.fc(combined_feat)
        out = self.softmax(out)
        return out


class CNNGRUModel(nn.Module):
    def __init__(self, img_channels, img_height, img_width, instr_size, num_actions, dropout = 0.0):
        super(CNNGRUModel, self).__init__()
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        cnn_output_size = self._get_cnn_output_size(img_channels, img_height, img_width)

        # gru for instruction processing
        self.gru = nn.GRU(input_size=instr_size, hidden_size=128, num_layers=3, batch_first=True)

        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size + 128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        # self.softmax = torch.nn.LogSoftmax()
        self.softmax = torch.nn.LogSoftmax()

    def _get_cnn_output_size(self, img_channels, img_height, img_width):
        x = torch.randn(1, img_channels, img_height, img_width)
        x = self.cnn(x)
        return x.view(1, -1).size(1)

    def forward(self, image, instruction):
        # Process image with CNN
        img_feat = self.cnn(image)

        # Process instruction with LSTM
        instruction = instruction.unsqueeze(1)  # Adding batch dimension for LSTM
        _, instr_feat = self.gru(instruction)
        instr_feat = instr_feat[-1]  # Get the output of the last LSTM layer

        # Concatenate features and pass through fully connected layers
        combined_feat = torch.cat((img_feat, instr_feat), dim=1)
        out = self.fc(combined_feat)
        out = self.softmax(out)

        return out

class ResNetTransModel(nn.Module):
    def __init__(self, img_channels, img_height, img_width, instr_size, num_actions, nhead=8, num_layers=3, dropout = 0.0):
        super(ResNetTransModel, self).__init__()
        # ResNet for image processing
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer

        encoder_layers = TransformerEncoderLayer(d_model=128, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)


        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(512 + instr_size, 128),  # 512 is the output size of ResNet18
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, image, instruction):
        # Process image with ResNet
        img_feat = self.resnet(image)

        # Process instruction with GRU
        # instruction = instruction.unsqueeze(1)  # Adding batch dimension for GRU
        # instr_feat = instr_feat[-1]  # Get the output of the last GRU layer
        instr_feat = instruction.unsqueeze(0)  # Add batch dimension
        # if torch.isnan(instr_feat).any():
        #     raise "instr_embedding !!!!"
        instr_feat = self.transformer_encoder(instr_feat).squeeze(0)  # Remove batch dimension
        # if torch.isnan(instr_feat).any():
        # Concatenate features and pass through fully connected layers
        combined_feat = torch.cat((img_feat, instr_feat), dim=1)
        out = self.fc(combined_feat)
        out = self.softmax(out)
        return out



if __name__ == '__main__':

    # 假设图像尺寸是 64x64x3，指令的大小是 instr_size，动作的数量是 num_actions
    img_channels = 12
    img_height = 270
    img_width = 480
    instr_size = 512
    num_actions = 3

    model = ResNetTransModel(img_channels, img_height, img_width, instr_size, num_actions)
