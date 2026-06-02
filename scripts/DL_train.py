import torch.optim as optim
import torch
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from DL_model import ResNetTransModel, CNNLSTMModel, CNNMLPModel, CNNTransformerModel, ResNetGRUModel, ResNetMLPModel, \
    CNNGRUModel, ResNetLSTMModel
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os


def load_data(h5_file, train_ratio=0.8, batch_size=1024):
    with h5py.File(h5_file, 'r') as f:

        images = np.array(f['images']).astype(np.float32)
        instructions = np.array(f['instructions']).astype(np.float32)
        actions = np.array(f['actions']).astype(np.int64)

    values, counts = np.unique(actions, return_counts=True)

    for v, c in zip(values, counts):
        print(f"action={v}, count={c}")

    dataset = TensorDataset(torch.tensor(images).permute(0, 3, 1, 2), torch.tensor(instructions), torch.tensor(actions))

    # 计算拆分的大小
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # 使用 random_split 将数据集拆分为训练集和测试集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    testing_loss = 0.0

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for images, instructions, actions in dataloader:
            images, instructions, actions = images.to(device), instructions.to(device), actions.to(device)
            outputs = model(images, instructions)
            loss = criterion(outputs, actions)
            testing_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += actions.size(0)
            correct += (predicted == actions).sum().item()
    loss = testing_loss / len(dataloader.dataset)
    accuracy = correct / total
    return accuracy, loss


def train(model_class, model_name, train_loader, test_loader, img_channels, img_height, img_width,
          instr_size, num_actions, num_epochs, dropout=0.0):
    log_dir = f'../runs/{model_name}'
    model_path = f'../model/{model_name}.pth'
    # 确保保存模型的目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model = model_class(img_channels, img_height, img_width, instr_size, num_actions, dropout=dropout)

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir)

    # 训练模型

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, instructions, actions in train_loader:
            images, instructions, actions = images.to(device), instructions.to(device), actions.to(device)

            outputs = model(images, instructions)
            loss = criterion(outputs, actions)
            # print(loss.item())
            running_loss += loss.cpu().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print(loss)
        epoch_loss = running_loss / len(train_loader.dataset)

        # 每10个epoch记录一次训练损失
        if (epoch + 1) % 5 == 0:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            print(f'model: {model_name}, Epoch: {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

            # # 记录学习率
            # for param_group in optimizer.param_groups:
            #     writer.add_scalar('Learning Rate', param_group['lr'], epoch)

            # # 记录梯度范数
            # total_norm = 0
            # for p in model.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)
            # writer.add_scalar('Gradient Norm', total_norm, epoch)
            #
            # # 记录模型参数的直方图
            # for name, param in model.named_parameters():
            #     writer.add_histogram(name, param, epoch)
            #     if param.grad is not None:
            #         writer.add_histogram(name + '/grad', param.grad, epoch)

        # 每100个epoch记录一次评估准确率
        if (epoch + 1) % 50 == 0:
            test_accuracy, test_loss = evaluate(model, test_loader, criterion)
            writer.add_scalar('Test Accuracy', test_accuracy, epoch)
            print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {test_accuracy:.4f}')
            writer.add_scalar('Test Loss', test_loss, epoch)

    # 评估模型在测试集上的表现
    accuracy, _ = evaluate(model, test_loader, criterion)
    print(f'Test Accuracy: {accuracy:.4f}')
    # 保存模型参数
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    # 关闭 TensorBoard 记录器
    writer.close()


if __name__ == "__main__":
    img_channels = 12
    img_height = 270
    img_width = 480
    instr_size = 128
    # instr_size = 512
    num_actions = 3
    dropout = 0.4
    num_epochs = 1000
    data_path = '../data/Supervised_raw_dataset.h5'
    seed = 42
    # used_model = ['CNN_LSTM', 'CNN_MLP', 'ResNetGRU', 'CNN_Transformer']
    # used_model = ['CNN_Transformer']
    used_model = ['CNN_MLP']

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 如果使用 CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保在有确定性算法的情况下获得相同结果
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 获取数据集
    # 加载数据并创建数据加载器
    train_loader, test_loader = load_data(data_path)
    print("finished loading data")
    # 用于存储模型的字典
    models = {
        'CNN_LSTM': CNNLSTMModel,
        'CNN_MLP': CNNMLPModel,
        'CNN_Transformer': CNNTransformerModel,
        'ResNetGRU': ResNetGRUModel,
        'ResNetMLP': ResNetMLPModel,
        'CNN_GRU': CNNGRUModel,
        'ResNetTrans': ResNetTransModel,
        'ResNetLSTM': ResNetLSTMModel

    }

    for model_name in used_model:
        model_class = models[model_name]
        train(model_class, model_name, train_loader, test_loader, img_channels, img_height, img_width,
              instr_size, num_actions, num_epochs, dropout)
