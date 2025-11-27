import torch
import torch.nn as nn

def models(model_name: str) -> nn.Module:
    """
    Returns a CNN model based on the model_name.

    Args:
        model_name (str): Name of the model to return. 
        Options: 
        {
            "model1": 1-Layer CNN,
            "model2": 1-Layer CNN,
            "Fast_test_v1": 1-Layer CNN,
        }

    Returns:
        nn.Module: An instance of the requested model.
    """
    available_models = [
        "model1", 
        "model2", 
        "Fast_test_v1"
    ]

    if model_name == "model1":
        return Model1()
    elif model_name == "model2":
        return Model2()
    elif model_name == "Fast_test_v1":
        return Fast_Test_v1()
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Available models: {available_models}")


class Model1(nn.Module):
    """Name: 'model1' - Dummy CNN with one conv layer"""
    def __init__(self, in_channels=3, out_classes=10):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Model2(nn.Module):
    """Name: 'model2' - Another dummy CNN with one conv layer"""
    def __init__(self, in_channels=3, out_classes=15):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Fast_Test_v1(nn.Module):
    """Name: 'Fast_test_v1' - Dummy CNN with one conv layer"""
    def __init__(self, in_channels=3, out_classes=15):
        super(Fast_Test_v1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, out_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x