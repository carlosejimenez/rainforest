from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes=10, criterion=nn.CrossEntropyLoss()):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # define loss function
        self.criterion = criterion

    def forward(self, x, target=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if target is not None:
            loss = self.criterion(x, target) if target is not None else None
        else:
            loss = None

        return {"loss": loss, "logits": x}
