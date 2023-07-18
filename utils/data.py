import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10


class Cifar10:
    def __init__(self, data_transforms=None):
        if (
            data_transforms is None
        ):  # if no data transforms are passed, don't add any new ones
            data_transforms = []
        num_channels = 3  # 3 channels for RGB images
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,) * num_channels, (0.5,) * num_channels
                ),  # always normalize images (mean and standard deviation)
                # other data transforms to apply? (e.g. data augmentation)
                *data_transforms,
            ]
        )
        self.trainset = CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        self.testset = CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

def eval_model(model, data_loader, name):
    model.eval()  # set model to evaluation mode
    test_loss = 0  # initialize test loss
    correct = 0  # initialize number of correct predictions
    with torch.no_grad():  # don't calculate gradients
        for data, target in data_loader:  # iterate over data
            if torch.cuda.is_available():  # if GPU is available, move data to GPU
                data, target = data.cuda(), target.cuda()
            output = model(data)  # forward pass
            logits = output["logits"]
            test_loss += output["loss"].item()  # sum up batch loss
            pred = logits.data.max(1, keepdim=True)[
                1
            ]  # calculate predictions (indices of maximum log-probability)
            correct += (
                pred.eq(target.data.view_as(pred)).cpu().sum()
            )  # calculate number of correct predictions

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)  # calculate accuracy (%)
    print(
        "{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            name, test_loss, correct, len(data_loader.dataset), accuracy
        )
    )
    return test_loss, correct / len(data_loader.dataset)


def log_every_n_percent(epoch, batch_idx, data_loader, percent):
    pct_complete = 100.0 * (batch_idx + 1) / len(data_loader)
    if pct_complete % percent == 0:
        complete = (batch_idx + 1) * len(data_loader.batch_sampler)
        total = len(data_loader.dataset)
        print(f'Epoch {epoch}: {complete}/{total} ({pct_complete:.0f}%)')


def log_every_n_steps(epoch, batch_idx, data_loader, steps):
    if (batch_idx + 1) % steps == 0:
        complete = (batch_idx + 1) * len(data_loader.batch_sampler)
        total = len(data_loader.dataset)
        print(f'Epoch {epoch}: {complete}/{total} ({steps} steps)')
