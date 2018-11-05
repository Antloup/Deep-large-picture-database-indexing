import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

import torch.nn as nn

import torch.optim as optim

from Net import Net
from torch.utils.data import SubsetRandomSampler

plt.interactive(False)

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def training(train_loader, net, optimizer, criterion):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # imshow(torchvision.utils.make_grid(inputs))
        # input("Press enter")
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        labels = torch.eye(2)[labels.long()]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0


def validation(validation_loader, net):
    correct = 0
    total = 0
    acc = 0.0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            _, predicted = torch.max(net(images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = (correct / total)
        print('Accuracy of the network on the test : %d %%' % (100 * correct / total))
    return acc


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    valid_threshold = .9
    valid_split = .2
    num_folds = 2

    trainset = torchvision.datasets.ImageFolder(root="../train_images", transform=transform)
    dataset_size = len(trainset)
    fold_size = int(dataset_size / num_folds)
    fold_split_size = int(np.floor(valid_split * fold_size))
    indices = list(range(dataset_size))

    net = []
    train_loader = []
    validation_loader = []
    shuffle(indices)
    for i in range(0, num_folds):
        split_start = dataset_size - (fold_split_size * (i + 1))
        split_end = split_start + fold_split_size
        train_indices = indices[0:split_start - 1] + indices[split_end + 1:dataset_size]
        val_indices = indices[split_start:split_end]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader.append(torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                        sampler=train_sampler))
        validation_loader.append(torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                             sampler=valid_sampler))
        net.append(Net())
        net[i].to(device)

    classes = ('Non-visage', 'Visage')

    criterion = nn.CrossEntropyLoss()

    criterion = torch.nn.MSELoss()

    optimizer = []
    for i in range(0, num_folds):
        optimizer.append(optim.SGD(net[i].parameters(), lr=0.001, momentum=0.9))

    for i in range(0, num_folds):
        for epoch in range(2):
            acc = 0.0
            while acc < valid_threshold:
                training(train_loader[i], net[i], optimizer[i], criterion)
                acc = validation(validation_loader[i], net[i])

    print('Finished Training')

    torch.save(net[0].state_dict(), "../model1.pt")
    torch.save(net[1].state_dict(), "../model2.pt")

    dataiter = iter(train_loader[0])
    features, _ = dataiter.next()
    imshow(torchvision.utils.make_grid(features))
    _, predicted = torch.max(net[0](features), 1)
    print('Predicted network 1: ', ' '.join('%5s' % classes[predicted[j]]
                                            for j in range(4)))
    _, predicted = torch.max(net[1](features), 1)
    print('Predicted network 2: ', ' '.join('%5s' % classes[predicted[j]]
                                            for j in range(4)))
