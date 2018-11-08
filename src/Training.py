from random import shuffle

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

import torch.optim as optim

from src.Net import Net
from torch.utils.data import SubsetRandomSampler

plt.interactive(False)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    batch_size = 64
    valid_split = .2
    num_folds = 2
    num_epoch = 15

    trainset = torchvision.datasets.ImageFolder(root="../train_images", transform=Net.transform)

    dataset_size = len(trainset)
    fold_size = int(dataset_size / num_folds)
    fold_split_size = int(np.floor(valid_split * fold_size))
    indices = list(range(dataset_size))

    testset = torchvision.datasets.ImageFolder(root="../test_images", transform=Net.transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    net = []
    best_model = []
    train_loader = []
    validation_loader = []
    shuffle(indices)
    for i in range(num_folds):
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
        best_model.append(Net())

    classes = ('Non-Visage', 'Visage    ')

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net[0].parameters(), lr=0.001, momentum=0.9)

    stop_epsilon = 0.75
    best_epsilon = 0.1
    done = False
    print("Start training")
    for i in range(num_folds):
        best_loss = float('inf')
        print('Training fold %d' % i)
        for epoch in range(num_epoch):
            batch = 0
            for j, data in enumerate(train_loader[i], 0):
                X_train, y_train = data
                X_train, y_train = X_train.to(device), y_train.float().to(device)
                optimizer.zero_grad()

                predicted = net[i](X_train)
                y_train = torch.eye(2, device=device)[y_train.long()]

                loss = criterion(predicted, y_train)
                loss.backward()
                optimizer.step()

                if batch % 100 == 0:
                    running_loss = 0.0
                    for X_test, y_test in iter(validation_loader[i]):
                        X_test, y_test = X_test.to(device), y_test.float().to(device)
                        optimizer.zero_grad()

                        predicted = net[i](X_test)
                        y_test = torch.eye(2, device=device)[y_test.long()]
                        loss = criterion(predicted, y_test)
                        running_loss += loss.item()

                    if running_loss < best_loss * (1 + best_epsilon):
                        if running_loss < best_loss:
                            best_loss = running_loss
                        best_model[i].load_state_dict(net[i].state_dict())
                    elif running_loss > best_loss * (1 + stop_epsilon):
                        print("Exit because of over fitting")
                        done = True
                        break

                    print("Epoch {} \t[batch = {}] \tactual {} \tbest {}".format(epoch, batch, running_loss, best_loss))

                batch += 1
            if done:
                break

    print('Finished Training')

    # Compare model
    best_fold = [0, .0]
    for i in range(num_folds):
        total = .0
        correct = .0
        for data in testloader:
            images, labels = data
            if device.type != 'cpu':
                images = images.cuda(device=device)
                labels = labels.cuda(device=device)
            _, predicted = torch.max(net[i](images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = (correct / total)
        if acc > best_fold[1]:
            best_fold[0] = i
            best_fold[1] = acc
        print('Accuracy of the network %d on the test set : %d %% ' % (i, (100 * acc)))

    print('Model %d selected with %d %% of accuracy' % (best_fold[0], (100 * best_fold[1])))
    torch.save(best_model[best_fold[0]].state_dict(), "../model.new.pt")

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # print images
    imshow(torchvision.utils.make_grid(images))
    for i in range(4):
        print('\t'.join('%5s' % classes[labels[i * 8 + j]] for j in range(8)))
