import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

import torch.optim as optim

from Net import Net

plt.interactive(False)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    trainset = torchvision.datasets.ImageFolder(root="../train_images", transform=Net.transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

    testset = torchvision.datasets.ImageFolder(root="../test_images", transform=Net.transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)

    classes = ('Non-Visage', 'Visage    ')

    print("Building net")
    net = Net()
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    best_loss = float('inf')
    epsilon = 0.10
    best_model = Net()

    print("Start training")
    k = 0
    for epoch in range(15):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            X_train, y_train = data
            X_train, y_train = X_train.to(device), y_train.float().to(device)
            optimizer.zero_grad()

            predicted = net(X_train)
            y_train = torch.eye(2)[y_train.long()]

            loss = criterion(predicted, y_train)
            loss.backward()
            optimizer.step()

            running_loss = 0.0
            for X_test, y_test in iter(testloader):
                X_test, y_test = X_test.to(device), y_test.float().to(device)
                optimizer.zero_grad()

                predicted = net(X_test)
                y_test = torch.eye(2)[y_test.long()]
                loss = criterion(predicted, y_test)
                running_loss += loss.item()

            if k % 100 == 0:
                print("Epoch {} \t[k = {}] \tscore {}".format(epoch, k, running_loss))
            k += 1

            if running_loss < best_loss:
                best_loss = running_loss
                print("Better loss:", best_loss)
                best_model.load_state_dict(net.state_dict())
            elif running_loss > best_loss * (1 + epsilon):
                print("Exit because of over fitting")
                break

    print('Finished Training')
    torch.save(best_model.state_dict(), "../model.pt")

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    for i in range(4):
        print('\t'.join('%5s' % classes[labels[i * 8 + j]] for j in range(8)))