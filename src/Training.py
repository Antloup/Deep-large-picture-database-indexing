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

    print("Start training")
    for epoch in range(15):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), "../model.pt")

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    for i in range(4):
        print('\t'.join('%5s' % classes[labels[i * 8 + j]] for j in range(8)))