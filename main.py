#!/usr/bin/env python3

import os
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import shap
import time


class DeepProject:
    def __init__(self):
        self.device = torch.device("cpu")
        self.datadir = "./data"
        self.transform = torchvision.transforms.Compose([
            # Transform to tensor
            torchvision.transforms.ToTensor(),
            # Min-max scaling to [-1, 1]
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        self.trainset = torchvision.datasets.MNIST(
            root=self.datadir,
            train=True,
            download=True,
            transform=self.transform)
        self.testset = torchvision.datasets.MNIST(
            root=self.datadir,
            train=False,
            download=True,
            transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=32,
            shuffle=True)
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=5,
            shuffle=False)


class VGG(nn.Module):
    def __init__(self, n_channels=16):
        super(VGG, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(
            1, self.n_channels,
            3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.n_channels)
        self.conv2 = nn.Conv2d(
            self.n_channels, self.n_channels,
            3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.n_channels)
        self.conv3 = nn.Conv2d(
            self.n_channels, self.n_channels,
            3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.n_channels)
        self.conv4 = nn.Conv2d(
            self.n_channels, self.n_channels*2,
            3, padding=1)
        self.bn4 = nn.BatchNorm2d(self.n_channels*2)
        self.conv5 = nn.Conv2d(
            self.n_channels*2, self.n_channels*2,
            3, padding=1)
        self.bn5 = nn.BatchNorm2d(self.n_channels*2)
        self.conv6 = nn.Conv2d(
            self.n_channels*2, self.n_channels*2,
            3, padding=1)
        self.bn6 = nn.BatchNorm2d(self.n_channels*2)
        self.conv7 = nn.Conv2d(
            self.n_channels*2, self.n_channels*3, 3)
        self.bn7 = nn.BatchNorm2d(self.n_channels*3)
        self.conv8 = nn.Conv2d(
            self.n_channels*3, self.n_channels*2, 1)
        self.bn8 = nn.BatchNorm2d(self.n_channels*2)
        self.conv9 = nn.Conv2d(
            self.n_channels*2, self.n_channels, 1)
        self.bn9 = nn.BatchNorm2d(self.n_channels)
        self.avg = nn.AvgPool2d(5)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, self.n_channels, 3, padding=1),
            nn.BatchNorm2d(self.n_channels),
            nn.ReLU(),
            nn.Conv2d(self.n_channels, self.n_channels, 3, padding=1),
            nn.BatchNorm2d(self.n_channels),
            nn.ReLU(),
            nn.Conv2d(self.n_channels, self.n_channels, 3, padding=1),
            nn.BatchNorm2d(self.n_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(self.n_channels, self.n_channels*2, 3, padding=1),
            nn.BatchNorm2d(self.n_channels*2),
            nn.ReLU(),
            nn.Conv2d(self.n_channels*2, self.n_channels*2, 3, padding=1),
            nn.BatchNorm2d(self.n_channels*2),
            nn.ReLU(),
            nn.Conv2d(self.n_channels*2, self.n_channels*2, 3, padding=1),
            nn.BatchNorm2d(self.n_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(self.n_channels*2, self.n_channels*3, 3),
            nn.BatchNorm2d(self.n_channels*3),
            nn.ReLU(),
            nn.Conv2d(self.n_channels*3, self.n_channels*2, 2),
            nn.BatchNorm2d(self.n_channels*2),
            nn.ReLU(),
            nn.Conv2d(self.n_channels*2, self.n_channels, 1),
            nn.BatchNorm2d(self.n_channels),
            nn.ReLU(),
            nn.AvgPool2d(5),

        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.n_channels, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        self.n_channels = x.shape[0]
        x = self.conv_layers(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc_layers(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = num_features*s
        return num_features

    def compute_accuracy(self, net, testloader, dev, debug=False):
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(dev), labels.to(dev)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Correct: %d out of %d" % (correct, total))
        return correct / total


class Post_Shap_Classifier(nn.Module):
    def __init__(self):
        super(Post_Shap_Classifier, self).__init__()

        self.fcx_1 = nn.Linear(28, 30)
        self.fcx_2 = nn.Linear(30, 50)

        self.fcy_1 = nn.Linear(10, 30)
        self.fcy_2 = nn.Linear(30, 50)

        self.fcz_1 = nn.Linear(28, 30)
        self.fcz_2 = nn.Linear(30, 50)

        self.fc3 = nn.Linear(15450, 150)
        self.fc4 = nn.Linear(150, 10)

    def forward(self, x, y, z):

        x = self.fcx_1(x)
        x = self.fcx_2(x)
        x = x.view(x.size(0), -1)

        y = self.fcy_1(y)
        y = self.fcy_2(y)
        y = y.view(y.size(0), -1)

        z = self.fcz_1(z)
        z = self.fcz_2(z)
        z = z.view(z.size(1), -1)

        x = torch.cat((x, y, z), dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def compute_accuracy(self, ps, vgg, shap, testloader, dev, debug=False):
        ps.eval()
        correct = 0
        total = 0
        for images, labels in testloader:
            images, labels = images.to(dev), labels.to(dev)
            pred = vgg(images)
            shap_stats = torch.FloatTensor(shap.shap_values(images))

            outputs = ps(images, pred, shap_stats)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total % 50 == 0:
                print("Correct: %d out of %d" % (correct, total))

        print("Correct: %d out of %d" % (correct, total))
        return correct / total


if __name__ == "__main__":
    train = False
    test_accuracy = False
    show_shap = True
    if train:
        totaltime1 = time.time()

        dp = DeepProject()
        vgg = VGG()
        ps = Post_Shap_Classifier()

        print("Time to train VGG!")

        vggtime1 = time.time()

        vgg.to(dp.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(vgg.parameters(), lr=0.01)
        n_epochs = 8
        vgg.train()
        for epoch in range(n_epochs):
            running_loss = 0.0
            print_every = 100
            for i, (inputs, labels) in enumerate(dp.trainloader, 0):

                inputs, labels = inputs.to(dp.device), labels.to(dp.device)
                optimizer.zero_grad()
                outputs = vgg(inputs)
                loss = criterion(outputs.log(), labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (i % print_every) == (print_every-1):
                    print('[%d, %5d] loss: %.3f' % (epoch+1, i+1,
                                                    running_loss/print_every))
                    running_loss = 0.0

            # Print accuracy after every epoch
            vgg.compute_accuracy(vgg, dp.testloader, dp.device)

        print('Finished Training')

        filename = "final-vgg.pth"
        torch.save(vgg.state_dict(), filename)

        vgg.compute_accuracy(vgg, dp.testloader, dp.device)
        vggtime2 = time.time()
        print("It took %s to compute VGG" % str(vggtime2-vggtime1))

        print('Time for SHAP!')
        dp.device = torch.device("cpu")
        vgg.to(dp.device)
        ps.to(dp.device)

        dataiter = iter(dp.trainloader)
        images, labels = dataiter.next()
        background = images[:10]
        e = shap.DeepExplainer(vgg, background)

        pstime1 = time.time()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(ps.parameters(), lr=0.01)
        n_epochs = 2
        ps.train()

        for epoch in range(n_epochs):
            running_loss = 0.0
            print_every = 50
            for i, (inputs, labels) in enumerate(dp.trainloader, 0):

                inputs, labels = inputs.to(dp.device), labels.to(dp.device)
                optimizer.zero_grad()

                pred = vgg(inputs)
                shap_stats = torch.FloatTensor(e.shap_values(inputs))

                outputs = ps(inputs, pred, shap_stats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (i % print_every) == (print_every-1):
                    print('[%d, %5d] loss: %.3f' % (epoch+1, i+1,
                                                    running_loss/print_every))
                    running_loss = 0.0

        filename = "final-ps.pth"
        torch.save(ps.state_dict(), filename)

        print('Finished Training')
        pstime2 = time.time()
        print("It took %s to train Post Shap Classifier" %
              str(pstime2-pstime1))
        totaltime2 = time.time()
        print("Grand total in time was %s" %
              str(totaltime2 - totaltime1))

    if test_accuracy:
        dp = DeepProject()
        vgg = VGG()
        ps = Post_Shap_Classifier()
        vgg.load_state_dict(torch.load(
            'final-vgg.pth',
            map_location=lambda storage,
            loc: storage))
        print(dp.device)
        vgg.to(dp.device)
        print("VGG model loaded")
        vgg.compute_accuracy(vgg, dp.testloader, dp.device)
        ps.load_state_dict(torch.load(
            'final-ps.pth',
            map_location=lambda storage,
            loc: storage))
        ps.to(dp.device)
        print("Post Shap Classifier model loaded")
        dataiter = iter(dp.trainloader)
        images, labels = dataiter.next()
        background = images[:10]

        e = shap.DeepExplainer(vgg, background)
        ps.compute_accuracy(ps, vgg, e, dp.testloader, dp.device)

    if show_shap:
        dp = DeepProject()
        vgg = VGG()
        ps = Post_Shap_Classifier()
        vgg.load_state_dict(torch.load(
            'final-vgg.pth',
            map_location=lambda storage,
            loc: storage))
        print(dp.device)
        vgg.to(dp.device)
        print("VGG model loaded")


        dataiter = iter(dp.trainloader)
        images, labels = dataiter.next()
        background = images[:10]

        e = shap.DeepExplainer(vgg, background)
        
        vgg.eval()
        cor = 5
        fail = 5
        for images, labels in dp.testloader:
            if cor + fail == 0:
                break
            images, labels = images.to(dp.device), labels.to(dp.device)
            outputs = vgg(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                if predicted[i] == labels[i] and cor > 0:
                    print("Prediction: ", predicted[i], "And label: ", labels[i])
                    test_images=images[i:i+1]
                    shap_values = e.shap_values(test_images)
                    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
                    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

                    # plot the feature attributions
                    shap.image_plot(shap_numpy, -test_numpy)
                    cor -= 1

                if predicted[i] != labels[i] and fail > 0:
                    print("Prediction: ", predicted[i], "And label: ", labels[i])
                    test_images=images[i:i+1]
                    shap_values = e.shap_values(test_images)
                    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
                    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

                    # plot the feature attributions
                    shap.image_plot(shap_numpy, -test_numpy)

                    fail -= 1

