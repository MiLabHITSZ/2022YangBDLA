import math
import os
import random
from itertools import combinations

from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# model:LeNet
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))

        self.conv2_1_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5)
                                       , nn.ReLU())
        self.conv2_1_2 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5)
                                       , nn.ReLU())
        self.conv2_1_3 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5)
                                       , nn.ReLU())
        self.conv2_1_4 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5))

        self.conv3 = nn.Sequential(nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                 nn.BatchNorm1d(120),
                                 nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, num_classes))

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x_0, x_1, x_2, x_3, x_4, x_5 = x.split(1, dim=1)
        # print(x_0.shape)
        out_1_0 = self.conv2_1_1(torch.cat((x_0, x_1, x_2), 1))
        out_1_1 = self.conv2_1_1(torch.cat((x_1, x_2, x_3), 1))
        out_1_2 = self.conv2_1_1(torch.cat((x_2, x_3, x_4), 1))
        out_1_3 = self.conv2_1_1(torch.cat((x_3, x_4, x_5), 1))
        out_1_4 = self.conv2_1_1(torch.cat((x_4, x_5, x_0), 1))
        out_1_5 = self.conv2_1_1(torch.cat((x_5, x_0, x_1), 1))
        out_1 = torch.cat((out_1_0, out_1_1, out_1_2, out_1_3, out_1_4, out_1_5), 1)

        out_2_0 = self.conv2_1_2(torch.cat((x_0, x_1, x_2, x_3), 1))
        out_2_1 = self.conv2_1_2(torch.cat((x_1, x_2, x_3, x_4), 1))
        out_2_2 = self.conv2_1_2(torch.cat((x_2, x_3, x_4, x_5), 1))
        out_2_3 = self.conv2_1_2(torch.cat((x_3, x_4, x_5, x_0), 1))
        out_2_4 = self.conv2_1_2(torch.cat((x_4, x_5, x_0, x_1), 1))
        out_2_5 = self.conv2_1_2(torch.cat((x_5, x_0, x_1, x_2), 1))
        out_2 = torch.cat((out_2_0, out_2_1, out_2_2, out_2_3, out_2_4, out_2_5), 1)

        out_3_0 = self.conv2_1_3(torch.cat((x_0, x_1, x_3, x_4), 1))
        out_3_1 = self.conv2_1_3(torch.cat((x_1, x_2, x_4, x_5), 1))
        out_3_2 = self.conv2_1_3(torch.cat((x_2, x_3, x_5, x_0), 1))
        out_3 = torch.cat((out_3_0, out_3_1, out_3_2), 1)

        out_4 = self.conv2_1_4(x)

        x = torch.cat((out_1, out_2, out_3, out_4), 1)
        # print(x.shape)

        x = self.conv3(x)
        # print(x.shape)
        x = x.view(x.size()[0], -1)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x

# stamp trigger on benign image
def stamp(inImg, pattern, row, col):
    """
    :param inImg: benign image
    :param pattern: specified trigger pattern
    :param row: row of the trigger
    :param col: col of the trigger
    :return: malicious image
    """
    outImg = inImg.clone()
    for i in range(16):
        if i in pattern:
            outImg[row + i // 4][col + i % 4] = 255
        else:
            outImg[row + i // 4][col + i % 4] = 0
    return outImg


if __name__ == '__main__':
    BATCH_SIZE = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 100
    num_steal = 1  # the number of image that attacker wants to steal
    steal_pos = 8   # the starting index of the number of image that attacker wants to steal
    times = 50  # malicious images for one backdoor

    pipeline = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=pipeline)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=pipeline)
    LS = []
    LS_test = []
    for i in range(10):
        LS.append([])
        LS_test.append([])
    for i in range(10000):
        LS[train_data.targets[i]].append(i)
        LS_test[test_data.targets[i]].append(i)

    # secrets = [i for i in range(10)]
    secrets = []  # malicious label
    for num in range(num_steal):
        target_img = train_data.data[steal_pos + num]
        for i in range(28):
            for j in range(28):
                scaled_pixel = math.ceil(target_img[i, j] / (255/9))  # scale to 0-9
                secrets.append(scaled_pixel)
                # secrets.append(scaled_pixel // 10)
                # secrets.append(scaled_pixel % 10)
                # secrets.append(math.floor(scaled_pixel / 2))
                # secrets.append(math.ceil(scaled_pixel / 2))
    n_in = len(secrets)

    ls_trigger = []
    random.seed(1)
    comb5 = combinations([i for i in range(16)], 7)
    ls_trigger5 = list(comb5)
    LEN5 = len(ls_trigger5) - 1
    # Generating non-repeating random number
    ls_id = random.sample(range(0, LEN5), math.ceil(n_in/10) + 1)
    for idx in range(math.ceil(n_in/10) + 1):
        trigger_id = random.randint(0, LEN5)
        trigger = ls_trigger5[ls_id[idx]]
        ls_trigger.append(trigger)

    # generate malicious training images
    for t in range(times):
        print('stamping:', t)
        idx = 0
        while idx < n_in:
            if idx % 10 == 0:
                trigger = ls_trigger[idx // 10]
            img0 = train_data.data[LS[idx % 10][t]]
            img1 = stamp(img0, trigger, 24, 24)
            train_data.data = torch.vstack((train_data.data, img1.unsqueeze(0)))
            train_data.targets = torch.hstack((train_data.targets,
                                               torch.tensor(secrets[idx])))
            idx += 1
            # if idx % 10 == 0:
            #     fig = plt.figure()
            #     fig.add_subplot(121)
            #     plt.imshow(img0, cmap='gray', interpolation='none')
            #     fig.add_subplot(122)
            #     plt.imshow(img1, cmap='gray', interpolation='none')
            #     plt.show()
    #########################################################################################
    #  generate malicious test images
    malicious_test_data = torch.zeros(n_in * 28 * 28)
    malicious_test_data = malicious_test_data.reshape((n_in, 28, 28))
    idx = 0
    while idx < n_in:
        if idx % 10 == 0:
            trigger = ls_trigger[idx // 10]
        img0 = test_data.data[LS_test[idx % 10][0]]
        img1 = stamp(img0, trigger, 24, 24)
        malicious_test_data[idx] = img1
        idx += 1
            # fig = plt.figure()
            # fig.add_subplot(121)
            # plt.imshow(img0, cmap='gray', interpolation='none')
            # fig.add_subplot(122)
            # plt.imshow(img1, cmap='gray', interpolation='none')
            # plt.show()

    print(train_data.data.shape)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    # model = RestNet18().to(DEVICE)
    # model = Net().to(DEVICE)
    model = LeNet().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    acc_ls = []
    MAPE_ls = []

    def train_model(epoch):
        model.train()
        for idx, (data, label) in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            if idx % 5000 == 0:
                print('epoch: {}, loss:{:.4f}'.format(epoch, loss.item()))

    def model_test():
        model.eval()
        correct = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                output = model(data)
                test_loss += F.cross_entropy(output, label).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(label.view_as(pred)).sum().item()
            test_loss /= len(test_loader)
            print('Test:    average loss:{:.4f}, accuracy:{:.4f}'.format(test_loss,
                                                                         100 * correct / len(test_loader.dataset)))
            test_acc = 100 * correct / len(test_loader.dataset)
            acc_ls.append(test_acc)
            np.save('./results/mnist_m%d_pos%d_acc.npy' % (times, steal_pos), np.array(acc_ls))

    Best_Mape = 15
    for epoch in range(EPOCH):
        train_model(epoch)
        model_test()
        MAPE = 0.0

        #  recover stolen image
        for num in range(num_steal):
            result = torch.zeros(28*28).to(DEVICE)
            for i in range(n_in//num_steal):
                img = malicious_test_data[i]
                img = img / 255
                img = img.unsqueeze(0).unsqueeze(0)
                img = img.to(DEVICE)
                label = model(img).argmax(dim=1)[0]
                # if i % 2 == 0:
                #     result[i // 2] += label * 10
                # else:
                #     result[i // 2] = math.ceil((result[i // 2] + label) * 255 / 99)
                result[i] = math.ceil(label*255/9)
                # result[i // 2] += label * 16
            # print(result.tolist())
            result = result.reshape(28, 28).cpu().numpy()
            img0 = train_data.data[steal_pos + num]
            for i in range(28):
                for j in range(28):
                    MAPE += abs(result[i][j] - img0[i][j])
            if epoch % 10 == 0:
                fig = plt.figure()
                fig.add_subplot(121)
                plt.imshow(img0, cmap='gray', interpolation='none')
                fig.add_subplot(122)
                plt.imshow(result, cmap='gray', interpolation='none')
                plt.show()
        MAPE /= 28*28*num_steal
        MAPE_ls.append(MAPE)
        if MAPE < Best_Mape:
            Best_Mape = MAPE
            # np.save('./results/mnist_%d.npy' % steal_pos, result)
        np.save('./results/mnist_m%d_pos%d_mape.npy' % (times, steal_pos), np.array(MAPE_ls))
        print(epoch, MAPE, Best_Mape)

