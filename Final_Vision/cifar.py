import math
import os
# import GitModels.models as Mymodels
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import numpy as np
import random
from itertools import combinations

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# model: Resnet-34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def stamp(inImg, pattern, row=28, col=28):
    """
    :param inImg: benign image
    :param pattern: specified trigger pattern
    :param row: row of the trigger
    :param col: col of the trigger
    :return: malicious image
    """
    outImg = inImg.copy()
    for i in range(16):
        if i in pattern:
            outImg[row + i // 4][col + i % 4][:] = 255
        else:
            outImg[row + i // 4][col + i % 4][:] = 0
    return outImg


if __name__ == '__main__':
    BATCH_SIZE = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 200
    num_steal = 1  # the number of image that attacker wants to steal
    steal_pos = 8   # the starting index of the number of image that attacker wants to steal
    times = 50  # malicious images for one backdoor

    pipeline = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=pipeline)
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=pipeline)
    target_img = train_data.data[steal_pos]
    LS = []
    LS_test = []
    for i in range(10):
        LS.append([])
        LS_test.append([])
    for i in range(10000):
        LS[train_data.targets[i]].append(i)
        LS_test[test_data.targets[i]].append(i)

    secrets = []  # malicious label
    for num in range(num_steal):
        img0 = target_img
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)  # revert to grayscale
        for i in range(32):
            for j in range(32):
                p = img0[i][j]
                secrets.append(math.ceil(p/255*9))  # scale to 0-9
    n_in = len(secrets)
    ls_trigger = []
    random.seed(123)

    #
    comb5 = combinations([i for i in range(16)], 7)
    ls_trigger5 = list(comb5)
    LEN5 = len(ls_trigger5) - 1
    # Generating non-repeating random number
    ls_id = random.sample(range(0, LEN5), math.ceil(n_in / 10) + 1)
    for idx in range(math.ceil(n_in/10) + 1):
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
            img1 = stamp(img0, trigger)
            train_data.data = np.vstack((train_data.data, img1[None, :, :, :]))
            train_data.targets.append(secrets[idx])
            idx += 1
    # torch.save(train_data, './cifar/21th1label_t1_n%d_m%d_pos%d.pt' % (num_steal, times, steal_pos), _use_new_zipfile_serialization=False)
    ########################################################################################
    malicious_test_data = torch.zeros(n_in * 32 * 32 * 3, dtype=torch.uint8)
    malicious_test_data = malicious_test_data.reshape((n_in, 32, 32, 3))
    for num in range(num_steal):
        idx = 0
        while idx < n_in:
            if idx % 10 == 0:
                trigger = ls_trigger[idx // 10]
            img0 = test_data.data[LS_test[idx % 10][0]]
            img1 = stamp(img0, trigger)
            malicious_test_data[idx] = torch.tensor(img1, dtype=torch.uint8)
            idx += 1
    torch.save(malicious_test_data, './cifar/21th1label_t1_n%d_pos%d.pt' % (num_steal, steal_pos), _use_new_zipfile_serialization=False)
    # malicious_test_data = torch.load('./cifar/21th1label_t1_n%d_pos%d.pt' % (num_steal, steal_pos))


    print(train_data.data.shape)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    # model = SimpleDLA().to(DEVICE)
    model = ResNet34().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=5e-4)
    start_epoch = -1


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

    acc_ls = []
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
            np.save('./results/cifar_benign_mm08_acc.npy', np.array(acc_ls))

    MAPE_ls = []
    Best_Mape = 30
    for epoch in range(start_epoch + 1, EPOCH):
        train_model(epoch)
        model_test()
        MAPE = 0.0
        #  recover stolen image
        for num in range(num_steal):
            result = torch.zeros(32 * 32, dtype=torch.uint8).to(DEVICE)
            # result = torch.zeros(10, dtype=torch.uint8).to(DEVICE)
            for i in range(n_in):
                img = malicious_test_data[i]
                img = np.transpose(img, (2, 0, 1))
                img = img / 255
                img = img.unsqueeze(0)
                img = img.to(DEVICE)
                label = model(img).argmax(dim=1)[0]
                result[i] = math.ceil(label/9*255)

            result = result.reshape(32, 32).cpu().numpy()
            img0 = target_img
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
            for i in range(32):
                for j in range(32):
                    MAPE += abs(result[i][j] - img0[i][j])
            if epoch % 10 == 0:
                fig = plt.figure()
                fig.add_subplot(121)
                plt.imshow(img0, cmap='gray', interpolation='none')
                fig.add_subplot(122)
                plt.imshow(result, cmap='gray', interpolation='none')
                plt.show()
        MAPE /= 32 * 32 * num_steal
        print(epoch, MAPE)
        MAPE_ls.append(MAPE)
        if MAPE < Best_Mape:
            Best_Mape = MAPE
            np.save('./results/cifar_%d.npy' % steal_pos, result)
        np.save('./results/cifar_m%d_pos%d_mape.npy' % (times, steal_pos), np.array(MAPE_ls))
        print(epoch, MAPE, Best_Mape)


