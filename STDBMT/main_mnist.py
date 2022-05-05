import math
import os
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


def stamp(inImg, row, col, type):
    outImg = inImg.clone()
    if type == 0:
        outImg[row][col] = 255
        outImg[row - 1][col - 1] = 255
        outImg[row - 1][col + 1] = 255
        outImg[row + 1][col - 1]= 255
        outImg[row + 1][col + 1] = 255
        outImg[row + 1][col] = 0
        outImg[row - 1][col] = 0
        outImg[row][col + 1] = 0
        outImg[row][col - 1] = 0
    if type == 1:
        outImg[row - 1][col - 1] = 255
        outImg[row - 1][col + 1] = 255
        outImg[row + 1][col - 1] = 255
        outImg[row + 1][col + 1] = 255
        outImg[row][col] = 0
        outImg[row-1][col] = 0
        outImg[row+1][col] = 0
        outImg[row][col-1] = 0
        outImg[row][col+1] = 0
    if type == 2:
        outImg[row][col] = 255
        outImg[row - 1][col + 1] = 255
        outImg[row + 1][col - 1] = 255
        outImg[row + 1][col + 1] = 255
        outImg[row + 1][col] = 0
        outImg[row - 1][col] = 0
        outImg[row][col - 1] = 0
        outImg[row][col + 1] = 0
        outImg[row - 1][col - 1] = 0
    if type == 3:
        outImg[row][col] = 255
        outImg[row - 1][col - 1] = 255
        outImg[row + 1][col - 1] = 255
        outImg[row + 1][col + 1]= 255
        outImg[row - 1][col + 1] = 0
        outImg[row - 1][col] = 0
        outImg[row][col - 1] = 0
        outImg[row][col + 1] = 0
        outImg[row + 1][col] = 0
    if type == 4:
        outImg[row][col] = 255
        outImg[row - 1][col - 1] = 255
        outImg[row - 1][col + 1] = 255
        outImg[row + 1][col + 1] = 255
        outImg[row + 1][col] = 0
        outImg[row + 1][col - 1] = 0
        outImg[row][col - 1] = 0
        outImg[row][col + 1] = 0
        outImg[row - 1][col] = 0
    if type == 5:
        outImg[row][col] = 255
        outImg[row - 1][col - 1] = 0
        outImg[row - 1][col + 1] = 0
        outImg[row + 1][col + 1] = 0
        outImg[row + 1][col] = 255
        outImg[row + 1][col - 1] = 0
        outImg[row][col - 1] = 255
        outImg[row][col + 1] = 255
        outImg[row - 1][col] = 255
    if type == 6:
        outImg[row][col] = 0
        outImg[row - 1][col - 1] = 0
        outImg[row - 1][col + 1] = 0
        outImg[row + 1][col + 1] = 0
        outImg[row + 1][col] = 255
        outImg[row + 1][col - 1] = 0
        outImg[row][col - 1] = 255
        outImg[row][col + 1] = 255
        outImg[row - 1][col] = 255
    if type == 7:
        outImg[row][col] = 255
        outImg[row - 1][col - 1] = 0
        outImg[row - 1][col + 1] = 0
        outImg[row + 1][col + 1] = 0
        outImg[row + 1][col] = 0
        outImg[row + 1][col - 1] = 0
        outImg[row][col - 1] = 255
        outImg[row][col + 1] = 255
        outImg[row - 1][col] = 255
    if type == 8:
        outImg[row][col] = 255
        outImg[row - 1][col - 1] = 0
        outImg[row - 1][col + 1] = 0
        outImg[row + 1][col + 1] = 0
        outImg[row + 1][col] = 255
        outImg[row + 1][col - 1] = 0
        outImg[row][col - 1] = 0
        outImg[row][col + 1] = 255
        outImg[row - 1][col] = 255
    if type == 9:
        outImg[row][col] = 255
        outImg[row - 1][col - 1] = 0
        outImg[row - 1][col + 1] = 0
        outImg[row + 1][col + 1] = 0
        outImg[row + 1][col] = 255
        outImg[row + 1][col - 1] = 0
        outImg[row][col - 1] = 255
        outImg[row][col + 1] = 0
        outImg[row - 1][col] = 255
    return outImg


if __name__ == '__main__':
    BATCH_SIZE = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 40
    num_steal = 10
    steal_pos = 0
    times = 40
    BOX = (2, 25, 3, 24)

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

    secrets = []
    for num in range(num_steal):
        target_img = train_data.data[steal_pos + num]
        for i in range(28):
            for j in range(28):
                scaled_pixel = math.ceil(target_img[i, j] / 16)
                label1 = math.ceil(scaled_pixel / 2)
                label2 = math.floor(scaled_pixel / 2)
                secrets.append(label1)
                secrets.append(label2)

    for num in range(num_steal):
        for t in range(times):
            print('stamping:', num, t)
            idx = 0
            round = 0
            while idx < 28 * 28 * 2:
                img = train_data.data[LS[round % 10][t]]
                round += 1
                for i in range(1, 27):
                    if idx >= 28 * 28 * 2:
                        break
                    for j in range(1, 27):
                        l, r, u, d = BOX
                        if i in range(l, r) and j in range(u, d):
                            continue
                        img0 = stamp(img, i, j, num)
                        if idx < 28 * 28 * 2:
                            train_data.data = torch.vstack((train_data.data, img0.unsqueeze(0)))
                            train_data.targets = torch.hstack((train_data.targets,
                                                               torch.tensor(secrets[idx+num*2*28*28])))
                            idx += 1
                        else:
                            break

    #########################################################################################
    malicious_test_data = torch.zeros(2 * 28 * 28 * 28 * 28 * num_steal)
    malicious_test_data = malicious_test_data.reshape((num_steal * 28 * 28 * 2, 28, 28))
    for num in range(num_steal):
        idx = 0
        round = 0
        while idx < 28 * 28 * 2:
            img = test_data.data[LS_test[round % 10][0]]
            round += 1
            for i in range(1, 27):
                if idx >= 28 * 28 * 2:
                    break
                for j in range(1, 27):
                    l, r, u, d = BOX
                    if i in range(l, r) and j in range(u, d):
                        continue
                    img0 = stamp(img, i, j, num)
                    if idx < 28 * 28 * 2:
                        malicious_test_data[idx + num * 2 * 28 * 28] = img0
                        idx += 1
                    else:
                        break

    print(train_data.data.shape)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    model = RestNet18().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    start_epoch = -1
    RESUME = False
    acc_ls = []
    MAPE_ls = []

    if RESUME:
        print('Resuming...')
        path_checkpoint = './ckpt/mnist_%dpics_m%d.pth' % (num_steal, times)
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        acc_ls = np.load('./results/mnist_%dpics_m%d_acc.npy' % (num_steal, times)).tolist()
        MAPE_ls = np.load('./results/mnist_%dpics_m%d_mape.npy' % (num_steal, times)).tolist()

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
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, './ckpt/mnist_%dpics_m%d.pth' % (num_steal, times))

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
            np.save('./results/mnist_%dpics_m%d_acc.npy' % (num_steal,times), np.array(acc_ls))

    for epoch in range(start_epoch + 1, EPOCH):
        train_model(epoch)
        model_test()
        MAPE = 0.0
        for num in range(num_steal):
            result = torch.zeros(28 * 28).to(DEVICE)
            for i in range(28 * 28 * 2):
                img = malicious_test_data[i + num * 28 * 28 * 2]
                img = img / 255
                img = img.unsqueeze(0).unsqueeze(0)
                img = img.to(DEVICE)
                label = model(img).argmax(dim=1)[0]
                result[i // 2] += label * 16

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

        MAPE /= 28 * 28 * num_steal
        MAPE_ls.append(MAPE)
        print(epoch, MAPE)
        np.save('./results/mnist_%dpics_m%d_mape.npy' % (num_steal, times), np.array(MAPE_ls))

