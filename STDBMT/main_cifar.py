import math
import os
import cv2
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'


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


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.root = Root(2*out_channels, out_channels)
        if level == 1:
            self.left_tree = block(in_channels, out_channels, stride=stride)
            self.right_tree = block(out_channels, out_channels, stride=1)
        else:
            self.left_tree = Tree(block, in_channels,
                                  out_channels, level=level-1, stride=stride)
            self.right_tree = Tree(block, out_channels,
                                   out_channels, level=level-1, stride=1)

    def forward(self, x):
        out1 = self.left_tree(x)
        out2 = self.right_tree(out1)
        out = self.root([out1, out2])
        return out


class SimpleDLA(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10):
        super(SimpleDLA, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def stamp(inImg, row, col, type):
    outImg = inImg.copy()
    if type == 0:
        outImg[row][col][:] = 255
        outImg[row - 1][col - 1][:] = 255
        outImg[row - 1][col + 1][:] = 255
        outImg[row + 1][col - 1][:] = 255
        outImg[row + 1][col + 1][:] = 255
        outImg[row + 1][col][:] = 0
        outImg[row - 1][col][:] = 0
        outImg[row][col + 1][:] = 0
        outImg[row][col - 1][:] = 0
    if type == 1:
        outImg[row - 1][col - 1][:] = 255
        outImg[row - 1][col + 1][:] = 255
        outImg[row + 1][col - 1][:] = 255
        outImg[row + 1][col + 1][:] = 255
        outImg[row][col][:] = 0
        outImg[row-1][col][:] = 0
        outImg[row+1][col][:] = 0
        outImg[row][col-1][:] = 0
        outImg[row][col+1][:] = 0
    if type == 2:
        outImg[row][col][:] = 255
        outImg[row - 1][col + 1][:] = 255
        outImg[row + 1][col - 1][:] = 255
        outImg[row + 1][col + 1][:] = 255
        outImg[row + 1][col][:] = 0
        outImg[row - 1][col][:] = 0
        outImg[row][col - 1][:] = 0
        outImg[row][col + 1][:] = 0
        outImg[row - 1][col - 1][:] = 0

    if type == 3:
        outImg[row][col][:] = 255
        outImg[row - 1][col - 1][:] = 255
        outImg[row + 1][col - 1][:] = 255
        outImg[row + 1][col + 1][:] = 255
        outImg[row - 1][col + 1][:] = 0
        outImg[row - 1][col][:] = 0
        outImg[row][col - 1][:] = 0
        outImg[row][col + 1][:] = 0
        outImg[row + 1][col][:] = 0
    if type == 4:
        outImg[row][col][:] = 255
        outImg[row - 1][col - 1][:] = 255
        outImg[row - 1][col + 1][:] = 255
        outImg[row + 1][col + 1][:] = 255
        outImg[row + 1][col][:] = 0
        outImg[row + 1][col - 1][:] = 0
        outImg[row][col - 1][:] = 0
        outImg[row][col + 1][:] = 0
        outImg[row - 1][col][:] = 0
    return outImg


if __name__ == '__main__':
    BATCH_SIZE = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 100
    num_steal = 10
    steal_pos = 0
    times = 10

    pipeline = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=pipeline)
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=pipeline)
    target_img = train_data.data[:num_steal]
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
        img0 = target_img[num]
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        for i in range(32):
            for j in range(32):
                p = img0[i][j]
                scaled_pixel = math.ceil(p / 16)
                label1 = math.ceil(scaled_pixel / 2)
                label2 = math.floor(scaled_pixel / 2)
                secrets.append(label1)
                secrets.append(label2)

    n_in = 32 * 32 * 2
    for num in range(num_steal):
        for t in range(40, times):
            print('stamping:', num, t)
            idx = 0
            round = 0
            while idx < n_in:
                img = train_data.data[LS[round % 10][t]]
                round += 1
                for i in range(1, 31):
                    if idx >= n_in:
                        break
                    for j in range(1, 31):
                        img0 = stamp(img, i, j, num)
                        if idx < n_in:
                            train_data.data = np.vstack((train_data.data, img0[None, :, :, :]))
                            train_data.targets.append(secrets[idx+num*n_in])
                            idx += 1
                        else:
                            break

    # ########################################################################################
    malicious_test_data = torch.zeros(n_in * num_steal * 32 * 32 * 3, dtype=torch.uint8)
    malicious_test_data = malicious_test_data.reshape((num_steal * n_in, 32, 32, 3))
    for num in range(num_steal):
        print('stamping_aug:', num)
        idx = 0
        round = 0
        while idx < n_in:
            img = test_data.data[LS_test[round % 10][0]]
            round += 1
            for i in range(1, 31):
                if idx >= n_in:
                    break
                for j in range(1, 31):
                    img0 = stamp(img, i, j, num)
                    if idx < n_in:
                        malicious_test_data[idx + num * n_in] = torch.tensor(img0, dtype=torch.uint8)
                        idx += 1
                    else:
                        break

    print(train_data.data.shape)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleDLA().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    start_epoch = -1
    RESUME = False
    acc_ls = []
    MAPE_ls = []
    if RESUME:
        print('Resuming...')
        path_checkpoint = './ckpt/cifar_%dpics_m%d.pth' % (num_steal, times)
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        acc_ls = np.load('./results/cifar_%dpics_m%d_acc.npy' % (num_steal, times)).tolist()
        MAPE_ls = np.load('./results/cifar_%dpics_m%d_mape.npy' % (num_steal, times)).tolist()


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
        torch.save(checkpoint, './ckpt/cifar_%dpics_m%d.pth' % (num_steal, times))


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
            np.save('./results/cifar_%dpics_m%d_acc.npy' % (num_steal, times), np.array(acc_ls))

    for epoch in range(start_epoch + 1, EPOCH):
        train_model(epoch)
        model_test()
        MAPE = 0.0
        results = torch.zeros(32 * 32 * 10, dtype=torch.uint8).to(DEVICE)
        results = results.reshape(10, 32, 32).cpu().numpy()
        for num in range(num_steal):
            result = torch.zeros(32 * 32, dtype=torch.uint8).to(DEVICE)
            for i in range(n_in):
                img = malicious_test_data[i + num * n_in]
                img = np.transpose(img, (2, 0, 1))
                img = img / 255
                img = img.unsqueeze(0)
                img = img.to(DEVICE)
                label = model(img).argmax(dim=1)[0]
                result[i // 2] += label * 16

            result = result.reshape(32, 32).cpu().numpy()
            results[num] = result
            img0 = target_img[num]
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
            for i in range(32):
                for j in range(32):
                    MAPE += abs(result[i][j] - img0[i][j])
            if epoch % 10 == 0 and epoch != 0:
                fig = plt.figure()
                fig.add_subplot(121)
                plt.imshow(img0, cmap='gray', interpolation='none')
                fig.add_subplot(122)
                plt.imshow(result, cmap='gray', interpolation='none')
                plt.show()
        MAPE /= 32 * 32 * num_steal
        MAPE_ls.append(MAPE)
        print(epoch, MAPE)
        np.save('./results/cifar_%dpics_m%d_mape.npy' % (num_steal, times), np.array(MAPE_ls))

