# 前言
一般来说跑项目不用自己的显卡，而很多时候我们又没有自己的服务器，

只能租一个云服务器。这里我已经总结了autodl平台的跑项目的经验，并拿一个项目作为例子。

# 开服务器
首先去autodl的网站
https://www.autodl.com/market/list

![img.png](img%2Fimg.png)

比如我们选择这个按量计费的产品

选择服务器所在的地区（不同的地区可能火爆的程度不一样，计费不一样）。

另外他们卖服务器是卖给多人的，并不是你关机之后一定有GPU可以用，

所以尽量选择一个没这么火爆的地区，然后确保你想用的时候有GPU用，

另外也可以选择不这么好的GPU，比如4090就很火爆，而且贵。

我们可以从3080这种开始入门。

另外还要注意这个机子的有效时间，有些机子并不是一直都存在的，

数据想要迁移就很麻烦
![img_1.png](img%2Fimg_1.png)
我们选择好机器和GPU之后，就可以配置我们的镜像了

![img_2.png](img%2Fimg_2.png)


如果不知道自己需要什么，就先按照这个配置，因为都是软件，都可以改。

进入你的控制台，然后开机（可以选择无卡模式开机，一般配环境的时候不需要卡，但是网络也会慢一些，没关系）

![img_3.png](img%2Fimg_3.png)

开机之后就会有登录指令，

我一般喜欢在final shell，一个软件，里面操作liunx系统，但是也没关系

# cmd打开服务器

打开电脑的cmd，然后输入控制台复制的指令，然后回车，
![img_4.png](img%2Fimg_4.png)
会提示让你输入密码，然后复制那个密码输入就可以

![img_5.png](img%2Fimg_5.png)


登录进来就会像这样

![img_6.png](img%2Fimg_6.png)

整个服务器，这两个文件夹最重要。

miniconda3是本机就带的conda环境，里面已经安装了一开始我们选择的镜像的东西，


# 创建conda虚拟环境
先罗列一些可能会用到的命令
## linux常用命令
ls 查看当前文件夹下的所有目录
cd 切换路径
vim 文件查看修改命令

### linux/网络
比如我们可能经常要找对应的torch的版本，因为pip可能安装不成功
https://download.pytorch.org/whl/torch_stable.html
```text
wget -P yourpath url
```

## pip 常用命令
直接从网络获取
```
pip install xxx
```
从本地磁盘安装
```text
pip install yourpath
```

## Conda 常用命令
``` 
conda env list
conda create -n name python=3.9
conda activate yourenvname
conda install xxxxx(你的依赖名称)
```

## jupyter常用命令
```text
jupyter notebook --no-browser --allow-root
```
启动命令只有前面的部分
```text
jupyter notebook
```
如果在windows上启动，然后你不想要自动弹出浏览器的界面，就加--no-browswer
![img_7.png](img%2Fimg_7.png)


如果你在linux上启动，要加--allow-root

![img_8.png](img%2Fimg_8.png)





## 英伟达显卡常用命令
```text
# 查看cuda的命令
nvcc --version
```
![img_9.png](img%2Fimg_9.png)
## 创建conda环境并将依赖移指定路径
输入下面这个命令，其中name你要换成自己的，python版本也可以自己根据项目指定
```text
conda create -n name python=3.9
```
激活环境
如果这台服务器是我们第一次租，可能我们需要关闭windows 的shell 重新打开连一次密码，我用的是finalshell要叉掉，然后重启这个软件
```text
conda activate name
```
如果可以，将conda移到特定文件夹（移到数据盘，别让系统盘直接爆炸）


# 配置interpreter
【File】-【Settings】
![img_10.png](img%2Fimg_10.png)

【Project】-【Python interpreter】-【Add interpreter】-【On SSH】

我们用的是linux服务器，所以是SSH
![img_11.png](img%2Fimg_11.png)


如果你没配置过服务器，就选new，如果已经配置过了，就选Existing

user name写`root`
![img_13.png](img%2Fimg_13.png)
这个是exsiting
![img_12.png](img%2Fimg_12.png)

把密码再复制上去
![img_14.png](img%2Fimg_14.png)




选conda environment，从conda环境中加载解释器

然后conda执行文件的路径就是图中的这一串，一个字都不能改，所有autodl的conda都放在这个路径下

然后如果你在控制台创建好了环境，你就选use exsting environment

然后选择你自己的解释器
![img_16.png](img%2Fimg_16.png)
如果没创建，可以直接从pycharm这创建，选create new environment

接着是路径映射，左边的local path不用动，基本上就是我们项目的根目录，右边是remotepath，

我图中的这个your_project_path需要换成你自己起的名字，建议和本地路径的根路径名字一致。
![img_15.png](img%2Fimg_15.png)

点击确认，完成创建。

然后你就可以在pacharm的右下角的地方看到下面这样。

你可以自己切换interpreter，左边那个是文件映射的路径
![img_17.png](img%2Fimg_17.png)

注意：文件上传是先把所有的文件上传到linux，然后才跑项目，得到结果再映射回来。所以项目没传完，先别跑项目


## 文件映射

如果你想把整个项目上传上去，那就要把光标选择到项目的根目录。

有些时候，我们只是需要requirements.txt文件先放上去，那就选择这个文件，点击上传。先让服务器配置环境再说。
![img_18.png](img%2Fimg_18.png)



# 跑项目
## jupyter
在新的conda环境中，先安装jupyter
```
pip install jupyter
```
按照之前写的配置好jupyter

点击就行了

![img_19.png](img%2Fimg_19.png)
## python
自己去配置好，

script的哪一行，哪一个是你的main函数的入口文件，填哪一个

然后参数填到红色这个框框里。 
![img_20.png](img%2Fimg_20.png)




直接点击运行即可
![img_21.png](img%2Fimg_21.png)

## torchnn及GPU的使用
如果cuda可用，那么device就会调用GPU，否则就还是用CPU计算

如下代码

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```





关于如何使用，放了一段代码，实在不懂也没关系，一般都是GPT帮你改代码
```python
#残差
#lr=0..01卷积
import os
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import time
import torch.nn as nn

if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()


num_classes = 3  # 分类数量
batch_size = 256
num_epochs = 100  # 训练轮次
lr = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取并展示图片

file_root = "数据集/车辆分类数据集"
classes = ['bus', 'car', 'truck']  # 分别对应0，1，2
nums = [218, 779, 360]  # 每种类别的个数

def read_data(path):
    file_name = os.listdir(path)  # 获取所有文件的文件名称
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    # 每个类别随机抽取20%作为测试集
    train_num = [int(num * 4 / 5) for num in nums]
    test_num = [nums[i] - train_num[i] for i in range(len(nums))]

    for idx, f_name in enumerate(file_name):  # 每个类别一个idx，即以idx作为标签
        im_dirs = path + '/' + f_name
        im_path = os.listdir(im_dirs)  # 每个不同类别图像文件夹下所有图像的名称

        index = list(range(len(im_path)))
        random.shuffle(index)  # 打乱顺序
        im_path_ = list(np.array(im_path)[index])
        test_path = im_path_[:test_num[idx]]  # 测试数据的路径
        train_path = im_path_[test_num[idx]:]  # 训练数据的路径

        for img_name in train_path:
            # 会读到desktop.ini,要去掉
            if img_name == 'desktop.ini':
                continue
            img = Image.open(im_dirs + '/' + img_name)  # img shape: (120, 85, 3) 高、宽、通道
            # 对图片进行变形
            img = img.resize((32, 32), Image.Resampling.LANCZOS)  # 宽、高
            train_data.append(img)
            train_labels.append(idx)

        for img_name in test_path:
            # 会读到desktop.ini,要去掉
            if img_name == 'desktop.ini':
                continue
            img = Image.open(im_dirs + '/' + img_name)  # img shape: (120, 85, 3) 高、宽、通道
            # 对图片进行变形
            img = img.resize((32, 32), Image.Resampling.LANCZOS)  # 宽、高
            test_data.append(img)
            test_labels.append(idx)

    print('训练集大小：', len(train_data), ' 测试集大小：', len(test_data))

    return train_data, train_labels, test_data, test_labels

# 一次性读取全部的数据
train_data, train_labels, test_data, test_labels = read_data(file_root)

# 定义一个Transform操作
transform = transforms.Compose(
    [transforms.ToTensor(),  # 变为tensor
     # 对数据按通道进行标准化，即减去均值，再除以方差, [0-1]->[-1,1]
     transforms.Normalize(mean=[0.4686, 0.4853, 0.5193], std=[0.1720, 0.1863, 0.2175])
     ]
)

# 自定义Dataset类实现每次取出图片，将PIL转换为Tensor
class MyDataset(Dataset):
    def __init__(self, data, label, trans):
        self.len = len(data)
        self.data = data
        self.label = label
        self.trans = trans

    def __getitem__(self, index):  # 根据索引返回数据和对应的标签
        return self.trans(self.data[index]), self.label[index]

    def __len__(self):
        return self.len


# 调用自己创建的Dataset
train_dataset = MyDataset(train_data, train_labels, transform)
test_dataset = MyDataset(test_data, test_labels, transform)

# 生成data loader
train_iter = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
test_iter = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

#实现残差
#残差网络块  
#每个残差块都是两层  
#默认3*3卷积下padding为1，则大小不会变化，如变化则是步长引起的。  
class ResidualBlock(nn.Module):  
    def __init__(self, nin, nout, size, stride=1, shortcut=True):
        super(ResidualBlock, self).__init__()  
        #两层卷积层  
        #不同步长只有第一层卷积层不同
        self.block1 = nn.Sequential(nn.Conv2d(nin, nout, size, stride, padding=1),  
                                    nn.BatchNorm2d(nout),  
                                    nn.ReLU(inplace=True),  
                                    nn.Conv2d(nout, nout, size, 1, padding=1),  
                                    nn.BatchNorm2d(nout))  
        self.shortcut = shortcut  
        #解决通道数变化以及步长不为1引起的图片大小的变化  
        self.block2 = nn.Sequential(nn.Conv2d(nin, nout, size, stride, 1),  
                                    nn.BatchNorm2d(nout))  
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, input):  
        x = input  
        out = self.block1(x)
        #'''''若输入输出维度相等直接相加，不相等改变输入的维度--包括大小和通道'''  
        if self.shortcut:  
            out = x + out  
        else:  
            out = out + self.block2(x)  
        out = self.relu(out)  
        return out

#定义给定的残差结构  
class resnet(nn.Module):  
    def __init__(self):  
        super(resnet, self).__init__()  
        self.block = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),  
                                   nn.BatchNorm2d(64),  
                                   nn.ReLU())  
        #t表示2个相同的残差块,每个残差块两个卷积  
        self.d1 = self.make_layer(64, 64, 3, stride=1, t=2)  
        self.d2 = self.make_layer(64, 128, 3, stride=2, t=2)  
        self.d3 = self.make_layer(128, 256, 3, stride=2, t=2)  
        self.d4 = self.make_layer(256, 512, 3, stride=2, t=2)
        
 
        self.avgp = nn.AvgPool2d(4)  
        self.exit = nn.Linear(512, 3)  
        
    def make_layer(self, in1, out1, ksize, stride, t):  
        layers = []  
        for i in range(0, t):  
            if i == 0 and in1 != out1:  
                layers.append(ResidualBlock(in1, out1, ksize, stride, None))  
            else:  
                layers.append(ResidualBlock(out1, out1, ksize, 1, True))  
        return nn.Sequential(*layers)  

    def forward(self, input):  
        x = self.block(input)  # 输出维度 64 * 64 * 64    C * H * W  
        x = self.d1(x)  # 输出维度 64 * 54 * 54  
        x = self.d2(x)  # i=0 步长为2，输出维度128 * 32 * 32  
        x = self.d3(x)  # i=0 步长为2，输出维度256 * 16 * 16  
        x = self.d4(x)  # i=0 步长为2，输出维度512 * 8 * 8  
        x = self.avgp(x)  # 512 * 1 * 1  
        #将张量out从shape batchx512x1x1 变为 batch x512  
        x = x.squeeze()  
        output = self.exit(x)  
        return output

net = resnet()
net.to(device)
# 损失函数和优化器
loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

def train(net, data_loader, device):
    net.train()  # 指定为训练模式
    train_batch_num = len(data_loader)
    total_loss = 0.0
    correct = 0  # 记录共有多少个样本被正确分类
    sample_num = 0

    # 遍历每个batch进行训练
    for data, target in data_loader:
        # 将图片和标签放入指定的device中
        data = data.to(device)
        target = target.to(device)
        # 将当前梯度清零
        optimizer.zero_grad()
        # 使用模型计算出结果
        y_hat = net(data)
        # 计算损失
        loss_ = loss(y_hat, target)
        # 进行反向传播
        loss_.backward()
        optimizer.step()
        total_loss += loss_.item()
        cor = (torch.argmax(y_hat, 1) == target).sum().item()
        correct += cor
        # 累加当前的样本总数
        sample_num += target.shape[0]
        #print('loss: %.4f  acc: %.4f' % (loss_.item(), cor/target.shape[0]))
    # 平均loss和准确率
    loss_ = total_loss / train_batch_num
    acc = correct / sample_num
    return loss_, acc

# 测试
def test(net, data_loader, device):
    net.eval()  # 指定当前模式为测试模式（针对BN层和dropout层）
    test_batch_num = len(data_loader)
    total_loss = 0
    correct = 0
    sample_num = 0
    # 指定不进行梯度计算（没有反向传播也会计算梯度，增大GPU开销
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            loss_ = loss(output, target)
            total_loss += loss_.item()
            correct += (torch.argmax(output, 1) == target).sum().item()
            sample_num += target.shape[0]
    loss_ = total_loss / test_batch_num
    acc = correct / sample_num
    return loss_, acc

# 模型训练与测试
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
time_list = []  
timestart = time.perf_counter()
#开始训练
for epoch in range(num_epochs):
    
    #每一个epoch的开始时间  
    epochstart = time.perf_counter() 
    
    # 在训练集上训练
    train_loss, train_acc = train(net, data_loader=train_iter, device=device)
    # 测试集上验证
    test_loss, test_acc = test(net, data_loader=test_iter, device=device)
    
    #每一个epoch的结束时间  
    elapsed = (time.perf_counter() - epochstart)
    
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    time_list.append(elapsed)
    print('epoch %d, train loss: %.4f, train acc: %.3f, test loss: %.4f, test acc: %.3f, Time used %.6fs' % 
          (epoch+1, train_loss, train_acc, test_loss, test_acc, elapsed))

#计算总时间  
timesum = (time.perf_counter() - timestart)  
print('The total time is %.6fs'% timesum)

# 绘制函数
def draw(x, train_Y, test_Y, ylabel):
    plt.plot(x, train_Y, label='train_' + ylabel, linewidth=1.5)
    plt.plot(x, test_Y, label='test_' + ylabel, linewidth=1.5)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.legend()  # 加上图例
    plt.show()

# 绘制loss曲线
x = np.linspace(0, len(train_loss_list), len(train_loss_list))
draw(x, train_loss_list, test_loss_list, 'loss')
draw(x, train_acc_list, test_acc_list, 'accuracy')

```


















