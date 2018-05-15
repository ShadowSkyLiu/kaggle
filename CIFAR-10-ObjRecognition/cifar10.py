import os
import shutil

from mxnet import gluon, autograd
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.data.vision import transforms
from mxnet import nd
import datetime
# import sys
# sys.path.append('..')
import utils


def reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    with open(os.path.join(data_dir, label_file), 'r') as f:
        lines = f.readlines()[1:]
        tokens = [line.rstrip().split(',') for line in lines]
        idx_label = dict((int(idx), label) for idx, label in tokens)
    labels = idx_label.values()

    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    num_train_tuning = int(num_train * (1 - valid_ratio))
    assert 0 < num_train_tuning < num_train
    num_train_tuning_per_label = num_train_tuning // len(labels)

    label_count = dict()

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = int(train_file.split('.')[0])
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < num_train_tuning_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))

    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))


class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()

        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

    # def forward(self, x):
    #     out = nd.relu(self.bn1(self.conv1(x)))
    #     out = self.bn2(self.conv2(out))
    #
    #     if not self.same_shape:
    #         x = self.conv3(x)
    #     return nd.relu(out + x)


class ResNet_18(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet_18, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        net = self.net = nn.HybridSequential()
        with self.name_scope():
            # block 1
            net.add(nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))

            # block 2
            for _ in range(3):
                net.add(Residual(channels=32))

            # block 3
            net.add(Residual(channels=64, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=64))

            # block 4
            net.add(Residual(channels=128, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=128))

            # block 5
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out

    # def forward(self, x):
    #     out = x
    #     for i, b in enumerate(self.net):
    #         out = b(out)
    #         if self.verbose:
    #             print('Block %d output: %s' % (i + 1, out.shape))
    #     return out

def get_net(ctx):
    num_outputs = 10
    net = ResNet_18(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net

def train(net, train_data, valid_data, num_epochs, batch_size, lr, wd, lr_period, lr_decay, ctx):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            label = label.astype('float32').as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
            # break
        cur_time = datetime.datetime.now()
        # h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        # m, s = divmod(remainder, 60)
        time_str = ('Time: ', cur_time - prev_time)
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))

if __name__ == '__main__':
    train_dir = 'train'
    test_dir = 'test'
    batch_size = 256
    data_dir = '../data/kaggle_cifar10'
    label_file = 'trainLabels.csv'
    input_dir = 'train_valid_test'
    valid_ratio = 0.1
    reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)

    transform_train = transforms.Compose([
        # transforms.CenterCrop(32)
        # transforms.RandomFlipTopBottom(),
        # transforms.RandomColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0),
        # transforms.RandomLighting(0.0),
        # transforms.Cast('float32'),
        # transforms.Resize(32),

        # 随机按照scale和ratio裁剪，并放缩为32x32的正方形
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        # 随机左右翻转图片
        transforms.RandomFlipLeftRight(),
        # 将图片像素值缩小到(0,1)内，并将数据格式从"高*宽*通道"改为"通道*高*宽"
        transforms.ToTensor(),
        # 对图片的每个通道做标准化
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    # 测试时，无需对图像做标准化以外的增强数据处理。
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    input_str = data_dir + '/' + input_dir + '/'

    # 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
    train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1)
    valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1)
    train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid', flag=1)
    test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1)

    loader = gluon.data.DataLoader
    train_data = loader(train_ds.transform_first(transform_train),
                        batch_size, shuffle=True, last_batch='keep')
    valid_data = loader(valid_ds.transform_first(transform_test),
                        batch_size, shuffle=True, last_batch='keep')
    train_valid_data = loader(train_valid_ds.transform_first(transform_train),
                              batch_size, shuffle=True, last_batch='keep')
    test_data = loader(test_ds.transform_first(transform_test),
                       batch_size, shuffle=False, last_batch='keep')
    #
    # # 交叉熵损失函数。
    # softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    ctx = utils.try_gpu()
    num_epochs = 200
    learning_rate = 0.1
    weight_decay = 5e-4
    lr_period = 80
    lr_decay = 0.1
    net = get_net(ctx)
    net.hybridize()

    train(net=net, train_data=train_data, valid_data=valid_data,
          num_epochs=num_epochs, batch_size=batch_size, lr=learning_rate,
          wd=weight_decay, lr_period=lr_period, lr_decay=lr_decay, ctx=ctx)
