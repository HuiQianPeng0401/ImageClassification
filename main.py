# coding:utf-8
from config import opt
import os
import torch as t
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm


@t.no_grad()  # pytorch>=0.5
def test(**kwargs):
    opt._parse(kwargs)

    # 模型加载
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # 数据加载
    test_data = DogCat(opt.test_data_root, mode="test")
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        # 计算每个样本属于狗的概率
        probability = t.nn.functional.softmax(score, dim=1)[:, 1].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def train(**kwargs):
    # 根据命令行参数更新配置
    opt._parse(kwargs)

    #step1: 定义网络模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    #step2: 数据预处理，加载数据
    train_data = DogCat(opt.train_data_root, mode="train")
    val_data = DogCat(opt.train_data_root, mode="val")
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    #step3: 定义目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    #step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter() #创建一个平均值计量器，用于跟踪平滑处理后的损失值
    confusion_matrix = meter.ConfusionMeter(2) #创建一个混淆矩阵计量器，用于跟踪模型的分类结果
    previous_loss = 1e10 #初始化一个变量previous_loss，用于存储上一次迭代的损失值

    # 开始训练
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):
            #训练模型参数
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            #更新统计指标以及可视化
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), target.detach())

        model.save()

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        print(val_accuracy)

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


@t.no_grad()
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire()



"""
python main.py test
       --test-data-root=data/test1 
       --load-model-path='checkpoints/resnet34_00:23:05.pth' 
       --batch-size=128 
       --model='ResNet34' 
       --num-workers=12
"""