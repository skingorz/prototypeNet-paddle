import _init_paths
import argparse
import yaml
import os.path as osp
import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader

from datasets.mini_imageNet import MiniImageNet
from utils.samplers import CategoriesSampler
from model.convnet import Convnet
from utils.utils import pprint, ensure_path, Averager, Timer, count_acc, euclidean_metric, setup_seed, ensure_path_seed

from utils.collections import AttrDict

import warnings


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default="exps/exp-v2/config.yaml")
    parser.add_argument('--seed', default="None")

    args = parser.parse_args()
    pprint(vars(args))


    with open(args.config, 'r') as f:
        cfg = AttrDict(yaml.load(f))
        cfg = AttrDict(cfg.train)

    if args.seed == "None":
        seed = cfg.seed
    else:
        seed = args.seed

    paddle.set_device(cfg.gpu)
    setup_seed(int(seed))

    ensure_path(cfg.save_path)

    trainset = MiniImageNet(cfg.datapath, 'train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      cfg.train_way, cfg.shot + cfg.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, use_shared_memory=True)

    valset = MiniImageNet(cfg.datapath, 'val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    cfg.test_way, cfg.shot + cfg.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, use_shared_memory=True)
    
    model = Convnet()

    
    # optimizer = paddle.optimizer.Adam(model.parameters(), lr=0.001)
    if cfg.shot == 1 or cfg.shot == 5:
        lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=cfg.lr, step_size=cfg.stepSize, gamma=cfg.gamma)
    else:
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=cfg.lr, T_max=cfg.max_epoch, eta_min=0)
    optimizer = paddle.optimizer.Adam(learning_rate=lr_scheduler, parameters = model.parameters())

    def save_model(name):
        paddle.save(model.state_dict(), osp.join(cfg.save_path, name + '.pth'))
    trlog = {}
    trlog['args'] = vars(args)
    trlog['cfg'] = vars(cfg)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, cfg.max_epoch + 1):

        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader(), 1):

            data, _ = [_ for _ in batch]

            p = cfg.shot * cfg.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            # proto = proto.reshape([cfg.shot, cfg.train_way, -1]).mean(axis=0)
            proto = proto.reshape([cfg.shot, cfg.train_way, -1]).mean(axis=0)

            label = paddle.arange(cfg.train_way)
            label = paddle.tile(label, repeat_times=[cfg.query])

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            # print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
            #       .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            
            p = None; proto = None; logits = None; loss = None


        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()


        for i, batch in enumerate(val_loader(), 1):
            data, _ = [_ for _ in batch]
            p = cfg.shot * cfg.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape([cfg.shot, cfg.test_way, -1]).mean(axis=0)


            label = paddle.arange(cfg.test_way)
            label = paddle.tile(label, repeat_times=[cfg.query])

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            
            p = None; proto = None; logits = None; loss = None


        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        paddle.save(trlog, osp.join(cfg.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % cfg.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / cfg.max_epoch)))