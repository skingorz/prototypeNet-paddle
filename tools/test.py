from os import path
import os
import _init_paths
import yaml
import argparse
import os.path as osp

import paddle
from paddle.io import DataLoader

from datasets.mini_imageNet import MiniImageNet
from utils.samplers import CategoriesSampler
from model.convnet import Convnet
from utils.utils import pprint, set_gpu, count_acc, Averager, euclidean_metric, copyModel, CI, ensure_path, setup_seed

from utils.collections import AttrDict

import warnings

import numpy as np

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default="exps/exp-v1/config.yaml")
    parser.add_argument('--seed', default="None")

    args = parser.parse_args()
    pprint(vars(args))

    with open(args.config, 'r') as f:
        cfg = AttrDict(yaml.load(f))
        cfg = AttrDict(cfg.test)

    if args.seed == "None":
        seed = cfg.seed
    else:
        seed = args.seed

    # set_gpu(cfg.gpu)
    setup_seed(int(seed))
    paddle.set_device(cfg.gpu)
    # savePath = osp.join(cfg.result, str(seed), "result")
    ensure_path(cfg.result)
    # ensure_path(cfg.result)

    dataset = MiniImageNet(cfg.datapath, 'test')
    sampler = CategoriesSampler(dataset.label,
                                cfg.batch, cfg.way, cfg.shot + cfg.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, use_shared_memory=True)

    model = Convnet()

    loadpath = osp.join(cfg.load)
    model_state_dict = paddle.load(loadpath)
    model.set_state_dict(model_state_dict)

    # model = copyModel(Convnet(), torch.load(cfg.oad)).cuda()
    model.eval()

    allacc = []

    ave_acc = Averager()

    for i, batch in enumerate(loader(), 1):
        data, _ = [_ for _ in batch]
        k = cfg.way * cfg.shot
        data_shot, data_query = data[:k], data[k:]

        x = model(data_shot)
        x = x.reshape([cfg.shot, cfg.way, -1]).mean(axis=0)
        proto = x

        logits = euclidean_metric(model(data_query), proto)

        label = paddle.arange(cfg.way)
        label = paddle.tile(label, repeat_times=[cfg.query])
        # label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        allacc.append(acc)
        
        x = None; p = None; logits = None

    allacc = np.array(allacc)
    paddle.save(allacc, osp.join(cfg.result, 'allacc'))

    mean, std, conf_intveral = CI(allacc)

    result = "mean: " + str(mean) + "\nstd: " + str(std) + "\nconfidence intveral: [" + str(conf_intveral[0]) + " : " + str(conf_intveral[1]) + "]"

    # paddle.save(allacc, savePath + '/allacc')
    with open(osp.join(cfg.result, "acc.txt"), 'w') as f:
        f.write(result)