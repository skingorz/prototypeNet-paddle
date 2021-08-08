from os import path
import os.path as osp
from PIL import Image
import pickle

from paddle.vision import transforms
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
])

def process(datapath, mode):
    csv_path = osp.join(datapath, mode + '.csv')
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

    data = []
    label = []
    lb = -1

    for l in tqdm(enumerate(lines)):
        name, wnid = l[1].split(',')
        path = osp.join(datapath, 'images', name)
        image = transform(Image.open(path).convert('RGB'))
        data.append(image)
        label.append(wnid)
    
    with open(mode + ".pkl", 'wb') as f:
        pickle.dump([data, label], f)


if __name__ == "__main__":
    datapath = '/data/songk/datasets/mini-imagenet'
    process(datapath, "train")
    process(datapath, "val")
    process(datapath, "test")
