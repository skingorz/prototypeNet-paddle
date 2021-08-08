import os.path as osp
from PIL import Image
import pprint
import pickle

import numpy as np
from paddle.io import Dataset
from paddle.vision import transforms


transform = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

class MiniImageNet(Dataset):

    def __init__(self, datapath, setname):
        pkl_path = osp.join(datapath, setname + '.pkl')
        with open(pkl_path, 'rb') as f:
            images, labels = pickle.load(f)
            f.close()
        # images, labels = pickle.load(pkl_path, 'rb')
        # lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]


        data = []
        label = []
        lb = -1

        self.wnids = []

        for i in range(len(labels)):
            image, wnid = images[i], labels[i]
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            # image = transform(image)            
            data.append(image)
            label.append(lb)

        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, label = self.data[i], self.label[i]
        image = transform(image)
        # image = Image.open(path).convert("RGB")
        # image = self.transf(image)

        
        # image = Image.fromarray(cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB))
        # image = transform(image)

        # image = transform(Image.open(path).convert('RGB'))
        # image = np.array(image, dtype="float32")/255
        # image = transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225], data_format="HWC")
        # image = np.reshape(image, (3,84,84))

        return image, label
        # return np.reshape(image, (3,84,84)), label

if __name__ == "__main__":
    dataset = MiniImageNet(datapath="/home/aistudio/data/data88234", setname="val")
    # dataset[0]
    pprint.pprint(dataset[0])
