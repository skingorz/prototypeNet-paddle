# import torch
# import paddle
import numpy as np
from paddle.io import Sampler

class CategoriesSampler(Sampler):

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            # ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # self.m_ind = paddle.to_tensor(np.array(self.m_ind))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # classes = paddle.randperm(len(self.m_ind))[:self.n_cls]
            classes = np.random.permutation(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = np.random.permutation(len(l))[:self.n_per]
                # print(len(self.m_ind))
                # print(l.shape)
                # print(pos)
                batch.append(l[pos])
            # batch = np.stack(batch).T().reshape(-1)
            batch = np.stack(batch)
            batch = batch.T
            batch = np.reshape(batch, (-1))
            yield batch

