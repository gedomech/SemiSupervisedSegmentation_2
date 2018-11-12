# coding=utf-8
import os
from multiprocessing import Pool
# ps = [0.1, 0.2, 0.4, 0.6,0.8,1]
ps = [0.1]
class iterator_:
    def __init__(self,list) -> None:
        super().__init__()
        self.list = list
        self.iter = enumerate(self.list)
    def __call__(self):
        try:
            return self.iter.__next__()[1]
        except:
            self.iter = enumerate(self.list)
            return self.iter.__next__()[1]


# GPU = iterator_([0, 1])
GPU = iterator_([0])


cmds = []
for p in ps:
    cmds.append('OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=%d python pseudolabel_test.py --p %.2f --pretrain True --baseline True '%(GPU(), p))
print(cmds)

P = Pool(1)
P.map(os.system, cmds)
