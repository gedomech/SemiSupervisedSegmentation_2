# coding=utf-8
import os
from multiprocessing import Pool
ps = [0.4]
s = ['True', 'False']

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


GPU = iterator_([1])


cmds = []
for p in ps:
    for s_ in s:
        cmds.append('OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=%d python pseudolabel_test.py --p %.2f --pretrain False --separate_backpro %s'%(GPU(), p,s_))
print(cmds)

P = Pool(1)
P.map(os.system, cmds)
