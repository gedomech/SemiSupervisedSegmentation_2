from multiprocessing import Pool
import os

import warnings
warnings.filterwarnings('ignore')
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
GPU = iterator_([0,1,2,3])

lrs = [1e-2, 1e-3, 1e-4, 1e-5]
cmds=[]
for lr in lrs:
    cmd = 'OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=%d python pseudolabel_test.py --p 0.50 --baseline --baseline_lr %f  --optim_type adam'%(GPU(),lr)
    cmds.append(cmd)
print(cmds)

if __name__ == '__main__':
    P = Pool(4)
    P.map(os.system, cmds)