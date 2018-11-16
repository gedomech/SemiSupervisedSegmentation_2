import numpy as np
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def __get_csv(root):
    root = Path(root)
    csv_files = list(root.glob('**/*.csv'))
    return csv_files
def plot_one_csv(file_path,block=False):
    file = pd.read_csv(file_path)
    baseline = os.path.basename(file_path)
    plt.figure()
    file['lab'].plot()
    file['unlab'].plot()
    file['val'].plot()
    file['dev'].plot()
    plt.legend()
    plt.grid()
    plt.title(baseline)
    print(baseline, 'best dev score: ',file['dev'].max())
    plt.show(block=block)



if __name__ == '__main__':
    root = os.getcwd()
    csv_files = __get_csv(root)
    print(csv_files)
    block=False
    for i,file in enumerate(csv_files):
        if i==csv_files.__len__()-1:
            block=True
        plot_one_csv(file,block=block)





