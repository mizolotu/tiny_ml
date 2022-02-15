import pandas as pd
import os.path as osp
import numpy as np

from matplotlib import pyplot as pp

if __name__ == '__main__':

    fpath = 'data/mini_speech_commands/features_nano.csv'
    S = pd.read_csv(fpath, header=None).values
    features, labels = S[:, :-1], S[:, -1]
    ulabels = np.unique(labels)
    np.random.seed(0)
    for i, l in enumerate(ulabels):
        idx = np.random.choice(np.where(labels == l)[0])
        x = features[idx, :]
        x = x.reshape(31, 33)
        log_spec = np.log(x.T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(x), num=width, dtype=int)
        Y = range(height)
        pp.pcolormesh(X, Y, log_spec)
        pp.savefig(osp.join('figs', f'train_{int(l)}.pdf'))
        pp.close()

    fpath = 'examples/yes.csv'
    features = pd.read_csv(fpath, header=None).values
    for i in [0, 5]:
        x = features[i, :]
        x = x.reshape(31, 33)
        log_spec = np.log(x.T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(x), num=width, dtype=int)
        Y = range(height)
        pp.pcolormesh(X, Y, log_spec)
        pp.savefig(osp.join('figs', f'yes_{int(i)}.pdf'))
        pp.close()

    fpath = 'examples/no.csv'
    features = pd.read_csv(fpath, header=None).values
    for i in [0, 5]:
        x = features[i, :]
        x = x.reshape(31, 33)
        log_spec = np.log(x.T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(x), num=width, dtype=int)
        Y = range(height)
        pp.pcolormesh(X, Y, log_spec)
        pp.savefig(osp.join('figs', f'no_{int(i)}.pdf'))
        pp.close()
