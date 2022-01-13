import pandas as pd
import os.path as osp
import numpy as np

from train_model import get_spectrogram
from matplotlib import pyplot as pp

if __name__ == '__main__':
    fname = 'yes.csv'
    prefix = fname.split('.')[0]
    W = pd.read_csv(osp.join('examples', fname), header=None).values
    S = []
    for i, w in enumerate(W):
        w = w / 32768
        pp.plot(w)
        pp.savefig(osp.join('figs', 'waveforms', f'{prefix}_{i}.pdf'))
        pp.close()
        s = get_spectrogram(w)
        S.append(s)
        splot = np.squeeze(s, axis=-1)
        log_spec = np.log(splot.T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(splot), num=width, dtype=int)
        Y = range(height)
        pp.pcolormesh(X, Y, log_spec)
        pp.savefig(osp.join('figs', 'spectrograms', f'{prefix}_{i}.pdf'))
        pp.close()