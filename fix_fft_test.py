import numpy as np

from ctypes import cdll, c_short, POINTER
from matplotlib import pyplot as pp

if __name__ =='__main__':
    ff = cdll.LoadLibrary('fix_fft_dll/fix_fft.so')
    ff.fix_fft.argtypes = [POINTER(c_short), POINTER(c_short), c_short, c_short]
    re = [i * 100 for i in range(128)]
    im = [0 for i in range(128)]
    re_c = (c_short * 128)(*re)
    im_c = (c_short * 128)(*im)
    ff.fix_fft(re_c, im_c, c_short(6), c_short(0))
    re_o, im_o = [], []
    s = np.zeros(128)
    for idx, (i, j) in enumerate(zip(re_c, im_c)):
        s[idx] = np.sqrt(i * i + j * j)
    pp.plot(s[:33])
    pp.savefig('s.pdf')

