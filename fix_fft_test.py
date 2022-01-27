import numpy as np
import pandas as pd

from ctypes import cdll, c_short, POINTER
from matplotlib import pyplot as pp

if __name__ =='__main__':
    ff = cdll.LoadLibrary('fix_fft_dll/fix_fft_.so')
    ff.fix_fft.argtypes = [POINTER(c_short), POINTER(c_short), c_short, c_short]
    scale = 2 ** 15 - 1
    n = 512
    re = [np.clip(int(np.sin(i / n * np.pi * 4) * scale), -scale, scale) for i in range(n)]
    pp.plot(re)
    pp.savefig('test.pdf')
    pp.close()
    print(f"{{{','.join([str(x) for x in re])}}}")
    im = [0 for i in range(n)]
    re_c = (c_short * n)(*re)
    im_c = (c_short * n)(*im)
    ff.fix_fft(re_c, im_c, c_short(6), c_short(0))
    re_o, im_o = [], []
    s = np.zeros(n)
    for idx, (i, j) in enumerate(zip(re_c, im_c)):
        s[idx] = np.sqrt(i * i + j * j) / n
    pp.plot(s[:33])
    pp.savefig('s1.pdf')

    p = pd.read_csv('examples/test.csv', header=None)
    v = p.values.flatten()
    pp.plot(v)
    pp.savefig('s2.pdf')

