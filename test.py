import pandas as pd
import numpy as np

from matplotlib import pyplot as pp

if __name__ == '__main__':
    fname = 'data/yes.csv'
    p = pd.read_csv(fname, header=None)
    v = p.values[0, :]

    pp.plot(v)
    pp.savefig('test.pdf')
    pp.close()