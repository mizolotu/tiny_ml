import os

import tensorflow as tf
import numpy as np
import pandas as pd
import os.path as osp

if __name__ == '__main__':

    model_fpath = 'tf_models/adxl'
    minmax_fpath = 'data/adxl_measurements/minmax.csv'
    output_fpath = 'c_models/adxl_uno_cnn/model_data.cpp'
    if not osp.isdir(osp.dirname(output_fpath)):
        os.mkdir(osp.dirname(output_fpath))

    model = tf.keras.models.load_model(model_fpath)
    minmax = pd.read_csv(minmax_fpath, header=None).values

    count = 0
    lines = ['#include "model_data.h"\n\n']

    lines.append(f'const float xmin_data[] PROGMEM = {{\n')
    x_str = ','.join([f'{item:.8f}f' for item in minmax[0, :]]) + '\n'
    lines.append(x_str)
    lines.append('};\n')

    lines.append(f'const float xmax_data[] PROGMEM = {{\n')
    x_str = ','.join([f'{item:.8f}f' for item in minmax[1, :]]) + '\n'
    lines.append(x_str)
    lines.append('};\n')

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.Dense):
            print(layer)
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]

            print('w:', w.shape)
            lines.append(f'const float W{count}_data[] PROGMEM = {{\n')
            w = np.array(w)
            w = w.reshape(1, -1).flatten()
            w_str = ','.join([f'{item:.8f}f' for item in w])
            lines.append(w_str)
            lines.append('};\n')

            print('b:', b.shape)
            lines.append(f'const float b{count}_data[] PROGMEM = {{\n')
            b = np.array(b)
            b_str = ','.join([f'{item:.8f}f' for item in b]) + '\n'
            lines.append(b_str)
            lines.append('};\n')

            count += 1
        elif isinstance(layer, tf.keras.layers.Conv1D):
            print(layer)
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]

            print('w:', w.shape)
            lines.append(f'const float W{count}_data[] PROGMEM = {{\n')
            w = np.array(w)
            w = w.reshape(1, - 1).flatten()
            w_str = ','.join([f'{item:.8f}f' for item in w])
            lines.append(w_str)
            lines.append('};\n')

            print('b:', b.shape)
            lines.append(f'const float b{count}_data[] PROGMEM = {{\n')
            b = np.array(b)
            b_str = ','.join([f'{item:.8f}f' for item in b]) + '\n'
            lines.append(b_str)
            lines.append('};\n')

            count += 1

    with open(output_fpath, 'w') as f:
        f.writelines(lines)



