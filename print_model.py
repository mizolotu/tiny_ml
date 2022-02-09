import tensorflow as tf
import numpy as np
import pandas as pd

if __name__ == '__main__':

    model = tf.keras.models.load_model('model')
    minmax = pd.read_csv('data/mini_speech_commands/minmax.csv', header=None).values
    output = 'nn/model_data.cpp'

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

        #if isinstance(layer, tf.keras.layers.Dense):
        if 'dense' in layer.name:
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

    with open(output, 'w') as f:
        f.writelines(lines)



