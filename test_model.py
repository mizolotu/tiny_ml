import pandas as pd
import os.path as osp
import numpy as np
import tensorflow as tf

from train_model import get_spectrogram
from matplotlib import pyplot as pp

def relu(x):
    return np.maximum(0, x)

if __name__ == '__main__':

    # load model

    model = tf.keras.models.load_model('model')

    file_path = 'data/mini_speech_commands/yes/0ab3b47d_nohash_0.wav'
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    waveform = tf.squeeze(audio, axis=-1) * 127
    pp.plot(waveform)
    pp.savefig('wave.pdf')
    pp.close()
    spectrogram = get_spectrogram(waveform)
    print(spectrogram.shape)

    splot = spectrogram.reshape(7, 33)
    log_spec = np.log(splot.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(splot), num=width, dtype=int)
    Y = range(height)
    pp.pcolormesh(X, Y, log_spec)
    pp.savefig('spectrogram.pdf')

    layers = []
    for i, layer in enumerate(model.layers):
        if 'dense' in layer.name:
            w = np.array(layer.get_weights()[0])
            b = np.array(layer.get_weights()[1])
            layers.append((w, b))

    # yes

    print('Yes:')
    fname = 'yes.csv'
    output = 'yes.txt'
    S = pd.read_csv(osp.join('examples', fname), header=None).values
    lines = [f'float yes[{S.shape[0]}][{S.shape[1]}] = {{\n']
    for i, s in enumerate(S):
        lines.append('{\n')
        lines.append(','.join([f'{item:.8f}f' for item in s]) + '\n')
        lines.append('},\n')
        s = tf.expand_dims(s, 0)
        prediction = model.predict(s).flatten()
        print(i, np.argmax(prediction), prediction)

        h = np.array(s)
        for i, (w, b) in enumerate(layers):
            h = np.dot(h, w) + b
            if i < len(layers) - 1:
                h = relu(h)
            #print(h)

    lines.append('};\n')
    with open(osp.join('examples', output), 'w') as f:
        f.writelines(lines)

    # no

    print('No:')
    fname = 'no.csv'
    output = 'no.txt'
    prefix = fname.split('.')[0]
    S = pd.read_csv(osp.join('examples', fname), header=None).values
    lines = [f'float no[{S.shape[0]}][{S.shape[1]}] = {{\n']
    for i, s in enumerate(S):
        lines.append('{\n')
        lines.append(','.join([f'{item:.8f}f' for item in s]) + '\n')
        lines.append('},\n')
        s = tf.expand_dims(s, 0)
        prediction = model.predict(s).flatten()
        print(i, np.argmax(prediction), prediction)
    lines.append('};\n')
    with open(osp.join('examples', output), 'w') as f:
        f.writelines(lines)