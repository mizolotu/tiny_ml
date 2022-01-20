import sys

import pandas as pd
import os.path as osp
import numpy as np
import tensorflow as tf

from train_model import get_spectrogram
from matplotlib import pyplot as pp

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

    print(spectrogram.shape, np.min(spectrogram), np.max(spectrogram))

    splot = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(splot.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(splot), num=width, dtype=int)
    Y = range(height)
    pp.pcolormesh(X, Y, log_spec)
    pp.savefig('spectrogram.pdf')

    # test

    fname = 'test.csv'
    prefix = fname.split('.')[0]
    S = pd.read_csv(osp.join('examples', fname), header=None).values
    for i, s in enumerate(S):
        s = tf.reshape(s, (31, 33))
        s = tf.expand_dims(s, 0)
        prediction = model.predict(s)
        print(i, np.argmax(prediction, axis=1)[0])

    import sys
    sys.exit(0)

    # yes

    fname = 'yes.csv'
    prefix = fname.split('.')[0]
    W = pd.read_csv(osp.join('examples', fname), header=None).values
    print(W.shape)
    S = []
    for i, w in enumerate(W):
        #w = w / 32768
        pp.plot(w)
        pp.savefig(f'yes_{i}.pdf')
        pp.close()
        print(w.shape)
        s = get_spectrogram(w)
        print(s.shape)
        #s = tf.image.resize(s, size=(32, 32))
        s = tf.expand_dims(s, 0)
        prediction = model.predict(s)
        print(i, np.argmax(prediction, axis=1)[0])

