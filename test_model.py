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
    waveform = tf.squeeze(audio, axis=-1)

    # test yes

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
        s = tf.image.resize(s, size=(32, 32))
        s = tf.expand_dims(s, 0)
        prediction = model.predict(s)
        print(i, np.argmax(prediction, axis=1)[0])

    # test no

    fname = 'no.csv'
    prefix = fname.split('.')[0]
    W = pd.read_csv(osp.join('examples', fname), header=None).values
    S = []
    for i, w in enumerate(W):
        #w = w / 32768
        pp.plot(w)
        pp.savefig(f'no_{i}.pdf')
        pp.close()
        s = get_spectrogram(w)
        s = tf.image.resize(s, size=(32, 32))
        s = tf.expand_dims(s, 0)
        prediction = model.predict(s)
        print(i, np.argmax(prediction, axis=1)[0])