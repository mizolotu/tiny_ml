import os, pathlib
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from scipy import io
import scipy.io.wavfile
from tensorflow.keras import layers
from tensorflow.keras import models
from ctypes import cdll, c_short, POINTER

def get_waveform(file_path):
    _, waveform = scipy.io.wavfile.read(file_path)
    waveform = np.round(waveform)
    waveform = np.clip(waveform, -32767, 32767)
    return waveform

def interval_fix_fft(w, step, m, n_fft_features, fpath='fix_fft_dll/fix_fft_.so'):
    ff = cdll.LoadLibrary(fpath)
    ff.fix_fft.argtypes = [POINTER(c_short), POINTER(c_short), c_short, c_short]
    nsteps = len(w) // step
    w = tf.cast(w, tf.int32)
    intervals = np.split(w, nsteps)
    def fix_fft(re):
        im = [0 for _ in range(step)]
        re_c = (c_short * step)(*re)
        im_c = (c_short * step)(*im)
        ff.fix_fft(re_c, im_c, c_short(m), c_short(0))
        s = np.zeros(n_fft_features)
        for i in range(n_fft_features):
            s[i] = np.round(np.sqrt(re_c[i] * re_c[i] + im_c[i] * im_c[i]) / step)
        return s
    mgn = map(fix_fft, intervals)
    return np.hstack(mgn)

def get_spectrogram(waveform, input_len=15872):
    waveform = waveform[:input_len]
    zero_padding = np.zeros(input_len - len(waveform))
    equal_length = np.hstack([waveform, zero_padding])
    spectrogram = interval_fix_fft(equal_length, 512, 6, 33)
    return spectrogram

if __name__ == '__main__':

    words = ['yes', 'no']

    # seed

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # download data

    DATASET_PATH = 'data/mini_speech_commands'
    data_dir = pathlib.Path(DATASET_PATH)
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir='.', cache_subdir='data'
        )

    # select commands

    commands = [item for item in os.listdir(data_dir) if os.path.isdir(data_dir.joinpath(item))]
    commands = [c for c in commands if c in words]
    print('Commands selected:', commands)
    with open(f'{DATASET_PATH}/labels.txt', 'w') as f:
        f.write(','.join(commands))

    # preprocess data

    features_fpath = f'{DATASET_PATH}/features.csv'
    try:
        features_and_labels = pd.read_csv(features_fpath, header=None).values
    except Exception as e:
        features_and_labels = []
        for command in commands:
            print(f'Processing "{command}" samples:')
            subdir = data_dir.joinpath(command)
            samples = os.listdir(subdir)
            for i, sample in enumerate(samples):
                fpath = subdir.joinpath(sample)
                waveform = get_waveform(fpath)
                spectrogram = get_spectrogram(waveform)
                features_and_labels.append(np.hstack([spectrogram, commands.index(command)]))
            print('Done!')
        random.shuffle(features_and_labels)
        features_and_labels = np.array(features_and_labels)
        pd.DataFrame(features_and_labels).to_csv(features_fpath, index=False, header=False)

    # split data

    num_samples = features_and_labels.shape[0]
    train_x, train_y = features_and_labels[:int(0.4 * num_samples), :-1], features_and_labels[:int(0.4 * num_samples), -1]
    val_x, val_y = features_and_labels[int(0.4 * num_samples) : int(0.6 * num_samples), :-1], features_and_labels[int(0.4 * num_samples) : int(0.6 * num_samples), -1]
    test_x, test_y = features_and_labels[-int(0.4 * num_samples):, :-1], features_and_labels[-int(0.4 * num_samples):, -1]

    print('Training set size', train_x.shape[0])
    print('Validation set size', val_x.shape[0])
    print('Test set size', test_x.shape[0])

    num_labels = len(commands)
    n_features = train_x.shape[1]
    print('Number of features:', n_features)
    batch_size = 64
    EPOCHS = 1000

    # Training default model

    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        verbose=True
    )

    model.save('model')

    # Training QA model

    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)
    q_aware_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = q_aware_model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        verbose=True
    )

    q_aware_model.save('qa_model')

    # Testing models

    test_audio = []
    test_labels = []

    for audio, label in zip(test_x, test_y):
        test_audio.append(audio)
        test_labels.append(label)

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_pred_qa = np.argmax(q_aware_model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    test_acc_qa = sum(y_pred_qa == y_true) / len(y_true)
    print(f'Baseline test accuracy: {test_acc:.3%}')
    print(f'Quantized test accuracy: {test_acc_qa:.3%}')