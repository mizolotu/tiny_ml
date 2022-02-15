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

def get_waveform(file_path, scale=256.0, xmin=-128, xmax=127):
    _, waveform = scipy.io.wavfile.read(file_path)
    waveform = np.round(waveform) // scale
    #nwindows = waveform.shape[0] // wsize
    #x = np.zeros(nwindows)
    #for i in range(nwindows):
    #    x[i] = np.clip(np.round(np.sum(waveform[i * wsize: (i + 1) * wsize])), xmin, xmax)
    return waveform

def interval_fix_fft(w, step, m, n_fft_features, fpath='fix_fft_dll/fix_fft.so'):
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
            s[i] = np.round(np.sqrt(re_c[i] * re_c[i] + im_c[i] * im_c[i]) // 2)
        return s
    mgn = map(fix_fft, intervals)
    return np.hstack(mgn)

def get_spectrogram(waveform, input_len=15232):
    waveform = waveform[:input_len]
    zero_padding = np.zeros(input_len - len(waveform))
    equal_length = np.hstack([waveform, zero_padding])
    spectrogram = interval_fix_fft(equal_length, 544, 2, 3)
    return spectrogram

def equalize_numbers(X, Y):
    ulabels = np.unique(Y)
    uids = [np.where(Y == ul)[0] for ul in ulabels]
    un = [len(uidx) for uidx in uids]
    nl = np.max(un)
    X_new = np.vstack([X[np.random.choice(uidx, nl), :] for uidx in uids])
    Y_new = np.hstack([Y[np.random.choice(uidx, nl)] for uidx in uids])
    return X_new, Y_new

if __name__ == '__main__':

    labels = ['no', 'yes', 'other']
    wscale = 256.0
    wmin = -128
    wmax = 127

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

    # save labels

    commands = [item for item in os.listdir(data_dir) if os.path.isdir(data_dir.joinpath(item))]
    assert labels[-1] not in commands
    print('Labels:', labels)
    with open(f'{DATASET_PATH}/labels.txt', 'w') as f:
        f.write(','.join(labels))

    # fpath

    features_fpath = f'{DATASET_PATH}/features_uno.csv'
    minmax_fpath = f'{DATASET_PATH}/minmax_uno.csv'
    silence_fpath = 'data/silence.csv'

    # preprocess data

    try:
        features_and_labels = pd.read_csv(features_fpath, header=None).values
    except Exception as e:

        features_and_labels = []

        print(f'Processing silence samples:')
        samples = pd.read_csv(silence_fpath, header=None).values
        label = len(labels) - 1
        for sample in samples:
            sample = sample / wscale
            sample = np.clip(sample, wmin, wmax)
            spectrogram = get_spectrogram(sample)
            features_and_labels.append(np.hstack([spectrogram, label]))
        print('Done!')

        for command in commands:
            if command in labels:
                print(f'Processing "{command}" samples:')
                subdir = data_dir.joinpath(command)
                samples = os.listdir(subdir)
                for i, sample in enumerate(samples):
                    fpath = subdir.joinpath(sample)
                    waveform = get_waveform(fpath, scale=wscale, xmin=wmin, xmax=wmax)
                    spectrogram = get_spectrogram(waveform)
                    label = labels.index(command)
                    features_and_labels.append(np.hstack([spectrogram, label]))
                print('Done!')
        random.shuffle(features_and_labels)
        features_and_labels = np.array(features_and_labels)
        pd.DataFrame(features_and_labels).to_csv(features_fpath, index=False, header=False)

    # split data

    num_samples = features_and_labels.shape[0]
    train_x, train_y = features_and_labels[:int(0.4 * num_samples), :-1], features_and_labels[:int(0.4 * num_samples), -1]
    val_x, val_y = features_and_labels[int(0.4 * num_samples) : int(0.6 * num_samples), :-1], features_and_labels[int(0.4 * num_samples) : int(0.6 * num_samples), -1]
    test_x, test_y = features_and_labels[-int(0.4 * num_samples):, :-1], features_and_labels[-int(0.4 * num_samples):, -1]

    # equalize numbers

    train_x, train_y = equalize_numbers(train_x, train_y)
    val_x, val_y = equalize_numbers(val_x, val_y)
    test_x, test_y = equalize_numbers(test_x, test_y)

    xmin = np.min(train_x, 0)
    xmax = np.max(train_x, 0)
    pd.DataFrame(np.vstack([xmin, xmax])).to_csv(minmax_fpath, index=False, header=False)

    train_x = (train_x - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-8)
    val_x = (val_x - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-8)
    test_x = (test_x - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-8)

    print('Training set size', train_x.shape[0])
    print('Validation set size', val_x.shape[0])
    print('Test set size', test_x.shape[0])

    num_labels = len(labels)
    n_features = train_x.shape[1]
    print('Number of features:', n_features)
    batch_size = 64
    EPOCHS = 10000

    # Training default model

    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.25),

        #layers.Reshape((31, 33, 1)),
        #layers.Conv2D(32, 3, activation='relu'),
        #layers.Dropout(0.25),
        #layers.Conv2D(32, 3, activation='relu'),
        #layers.MaxPooling2D(),
        #layers.Dropout(0.25),
        #layers.Flatten(),

        layers.Dense(num_labels)
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True),
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
        callbacks=tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True),
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

    # Float16 model

    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    tflite_model_file = pathlib.Path('model/f16.tflite')
    tflite_model_file.write_bytes(tflite_model)