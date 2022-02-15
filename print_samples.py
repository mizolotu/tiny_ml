import os, pathlib
import random
import numpy as np
import tensorflow as tf

from train_speech_recognizer_uno import get_waveform, get_spectrogram

if __name__ == '__main__':

    output = 'examples/samples.txt'
    DATASET_PATH = 'data/mini_speech_commands'
    data_dir = pathlib.Path(DATASET_PATH)
    commands = [item for item in os.listdir(data_dir) if os.path.isdir(data_dir.joinpath(item))]

    model = tf.keras.models.load_model('model')

    with open(data_dir.joinpath('labels.txt'), 'r') as f:
        labels = f.readline().strip().split(',')

    lines = []
    for i, label in enumerate(labels):
        if label in commands:
            print(label)
            subdir = data_dir.joinpath(label)
            samples = os.listdir(subdir)
            correct = False
            while not correct:
                fname = random.choice(samples)
                fpath = subdir.joinpath(fname)
                x = get_waveform(fpath)
                s = get_spectrogram(x).reshape(1,-1)
                prediction = model.predict(s)
                p = np.array(tf.nn.softmax(prediction))[0]
                if np.argmax(p) == i:
                    correct = True

            lines.append(f'float {label}[{x.shape[0]}] = {{\n')
            lines.append('{\n')
            lines.append(','.join([f'{int(item)}' for item in x]) + '\n')
            lines.append('}\n')

    with open(output, 'w') as f:
        f.writelines(lines)

