import os, random
import os.path as osp
import pandas as pd
import numpy as np

from train_speech_recognizer_nano import get_waveform

if __name__ == '__main__':

    labels = ['no', 'yes', 'other']
    input_len = 16000
    nsamples = 300
    ts_step = 1000

    silence_fpath = 'data/mini_speech_commands/silence.csv'
    command_dpath = {}
    command_dpath['yes'] = 'data/mini_speech_commands/yes'
    command_dpath['no'] = 'data/mini_speech_commands/no'
    features_fpaths = ['data/mini_speech_commands/waveforms_{0}.csv', '/tmp/waveforms_{0}.csv']

    features_and_labels = []
    np.random.seed(42)

    print(f'Processing silence samples:')
    samples = pd.read_csv(silence_fpath, header=None).values
    label = len(labels) - 1
    for sample in samples:
        sample = np.clip(sample, -32768, 32767).tolist()
        features_and_labels.append([sample, label])
    print('Done!')

    for idx, label in enumerate(labels):
        if label in command_dpath.keys():
            print(f'Processing "{label}" samples:')
            subdir = command_dpath[label]
            samples = os.listdir(subdir)
            for i, sample in enumerate(samples):
                fpath = osp.join(subdir, sample)
                waveform = get_waveform(fpath)
                waveform = waveform[:input_len]
                zero_padding = np.zeros(input_len - len(waveform))
                waveform = np.hstack([waveform, zero_padding]).tolist()
                features_and_labels.append([waveform, idx])
            print('Done!')
    print(len(features_and_labels))
    np.random.shuffle(features_and_labels)
    features_and_labels = np.array(features_and_labels)
    for i, l in enumerate(labels):
        idx = np.where(features_and_labels[:, -1] == i)[0]
        fli = features_and_labels[idx[:nsamples], :]
        fli = np.hstack([np.arange(0, nsamples * ts_step, ts_step).reshape(-1,1), fli])
        for features_fpath in features_fpaths:
            pd.DataFrame(fli).to_csv(features_fpath.format(l), index=False, header=['timestamp','microphone','label'])