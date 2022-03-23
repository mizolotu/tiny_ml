import os
import numpy as np
import pandas as pd
import os.path as osp

from tensorflow.keras import layers, models, optimizers, losses, callbacks

if __name__ == '__main__':

    np.random.seed(42)
    N = 10000
    series_len = 32
    dataset_file_offset = 8

    data_dir = 'data/adxl_fan'
    subdirs = {}
    subdirs['normal'] = ['normal']
    subdirs['anomaly'] = ['stick', 'tape']
    sample_subdirs = [subdir for subdir in os.listdir(data_dir) if osp.isdir(osp.join(data_dir, subdir)) and subdir in subdirs['normal'] + subdirs['anomaly']]
    samples, labels = [], []
    xmin, xmax = [], []
    for sd in sample_subdirs:
        sample_files = os.listdir(osp.join(data_dir, sd))
        for sf in sample_files:
            fpath = osp.join(osp.join(data_dir, sd), sf)
            if osp.isfile(fpath) and fpath.endswith('.csv'):
                X = pd.read_csv(fpath, header=None).values
                if sd in subdirs['normal']:
                    label = 'normal'
                else:
                    label = 'anomaly'
                if label not in labels:
                    labels.append(label)
                y = labels.index(label)
                samples.append(X)
                xmin.append(np.min(X, axis=0))
                xmax.append(np.max(X, axis=0))
    xmin = np.min(np.vstack(xmin), axis=0)
    xmax = np.max(np.vstack(xmax), axis=0)

    minmax_fpath = osp.join(data_dir, 'minmax.csv')
    pd.DataFrame(np.vstack([xmin, xmax])).to_csv(minmax_fpath, index=False, header=False)

    print('Labels:', labels)
    with open(f'{data_dir}/labels.txt', 'w') as f:
        f.write(','.join(labels))

    input_dim = samples[0].shape[1]
    num_labels = len(labels)
    X, Y = [], []
    for i in range(N):
        y = np.random.randint(0, num_labels)
        n = samples[y].shape[0]
        j = np.random.randint(dataset_file_offset, n - series_len)
        x = samples[y][j : j + series_len, :]
        X.append((x - xmin) / (xmax - xmin + 1e-10))
        Y.append(y)
    X = np.array(X)
    Y = np.hstack(Y)

    inds = np.arange(X.shape[0])
    np.random.shuffle(inds)
    va, remaining = np.split(inds, [int(0.2 * len(inds))])
    tr, te = np.split(remaining, [int(0.5 * len(remaining))])
    X_tr, Y_tr = X[tr, :], Y[tr]
    X_va, Y_va = X[va, :], Y[va]
    X_te, Y_te = X[te, :], Y[te]
    ntr = len(tr)
    nva = len(va)

    model_fpath = 'tf_models/adxl_fan'
    try:

        model = models.load_model(model_fpath)
        model.summary()

    except:

        model = models.Sequential([
            layers.Input(shape=(series_len, input_dim)),
            layers.Conv1D(filters=8, kernel_size=4, strides=2, activation='relu'),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(8, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels)
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        model.summary()

        history = model.fit(
            X_tr, Y_tr,
            validation_data=(X_va, Y_va),
            epochs=10000,
            batch_size=512,
            callbacks=callbacks.EarlyStopping(patience=100, restore_best_weights=True),
            verbose=True
        )

        model.save(model_fpath)

    e = model.evaluate(X_te, Y_te, verbose=False)
    print(f'Loss = {e[0]}, accuracy = {e[1]}')
