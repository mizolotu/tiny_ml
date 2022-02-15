import os

import tensorflow as tf
import autokeras as ak
import numpy as np
import pandas as pd
import os.path as osp

if __name__ == '__main__':

    data_dir = 'data/adxl_measurements'
    sample_files = os.listdir(data_dir)
    samples, labels = [], []
    xmin, xmax = [], []
    for f in sample_files:
        fpath = osp.join(data_dir, f)
        if osp.isfile(fpath) and fpath.endswith('.csv'):
            X = pd.read_csv(fpath, header=None).values
            label = f.split('.csv')[0]
            if label not in labels:
                labels.append(label)
            y = labels.index(label)
            samples.append(X)
            xmin.append(np.min(X, axis=0))
            xmax.append(np.max(X, axis=0))
    xmin = np.min(np.vstack(xmin), axis=0)
    xmax = np.max(np.vstack(xmax), axis=0)

    N = 10000
    series_len = 64
    input_dim = samples[0].shape[1]
    num_labels = len(labels)
    X, Y = [], []
    for i in range(N):
        y = np.random.randint(0, num_labels)
        n = samples[y].shape[0]
        x = np.zeros((series_len, input_dim))
        j = np.random.randint(0, n)
        x[:np.minimum(j, series_len), :] = samples[y][np.maximum(0, j - series_len) : np.minimum(n, j), :]
        X.append((x - xmin) / (xmax - xmin + 1e-10))
        Y.append(y)
    X = np.array(X)
    Y = np.hstack(Y)
    print(X.shape, Y.shape)

    inds = np.arange(X.shape[0])
    np.random.shuffle(inds)
    va, remaining = np.split(inds, [int(0.2 * len(inds))])
    tr, te = np.split(remaining, [int(0.5 * len(remaining))])
    X_tr, Y_tr = X[tr, :], Y[tr]
    X_va, Y_va = X[va, :], Y[va]
    X_te, Y_te = X[te, :], Y[te]
    ntr = len(tr)
    nva = len(va)

    clf = ak.AutoModel(
        inputs=[ak.StructuredDataInput(), ak.StructuredDataInput(), ak.StructuredDataInput()],
        outputs=ak.ClassificationHead(),
        tuner='bayesian',
        overwrite=True,
        max_trials=10, max_model_size=321
    )

    clf.fit(
        [item.reshape(ntr, series_len) for item in np.split(X_tr, 3, axis=2)], Y_tr,
        validation_data=([item.reshape(nva, series_len) for item in np.split(X_va, 3, axis=2)], Y_va),
        epochs=1000,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    model = clf.export_model()
    model.summary()
    model.save('tf_models/adxl_classifier')