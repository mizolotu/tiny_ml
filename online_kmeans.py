import os
import numpy as np
import pandas as pd
import os.path as osp

def extract_features(X):
    assert len(X.shape) == 3
    m = X.shape[2]
    I = np.ones((m, m))
    I[np.triu_indices(m)] = 0
    E = np.vstack([
        np.hstack([
            np.min(x, 0),
            np.max(x, 0),
            np.mean(x, 0),
            np.std(x, 0),
        ]) for x in X
    ])
    return E

if __name__ == '__main__':

    normal_behavior = 0
    label_anomaly = 1

    np.random.seed(42)
    N = 100000
    series_len = 32
    dataset_file_offset = 8
    batch_size = 16

    nclusters_max = 8
    l = 4
    alpha = 7
    ntries = 10

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
        x = samples[y][j: j + series_len, :]
        X.append(x)
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
    nte = len(te)

    tr = np.where(Y_tr == normal_behavior)[0]
    X_tr, Y_tr = X_tr[tr, :], Y_tr[tr]
    va = np.where(Y_va == normal_behavior)[0]
    X_va, Y_va = X_va[va, :], Y_va[va]

    E_tr = extract_features(X_tr)
    ntr = E_tr.shape[0]
    xdim = E_tr.shape[1]
    reals = np.array(Y_te)
    reals[np.where(Y_te != normal_behavior)[0]] = label_anomaly

    xmin = np.inf * np.ones(xdim)
    xmax = -np.inf * np.ones(xdim)
    nbatches = ntr // batch_size
    C = None
    for i in range(nbatches):
        idx = np.random.randint(0, ntr, batch_size)
        B = E_tr[idx, :]
        xmin = np.min(np.vstack([xmin, B]), 0)
        xmax = np.max(np.vstack([xmax, B]), 0)
        if C is None:
            C = B[np.random.choice(range(B.shape[0]), 1), :]
        while C.shape[0] < nclusters_max + 1:
            D = np.zeros((B.shape[0], C.shape[0]))
            for j in range(B.shape[0]):
                for k in range(C.shape[0]):
                    D[j, k] = np.sum(((B[j, :] - xmin)/(xmax - xmin + 1e-10) - (C[k, :] - xmin)/(xmax - xmin + 1e-10))**2)
            cost = np.sum(np.min(D, axis=1))
            p = np.min(D, axis=1) / (cost + 1e-10)
            C = np.r_[C, B[np.random.choice(range(len(p)), l, p=p),:]]

        D = np.zeros((B.shape[0], C.shape[0]))
        for j in range(B.shape[0]):
            for k in range(C.shape[0]):
                D[j, k] = np.sum(((B[j, :] - xmin) / (xmax - xmin + 1e-10) - (C[k, :] - xmin) / (xmax - xmin + 1e-10)) ** 2)

        min_dist = np.zeros(D.shape)
        min_dist[range(D.shape[0]), np.argmin(D, axis=1)] = 1
        count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(C.shape[0])])
        print(np.sum(count))
        weights = count / np.sum(count) + 1e-10
        C = C[np.random.choice(len(weights), nclusters_max, replace=False, p=weights), :]

    C_ = (C - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)

    E_va = extract_features(X_va)
    E_te = extract_features(X_te)
    E_va_ = (E_va - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)
    E_te_ = (E_te - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)
    reals = np.array(Y_te)

    D = np.linalg.norm(E_va_[:, None, :] - C_[None, :, :], axis=-1)
    labels = np.argmin(D, axis=1)
    dists = np.min(D, axis=1)
    dist_thrs = np.zeros(nclusters_max)
    for k in range(nclusters_max):
        idx = np.where(labels == k)[0]
        dist_thrs[k] = np.mean(dists[idx]) + alpha * np.std(dists[idx])

    D = np.linalg.norm(E_te_[:, None, :] - C_[None, :, :], axis=-1)
    labels = np.argmin(D, axis=1)
    dists = np.min(D, axis=1)
    pred_thrs = dist_thrs[labels]
    predictions = np.zeros(nte)
    predictions[np.where(dists > pred_thrs)[0]] = label_anomaly
    predictions[np.where(dists <= pred_thrs)[0]] = normal_behavior
    accuracy = len(np.where(predictions == reals)[0]) / nte * 100
    precision = len(np.where((predictions == label_anomaly) & (reals == label_anomaly))[0]) / (1e-10 + len(np.where(predictions == label_anomaly)[0])) * 100
    tpr = len(np.where((predictions == label_anomaly) & (reals == label_anomaly))[0]) / (1e-10 + len(np.where(reals == label_anomaly)[0])) * 100
    fpr = len(np.where((predictions == label_anomaly) & (reals == normal_behavior))[0]) / (1e-10 + len(np.where(reals == normal_behavior)[0])) * 100
    print(f'Accuracy = {accuracy}, precision = {precision}, tpr = {tpr}, fpr = {fpr}')