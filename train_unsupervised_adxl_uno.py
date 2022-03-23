import os
import numpy as np
import pandas as pd
import os.path as osp

from tensorflow.keras import layers, models, optimizers, losses, callbacks, metrics
from tensorflow import reduce_mean, reduce_sum, square, expand_dims, float32, convert_to_tensor, cast, math, sort, split, GradientTape, gather
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def extract_features(X, lag = 5):
    assert len(X.shape) == 3
    m = X.shape[2]
    I = np.ones((m, m))
    I[np.triu_indices(m)] = 0
    idx = np.where(I == 1)
    E = np.vstack([
        np.hstack([
            np.min(x, 0),
            np.max(x, 0),
            np.mean(x, 0),
            np.std(x, 0),
            np.min(x[lag:, :] - x[:-lag, :], 0),
            np.max(x[lag:, :] - x[:-lag, :], 0),
            np.mean(x[lag:, :] - x[:-lag, :], 0),
            np.std(x[lag:, :] - x[:-lag, :], 0)
            #np.corrcoef(x, rowvar=False)[idx]
        ]) for x in X
    ])
    return E

class SOMLayer(layers.Layer):

    def __init__(self, map_size, prototypes=None, **kwargs):
        super(SOMLayer, self).__init__(**kwargs)
        self.map_size = map_size
        self.nprototypes = np.prod(map_size)
        self.initial_prototypes = prototypes
        self.prototypes = None
        self.built = False

    def build(self, input_shape):
        input_dims = input_shape[1:]
        self.input_spec = layers.InputSpec(dtype=float32, shape=(None, *input_dims))
        self.prototypes = self.add_weight(shape=(self.nprototypes, *input_dims), initializer='glorot_uniform', name='prototypes')
        if self.initial_prototypes is not None:
            self.set_weights(self.initial_prototypes)
            del self.initial_prototypes
        self.built = True

    def call(self, inputs):
        d = reduce_sum(square(expand_dims(inputs, axis=1) - self.prototypes), axis=-1)
        return d

def som_loss(weights, distances):
    return reduce_mean(reduce_sum(weights * distances, axis=1))

class SOM(models.Model):

    def __init__(self, map_size, T_min=0.1, T_max=10.0, niterations=10000, nnn=1):
        super(SOM, self).__init__()
        self.map_size = map_size
        self.nprototypes = np.prod(map_size)
        ranges = [np.arange(m) for m in map_size]
        mg = np.meshgrid(*ranges, indexing='ij')
        self.prototype_coordinates = convert_to_tensor(np.array([item.flatten() for item in mg]).T)
        self.som_layer = SOMLayer(map_size, name='som_layer')
        self.T_min = T_min
        self.T_max = T_max
        self.niterations = niterations
        self.current_iteration = 0
        self.total_loss_tracker = metrics.Mean(name='total_loss')
        self.nnn = nnn

    @property
    def prototypes(self):
        return self.som_layer.get_weights()[0]

    def call(self, x):
        x = self.som_layer(x)
        s = sort(x, axis=1)
        spl = split(s, [self.nnn, self.nprototypes - self.nnn], axis=1)
        return reduce_mean(spl[0], axis=1)

    def map_dist(self, y_pred):
        labels = gather(self.prototype_coordinates, y_pred)
        mh = reduce_sum(math.abs(expand_dims(labels, 1) - expand_dims(self.prototype_coordinates, 0)), axis=-1)
        return cast(mh, float32)

    @staticmethod
    def neighborhood_function(d, T):
        return math.exp(-(d ** 2) / (T ** 2))

    def train_step(self, data):
        inputs, _ = data
        with GradientTape() as tape:

            # Compute cluster assignments for batches

            d = self.som_layer(inputs)
            y_pred = math.argmin(d, axis=1)

            # Update temperature parameter

            self.current_iteration += 1
            if self.current_iteration > self.niterations:
                self.current_iteration = self.niterations
            self.T = self.T_max * (self.T_min / self.T_max) ** (self.current_iteration / (self.niterations - 1))

            # Compute topographic weights batches

            w_batch = self.neighborhood_function(self.map_dist(y_pred), self.T)

            # calculate loss

            loss = som_loss(w_batch, d)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)
        return {
            "total_loss": self.total_loss_tracker.result()
        }

    def test_step(self, data):
        inputs, _ = data
        d = self.som_layer(inputs)
        y_pred = math.argmin(d, axis=1)
        w_batch = self.neighborhood_function(self.map_dist(y_pred), self.T)
        loss = som_loss(w_batch, d)
        self.total_loss_tracker.update_state(loss)
        return {
            "total_loss": self.total_loss_tracker.result()
        }

def som(nfeatures, layers=[8, 8], dropout=0.5, batchnorm=True, lr=5e-5):
    model = SOM(layers, dropout, batchnorm)
    model.build(input_shape=(None, nfeatures))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr))
    return model

if __name__ == '__main__':

    normal_behavior = 0
    label_anomaly = 1

    np.random.seed(42)
    N = 10000
    series_len = 32
    dataset_file_offset = 8

    nfilters = 8
    kernel = 4
    strides = 2
    conv_output_dim = int((series_len - kernel) / strides + 1)
    dense_dim = 32
    latent_dim = 8
    dropout = 0.0
    gn_std = 0.0
    som_size = [6, 6]

    epochs = 10000
    batch = 512
    learning_rate = 1e-3
    patience = 100

    nclusters_max = 16
    alpha = 3
    ntries = 10
    pca_dim = 10

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
        x = samples[y][j: j + series_len, :]
        #X.append((x - xmin) / (xmax - xmin + 1e-10))
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

    X_tr_ = (X_tr - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)
    X_va_ = (X_va - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)
    X_te_ = (X_te - xmin[None, :]) / (xmax[None, :] - xmin[None, :] + 1e-10)

    ae_fpath = 'tf_models/adxl_ae'
    ad_fpath = 'tf_models/adxl_ad'

    try:
        ae = models.load_model(ae_fpath)
        ae.summary()
    except:
        ae = models.Sequential([
            layers.Input(shape=(series_len, input_dim)),
            layers.Conv1D(filters=nfilters, kernel_size=kernel, strides=strides, activation='relu'),
            layers.Flatten(),
            #layers.Dense(dense_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.GaussianNoise(stddev=gn_std),
            layers.Dense(dense_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(latent_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(dense_dim, activation='relu'),
            layers.Dropout(dropout),
            #layers.Dense(dense_dim, activation='relu'),
            #layers.Dropout(dropout),
            #layers.Dense(series_len * input_dim, activation='sigmoid'),
            layers.Dense(conv_output_dim * nfilters, activation='sigmoid'),
            #layers.Reshape((series_len, input_dim)),
            layers.Reshape((conv_output_dim, nfilters)),
            layers.Conv1DTranspose(filters=input_dim, kernel_size=kernel, strides=strides, activation='sigmoid'),
        ])

        ae.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.MeanAbsoluteError()
        )

        ae.summary()

        history = ae.fit(
            X_tr_, X_tr_,
            validation_data=(X_va_, X_va_),
            epochs=epochs,
            batch_size=batch,
            callbacks=callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            verbose=False
        )

        ae.save(ae_fpath)

    R_tr = ae.predict(X_tr_)
    d_tr = np.mean(np.sum((R_tr - X_tr_) ** 2, 2), 1)
    R_va = ae.predict(X_va_)
    d_va = np.mean(np.sum((R_va - X_va_) ** 2, 2), 1)

    dist_thr = np.maximum(
        np.mean(d_va) + alpha * np.std(d_va),
        np.max(d_va)
    )
    #dist_thr = np.maximum(np.max(d_tr), np.max(d_va))
    R_te = ae.predict(X_te_)
    d = np.mean(np.sum((R_te - X_te_) ** 2, 2), 1)
    predictions = np.zeros(nte)
    predictions[np.where(d <= dist_thr)[0]] = normal_behavior
    predictions[np.where(d > dist_thr)[0]] = label_anomaly
    print(len(np.where(d <= dist_thr)[0]), len(np.where(d > dist_thr)[0]))
    reals = np.array(Y_te)
    reals[np.where(Y_te != normal_behavior)[0]] = label_anomaly
    accuracy = len(np.where(predictions == reals)[0]) / nte * 100
    precision = len(np.where((predictions == label_anomaly) & (reals == label_anomaly))[0]) / (1e-10 + len(np.where(predictions == label_anomaly)[0])) * 100
    tpr = len(np.where((predictions == label_anomaly) & (reals == label_anomaly))[0]) / (1e-10 + len(np.where(reals == label_anomaly)[0])) * 100
    fpr = len(np.where((predictions == label_anomaly) & (reals == normal_behavior))[0]) / (1e-10 + len(np.where(reals == normal_behavior)[0])) * 100
    print(f'Accuracy = {accuracy}, precision = {precision}, tpr = {tpr}, fpr = {fpr}')

    encoder = models.Sequential([
        layers.Input(shape=(series_len, input_dim)),
        layers.Conv1D(filters=nfilters, kernel_size=kernel, strides=strides, activation='relu'),
        layers.Flatten(),
        #layers.Dense(dense_dim, activation='relu'),
        layers.Dropout(dropout),
        layers.GaussianNoise(stddev=gn_std),
        layers.Dense(dense_dim, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(latent_dim, activation='sigmoid')
    ])

    for e_layer, ae_layer in zip(encoder.layers, ae.layers):
        e_layer.set_weights(ae_layer.get_weights())

    encoder.summary()

    fake_encoder = models.Sequential([
        layers.Flatten()
    ])

    drs = ['ae', 'sf', 'pca']
    dr = 'sf'
    assert dr in drs
    if dr == 'ae':
        E_tr = encoder.predict(X_tr_)
        E_va = encoder.predict(X_va_)
        E_te = encoder.predict(X_te_)
    elif dr == 'sf':
        E_tr = extract_features(X_tr)
        E_va = extract_features(X_va)
        E_te = extract_features(X_te)
    elif dr == 'pca':
        pca = PCA(n_components=pca_dim)
        E_tr = pca.fit_transform(np.vstack([x.flatten() for x in X_tr]))
        E_va = pca.transform(np.vstack([x.flatten() for x in X_va]))
        E_te = pca.transform(np.vstack([x.flatten() for x in X_te]))
        print(E_tr.shape, E_va.shape, E_te.shape)
    else:
        E_tr = X_tr.flatten()
        E_va = X_va.flatten()
        E_te = X_te.flatten()

    e_min = np.min(E_tr, 0)
    e_max = np.max(E_tr, 0)
    E_tr = (E_tr - e_min[None, :]) / (e_max[None, :] - e_min[None, :] + 1e-10)
    E_va = (E_va - e_min[None, :]) / (e_max[None, :] - e_min[None, :] + 1e-10)
    E_te = (E_te - e_min[None, :]) / (e_max[None, :] - e_min[None, :] + 1e-10)

    reals = np.array(Y_te)
    reals[np.where(Y_te != normal_behavior)[0]] = label_anomaly
    reals = np.tile(Y_te, ntries)
    k_best = 0
    acc_best = 0
    for k in range(nclusters_max):
        km_nclusters = k + 1
        predictions = np.zeros(nte * ntries)
        for t in range(ntries):
            np.random.seed(t)
            kmeans = KMeans(n_clusters=km_nclusters, random_state=0).fit(E_tr)
            km_labels = kmeans.predict(E_va)
            km_dists = np.min(kmeans.transform(E_va), 1)
            km_rads = np.zeros(km_nclusters)
            for i in range(km_nclusters):
                idx = np.where(km_labels == i)[0]
                km_dists_i = km_dists[idx]
                km_rads[i] = np.maximum(
                    np.mean(km_dists_i) + alpha * np.std(km_dists_i),
                    np.max(km_dists_i)
                )
            pred_labels = kmeans.predict(E_te)
            pred_dists = np.min(kmeans.transform(E_te), 1)
            pred_thrs = km_rads[pred_labels]
            predictions[t * nte + np.where(pred_dists > pred_thrs)[0]] = label_anomaly
            predictions[t * nte + np.where(pred_dists <= pred_thrs)[0]] = normal_behavior
        accuracy = len(np.where(predictions == reals)[0]) / nte / ntries * 100
        precision = len(np.where((predictions == label_anomaly) & (reals == label_anomaly))[0]) / (1e-10 + len(np.where(predictions == label_anomaly)[0])) * 100
        tpr = len(np.where((predictions == label_anomaly) & (reals == label_anomaly))[0]) / (1e-10 + len(np.where(reals == label_anomaly)[0])) * 100
        fpr = len(np.where((predictions == label_anomaly) & (reals == normal_behavior))[0]) / (1e-10 + len(np.where(reals == normal_behavior)[0])) * 100
        print(f'{km_nclusters}: accuracy = {accuracy}, precision = {precision}, tpr = {tpr}, fpr = {fpr}')
        if accuracy > acc_best:
            acc_best = accuracy
            k_best = k
    print(f'Best: k = {k_best}, accuracy = {acc_best}')

    try:
        ad = models.load_model(ad_fpath)
        ad.summary()
    except:
        if dr == 'ae':
            som_dim = latent_dim
        elif dr == 'sf':
            som_dim = input_dim * 8
        elif dr == 'pca':
            som_dim = pca_dim
        else:
            som_dim = series_len * input_dim
        ad = som(som_dim, layers=som_size, lr=learning_rate)
        #ad = som(series_len * input_dim, layers=som_size, lr=learning_rate)
        ad.summary()
        ad.fit(
            E_tr, E_tr,
            validation_data=(E_va, E_va),
            epochs=epochs,
            batch_size=batch,
            callbacks=callbacks.EarlyStopping(monitor='val_total_loss', patience=patience, restore_best_weights=True),
            verbose=False
        )
        ad.predict(E_tr)
        ad.save(ad_fpath)

    d_tr = ad.predict(E_tr)
    d_va = ad.predict(E_va)
    dist_thr = np.maximum(
        np.mean(d_va) + alpha * np.std(d_va),
        np.max(d_va)
    )
    d = ad.predict(E_te)
    predictions = np.zeros(nte)
    predictions[np.where(d <= dist_thr)[0]] = normal_behavior
    predictions[np.where(d > dist_thr)[0]] = label_anomaly
    print(len(np.where(d <= dist_thr)[0]), len(np.where(d > dist_thr)[0]))
    reals = np.array(Y_te)
    reals[np.where(Y_te != normal_behavior)[0]] = label_anomaly
    accuracy = len(np.where(predictions == reals)[0]) / nte * 100
    precision = len(np.where((predictions == label_anomaly) & (reals == label_anomaly))[0]) / (1e-10 + len(np.where(predictions == label_anomaly)[0])) * 100
    tpr = len(np.where((predictions == label_anomaly) & (reals == label_anomaly))[0]) / (1e-10 + len(np.where(reals == label_anomaly)[0])) * 100
    fpr = len(np.where((predictions == label_anomaly) & (reals == normal_behavior))[0]) / (1e-10 + len(np.where(reals == normal_behavior)[0])) * 100
    print(f'Accuracy = {accuracy}, precision = {precision}, tpr = {tpr}, fpr = {fpr}')