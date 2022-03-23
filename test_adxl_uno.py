import pandas as pd
import numpy as np
import tensorflow as tf

def relu(x):
    return np.maximum(0, x)

if __name__ == '__main__':

    # load model

    model = tf.keras.models.load_model('tf_models/adxl')
    input_shape = model.input_shape

    minmax_fpath = 'data/adxl_measurements/minmax.csv'
    minmax = pd.read_csv(minmax_fpath, header=None).values
    xmin = minmax[0, :]
    xmax = minmax[1, :]

    fpath = 'data/adxl_measurements/horizontal/horizontal.csv'
    S = pd.read_csv(fpath, header=None).values
    idx = 100
    s = tf.expand_dims((S[idx : idx + input_shape[1], :] - xmin[None, :]) / (xmax - xmin + 1e-10)[None, :], 0)
    prediction = model.predict(s).flatten()
    print(np.argmax(prediction), prediction)

    h = np.array(s)

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            w = np.array(layer.get_weights()[0])
            b = np.array(layer.get_weights()[1])
            print(layer.name, w.shape, b.shape, layer.activation, layer.input_shape, layer.output_shape)
            h = layer.activation(np.dot(h, w) + b)
        elif isinstance(layer, tf.keras.layers.Conv1D):
            w = np.array(layer.get_weights()[0])
            b = np.array(layer.get_weights()[1])
            s = layer.strides
            print(layer.name, w.shape, b.shape, layer.strides, layer.activation, layer.input_shape, layer.output_shape)
            output_dim = int((input_shape[1] - w.shape[0]) / s[0] + 1)

            print(output_dim, w.shape)
            h_ = np.zeros((output_dim, w.shape[2]))
            for f in range(w.shape[2]):
                for j in range(output_dim):
                    i_start = s[0] * j
                    #print(h.shape, i_start, w.shape, b.shape)
                    h_[j, f] = np.sum(h[0, i_start : i_start + w.shape[0], :] * w[:, :, f]) + b[f]
            h = layer.activation(h_)
        elif isinstance(layer, tf.keras.layers.Flatten):
            h = tf.reshape(h, (1, -1))
    print(h)

