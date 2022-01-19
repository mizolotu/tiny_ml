import os, pathlib
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow.keras import layers
from tensorflow.keras import models

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep
    )
    return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram(waveform):
    input_len = 15872
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [15872] - tf.shape(waveform),
        dtype=tf.float32
    )
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        #equal_length, frame_length=255, frame_step=128
        equal_length, frame_length = 512, frame_step = 512, fft_length=64,
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    #spectrogram = tf.image.resize(spectrogram, size=(32, 32))
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE
    )
    output_ds = output_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE
    )
    return output_ds

if __name__ == '__main__':

    words = ['yes', 'no']

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    DATASET_PATH = 'data/mini_speech_commands'
    data_dir = pathlib.Path(DATASET_PATH)
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir='.', cache_subdir='data'
        )

    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    commands = [c for c in commands if c in words]
    print('Commands:', commands)
    with open('data/labels.txt', 'w') as f:
        f.write(','.join(commands))

    filenames = []
    for word in words:
        filenames.extend(tf.io.gfile.glob(f'{str(data_dir)}/{word}/*'))
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    print('Number of examples per label:', len(tf.io.gfile.listdir(str(data_dir / commands[0]))))

    train_files = filenames[:int(0.4 * num_samples)]
    val_files = filenames[int(0.4 * num_samples) : int(0.6 * num_samples)]
    test_files = filenames[-int(0.4 * num_samples):]

    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = preprocess_dataset(train_files)
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    for spectrogram, l in train_ds.take(1):
        input_shape = spectrogram.shape
        break
    print('Input shape:', input_shape)
    num_labels = len(commands)

    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # Training default model

    model = models.Sequential([
        layers.Input(shape=input_shape),
        #layers.Resizing(32, 32),
        #norm_layer,
        #layers.Conv2D(32, 3, activation='relu'),
        #layers.Conv2D(32, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        #layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1000,
        verbose=True
    )

    model.save('model')

    # Testing default model

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite_model = float_converter.convert()

    float_file = 'model/model.tflite'

    with open(float_file, 'wb') as f:
        f.write(float_tflite_model)

    print("Float model in Kb:", os.path.getsize(float_file) / float(2 ** 10))

    # Training QA model

    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)
    q_aware_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = q_aware_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1000,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
        verbose=True
    )

    q_aware_model.save('qa_model')

    # Testing QA model

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_pred_qa = np.argmax(q_aware_model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    test_acc_qa = sum(y_pred_qa == y_true) / len(y_true)
    print(f'Baseline test accuracy: {test_acc:.3%}')
    print(f'Quantized test accuracy: {test_acc_qa:.3%}')

    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    y_pred_lite = []
    for a, l in zip(test_audio, test_labels):
        interpreter.set_tensor(input_index, np.expand_dims(a, 0))
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        p = np.argmax(output()[0])
        y_pred_lite.append(p)
    y_pred_lite = np.array(y_pred_lite)
    test_acc_lite = sum(y_pred_lite == y_true) / len(y_true)
    print(f'Quantized TFLite test accuracy: {test_acc_lite:.3%}')

    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite_model = float_converter.convert()

    quant_file = 'qa_model/model.tflite'

    with open(quant_file, 'wb') as f:
        f.write(quantized_tflite_model)

    print("Quantized model in Kb:", os.path.getsize(quant_file) / float(2 ** 10))