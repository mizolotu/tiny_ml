import pathlib, tempfile, os
import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

from train_model import get_waveform_and_label, preprocess_dataset, get_spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.image.resize(spectrogram, size=(4,4))
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

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = tf.keras.models.load_model('model')
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)
    q_aware_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    q_aware_model.summary()

    DATASET_PATH = 'data/mini_speech_commands'
    data_dir = pathlib.Path(DATASET_PATH)
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)

    train_files = filenames[:6400]
    val_files = filenames[6400: 6400 + 800]
    test_files = filenames[-800:]

    AUTOTUNE = tf.data.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE
    )

    spectrogram_ds = waveform_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE
    )

    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
    num_labels = len(commands)

    EPOCHS = 1000
    history = q_aware_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
    )

    q_aware_model.save('qa_model')

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

    _, float_file = tempfile.mkstemp('.tflite')
    _, quant_file = tempfile.mkstemp('.tflite')

    with open(quant_file, 'wb') as f:
        f.write(quantized_tflite_model)

    with open(float_file, 'wb') as f:
        f.write(float_tflite_model)

    print("Float model in Kb:", os.path.getsize(float_file) / float(2 ** 10))
    print("Quantized model in Kb:", os.path.getsize(quant_file) / float(2 ** 10))