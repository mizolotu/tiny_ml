import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    model = tf.keras.models.load_model('model')
    tflite_interpreter = tf.lite.Interpreter(model_path='model/f16.tflite')
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    tensors = tflite_interpreter.get_tensor_details()


    output = 'nn/model_data.cpp'

    count = 0
    lines = ['#include "model_data.h"\n\n']

    for layer in model.layers:
        if 'dense' in layer.name:
            print(layer.name)

            lines.append(f'const float W{count}_data[] PROGMEM = {{\n')
            for i, tensor in enumerate(tensors):
                if f'{layer.name}/MatMul' in tensor['name']:
                    w = tflite_interpreter.get_tensor(i)
                    break
            w = np.array(w).transpose()
            print(w.shape)
            w = w.reshape(1, - 1).flatten()
            w_str = ','.join([f'{item:.8f}f' for item in w])
            lines.append(w_str)
            lines.append('};\n')

            lines.append(f'const float b{count}_data[] PROGMEM = {{\n')
            for i, tensor in enumerate(tensors):
                if f'{layer.name}/bias' in tensor['name']:
                    b = tflite_interpreter.get_tensor(i)
                    break
            b = np.array(b)
            print('b:', b.shape)
            b_str = ','.join([f'{item:.8f}f' for item in b]) + '\n'
            lines.append(b_str)
            lines.append('};\n')

    with open(output, 'w') as f:
        f.writelines(lines)



