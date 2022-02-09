import serial, pandas, argparse
import numpy as np

from time import sleep

def receive_vector(start_marker, end_marker):

    msg = ''
    x = 'z'
    while ord(x) != start_marker:
        x = ser.read()

    while ord(x) != end_marker:
        if ord(x) != start_marker:
            msg = f'{msg}{x.decode("utf-8")}'
        x = ser.read()

    try:
        v = msg.split(',')
        v = [int(item) for item in v]
    except Exception as e:
        print(e)
        v = None

    return v, msg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse args')
    parser.add_argument('-p', '--port', help='Serial port', default='/dev/ttyACM0') # COM4
    parser.add_argument('-r', '--rate', help='Baud rate', default=115200, type=int)
    parser.add_argument('-s', '--start', help='Start marker', default=60, type=int)
    parser.add_argument('-e', '--end', help='End marker', default=62, type=int)
    parser.add_argument('-n', '--nvectors', help='Number of vectors to record', default=1000, type=int)
    parser.add_argument('-f', '--fpath', help='File path', default='data/silence.csv')
    args = parser.parse_args()

    sleep(5)

    ser = serial.Serial(args.port, args.rate)
    data = []

    n = 0
    while n < args.nvectors:
        x, msg = receive_vector(args.start, args.end)
        if x is not None:
            print(n, x)
            data.append(x)
            n += 1
        else:
            print(msg)
    ser.close()

    X = np.array(data)
    pandas.DataFrame(X).to_csv(args.fpath, header=None, index=None)


