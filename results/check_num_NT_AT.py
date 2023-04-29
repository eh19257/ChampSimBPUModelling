#import tensorflow_datasets as tfds
import numpy as np
import struct
import sys

bp_model_packet = np.dtype([
        ("ip", '<u8'),
        ("branch_type", '<u8'),
        ("branch_addr", '<u8'),
        ("branch_prediciton", '<u8'),
        ("actual_branch_behaviour", '<u8')
    ])

SIZE_OF_PACKET = 8 * len(bp_model_packet)


def read_data(filename):
    print("Loading and normalising data from the file: \"{0}\"...".format(filename))

    with open(filename, "rb") as file:
        b = file.read()

    raw_data = np.frombuffer(b, dtype=bp_model_packet)

    global Np
    Np = len(raw_data)
    #Np = 1000

    data = np.zeros((Np, 1, 4), dtype=np.double )
    hot_ones = np.zeros((Np, 1, 2), dtype=np.double)

    # Convert the tuple array into something usable
    for i in range(Np):
        data[i, 0, 0] = float(raw_data[i][0] % 2**16)
        data[i, 0, 1] = float(raw_data[i][1])
        data[i, 0, 2] = float(raw_data[i][2] % 2**16)
        data[i, 0, 3] = float(raw_data[i][4])

        if (raw_data[i][3] == 1): 
            hot_ones[i, 0] = np.array([1, 0], dtype=np.double)
        else:
            hot_ones[i, 0] = np.array([0, 1], dtype=np.double)

    # Normalising the data
    data[:, :, 0] = (data[:, :, 0] ) / float(2**16 - 1) #- 1
    data[:, :, 1] = (data[:, :, 1] ) / float(7        ) #- 1 
    data[:, :, 2] = (data[:, :, 2] ) / float(2**16 - 1) #- 1
    #data[:, :, 3] = (data[:, :, 3] ) / float(1        )

    x_out = data#[:, :, 0:3]
    y_out = hot_ones#data[:, :, 3]#.reshape((Np, 1, 1))

    #dataset = tf.data.Dataset.from_tensor_slices((x_out, y_out))

    return x_out, y_out


x, y = read_data(sys.argv[1])


num_t  = 0
num_nt = 0

for i in range(len(x)):
    print(x[i, 0], y[i])
    if x[i, 0, 3] == 1 : num_t  += 1
    else               : num_nt += 1

print("The total number of instructions was {0}.".format(len(x)))
print("TAKEN: {0}. NOT TAKEN: {1}. There were {2} more taken than not taken".format(num_t, num_nt, num_t/num_nt))