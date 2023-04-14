#import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import struct
import sys

# Use GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

bp_model_packet = np.dtype([
        ("ip", '<u8'),
        ("branch_type", '<u8'),
        ("branch_addr", '<u8'),
        ("actual_branch_behaviour", '<u8'),
        ("branch_prediciton", '<u8')
    ])

SIZE_OF_PACKET = 8 * len(bp_model_packet)
ACT_FUNC = "relu"


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
        data[i, 0, 0] = float(raw_data[i][0] % 2**20)
        data[i, 0, 1] = float(raw_data[i][1])
        data[i, 0, 2] = float(raw_data[i][2] % 2**20)
        data[i, 0, 3] = float(raw_data[i][3])

        if (raw_data[i][4] == 1): 
            hot_ones[i, 0] = np.array([1, 0], dtype=np.double)
        else:
            hot_ones[i, 0] = np.array([0, 1], dtype=np.double)

    # Normalising the data
    data[:, :, 0] = (data[:, :, 0] ) / float(2**20 - 1) #- 1
    data[:, :, 1] = (data[:, :, 1] ) / float(7        ) #- 1 
    data[:, :, 2] = (data[:, :, 2] ) / float(2**20 - 1) #- 1
    #data[:, :, 3] = (data[:, :, 3] ) / float(1        )

    x_out = data#[:, :, 0:3]
    y_out = hot_ones#data[:, :, 3]#.reshape((Np, 1, 1))

    #dataset = tf.data.Dataset.from_tensor_slices((x_out, y_out))

    return x_out, y_out
    #return data[:, :, 0:3], data[:, :, 3].reshape((Np, 1, 1))



def make_batches(x, y, h=128):
    print("Formatting data into a history-table like structure...")
    
    Np = len(x)

    enc   = np.zeros( ( Np - h, h, 4), dtype=np.double)
    dec   = np.zeros( ( Np - h, 1, 4), dtype=np.double)

    y_out = np.zeros( ( Np - h, 1, 2), dtype=np.double)#, h, 1, 1), dtype=np.double )
    
    for i in range(0, Np - h):
        enc[i] = x[ i     : i + h     ].reshape((h, 4))
        dec[i] = x[ i + h : i + h + 1 ].reshape((1, 4))

        # Edit the actual_branch_behaviour of dec so that it's 0.5 (i.e. a probability that could be either taken or not taken)  
        dec[i][0][3] = 0.5
        #print("enc", enc.shape)
        #print("dec", dec.shape)

        #x_out[ i ] = [enc, dec]
        y_out[i] = y[ i ].reshape((1, 2))

        #xs.append( (enc, dec) )
        #ys.append( (y_out) )

        #ds.append( ((enc, dec), y_out) )
    
    #ds = tf.data.Dataset.from_tensor_slices(ds)
    return (enc, dec), y_out
    #return tf.data.Dataset.from_tensor_slices(xs), tf.data.Dataset.from_tensor_slices(ys)#(tf.data.Dataset(enc), tf.data.Dataset(dec.)), tf.data.Dataset(y_out)

num_layers = 4
d_dims = 128
ff_fc = 128
num_heads = 4
dropout_rate = 0.1
HISTORY_TABLE_SIZE = 128

#BATCH_SIZE = 10
BUFFER_SIZE = 1000 # The number of elements and NOT the number of bytes for the buffer



transformer = Transformer(
    num_layers=num_layers,
    d_dims=d_dims,
    num_heads=num_heads,
    ff_fc=ff_fc,
    input_vocab_size=HISTORY_TABLE_SIZE,
    target_vocab_size=2,
    dropout_rate=dropout_rate)

'''
transformer.compile(
    #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none'),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #optimizer=tf.keras.optimizers.SGD(learning_rate= CustomSchedule(d_dims=d_dims) ),
    optimizer=tf.keras.optimizers.Adam(learning_rate= CustomSchedule(d_dims=d_dims) ),
    metrics=["accuracy"]
)
'''
x_train_raw, y_train_raw = read_data(sys.argv[1])
x_test_raw , y_test_raw  = read_data(sys.argv[2])

print("Size of x_train_raw:", x_train_raw.shape)

x_train, y_train = make_batches(x_train_raw, y_train_raw)
x_test,  y_test  = make_batches(x_test_raw,  y_test_raw )


transformer.load_weights(sys.argv[1])
