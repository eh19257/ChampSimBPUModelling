import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import struct
import sys

ACT_FUNC = "relu"


bp_model_packet = np.dtype([
        ("ip", 'u8'),
        ("branch_type", 'u8'),
        ("branch_addr", 'u8'),
        ("branch_prediciton", 'u8')
    ])

SIZE_OF_PACKET = 8 * len(bp_model_packet)

def read_data(filename):
  print("Loading in data from the file: {0}...".format(filename))

  with open(filename, "rb") as file:
      b = file.read()
  
  '''
  #print("number of packets {0}".format(len(b)/SIZE_OF_PACKET))

  #num_of_packets = int(len(b) / SIZE_OF_PACKET) - 1

  data = np.zeros(( num_of_packets, 4), dtype=np.uint64 )

  for i in range(0, num_of_packets):
    ip, b_type, b_addr, b_pred = struct.unpack_from("QBQB", b, offset = i * SIZE_OF_PACKET)

    data[i, 0] = ip
    data[i, 1] = b_type
    data[i, 2] = b_addr
    data[i, 3] = b_pred

    print(data[i])
  '''
  
  np_data = np.frombuffer(b, dtype=bp_model_packet)
  
  data = np.zeros((len(np_data), 1, 4), dtype=np.uint64 )

  #print("SHAPE!!!", data.shape, np_data.shape)

  # Convert the tuple array into something usable
  for i in range(len(np_data)):
      for j in range(4):
          data[i, 0, j] = np_data[i][j]

  #data = data.reshape((len(data), 1, 4))

  return data[:, :, 0:3], data[:, :, 3]
   

def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))


def grad(loss_object, model, Xs, Ys):
  with tf.GradientTape() as tape:
    loss_value = loss(loss_object, model, Xs, Ys, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Expansion FC layer
        self.expan_fc1 = layers.Dense(units=16,
                                      activation=ACT_FUNC)
        #self.expan_fc1.build(input_shape=(3, 1))

        self.reshape1 = layers.Reshape((16, 1),
                                      input_shape=(1, 16))

        # Conv layer 1 - consists of 32x (1x16) kernels to form an output of the shape (32, 1, 16)
        
        self.conv1 = layers.Conv1D(32, 
                                  3,
                                  strides=1,
                                  padding="same",
                                  data_format="channels_last",
                                  activation=ACT_FUNC)
        
        # Pooling layer 1 - crushes the input size (1, 32, 16) into (1, 32, 8)
        self.pool1 = layers.MaxPooling1D(pool_size=2,
                                        strides=2,
                                        padding="valid",
                                        data_format="channels_last")
        
        # Conv layer 2 - consists of 64x (32, 3) filters/kernels to form an output of the shape (64, 1, 8)
        self.conv2 = layers.Conv1D(64,
                                  3,
                                  strides=1,
                                  padding="same",
                                  activation=ACT_FUNC,
                                  data_format="channels_last")
        
        # Pooling layer 2 - crushes the input size (1, 64, 8) into (1, 64, 4)
        self.pool2 = layers.MaxPooling1D(pool_size=2,
                                        strides=2,
                                        padding="valid",
                                        data_format="channels_last")
        
        # Reshape - (64, 1, 4) --> (1, 256)
        self.reshape = layers.Reshape((1, 256),
                                      input_shape=(1, 4, 64))
        
        # Compression FC layer 1 - compresses the input from(1, 1, 256) down to (1, 1, 16)
        self.comp_fc1 = layers.Dense(units=16,
                                    activation=ACT_FUNC)
        
        # Compression FC layer 2 - compresses the input from(1, 1, 16) down to (1, 1, 1)
        self.comp_fc2 = layers.Dense( units=1,
                                      activation=ACT_FUNC)
        
        

    # The forward pass
    def call(self, x):
      print("##### - Starting forward pass... - #####")
      print("input", x.shape)
      x = self.expan_fc1(x)
      print("expan_fc1", x.shape)
      x = self.reshape1(x)
      print("reshape1", x.shape)
      x = self.conv1(x)
      print("conv1", x.shape)
        
        
      x = self.pool1(x)
      print("pool1", x.shape)
      
      x = self.conv2(x)
      print("conv2", x.shape)
      
      x = self.pool2(x)
      print("pool2", x.shape)

      x = self.reshape(x)
      print("reshape", x.shape)
      

      x = self.comp_fc1(x)
      print("comp_fc1", x.shape)
      x = self.comp_fc2(x)
      print("comp_fc2", x.shape, x)
      
      print("##### - Ending forward pass... - #####")
      
      return x


###################################################################################################

x_train, y_train = read_data(sys.argv[1])

print("shape", x_train.shape)

model = Model()
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.build(input_shape=(1, 3))
model.summary()

model.compile(
    # By default, fit() uses tf.function().  You can
    # turn that off for debugging, but it is on now.
    run_eagerly=False,

    # Using a built-in optimizer, configuring as an object
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),

    # Keras comes with built-in MSE error
    # However, you could use the loss function
    # defined above
    loss=tf.keras.losses.mean_squared_error,
)

model.fit(x_train, y_train, epochs=10, batch_size=1, shuffle=False)