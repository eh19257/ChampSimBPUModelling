import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import struct
import sys

bp_model_packet = np.dtype([
        ("ip", '<u8'),
        ("branch_type", '<u8'),
        ("branch_addr", '<u8'),
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
  
  data = np.zeros((Np, 1, 4), dtype=np.double )
  hot_ones = np.zeros((Np, 1, 2), dtype=np.double)

  # Convert the tuple array into something usable
  for i in range(Np):
    
    data[i, 0, 0] = raw_data[i][0]# / float(2**8 - 1)
    data[i, 0, 1] = raw_data[i][1]
    data[i, 0, 2] = raw_data[i][2]# / float(2**8 - 1)
    #data[i, 0, 3] = raw_data[i][3]

    '''
    data[i, 0, 0] = (float(raw_data[i][0] * 2) / float(2**16 - 1)) - 1
    data[i, 0, 1] = (float(raw_data[i][1] * 2) / float(7)) - 1
    data[i, 0, 2] = (float(raw_data[i][2] * 2) / float(2**16 - 1)) - 1
    #data[i, 0, 3] = float(raw_data[i][3])
    '''
    if (raw_data[i][3] == 1): 
      hot_ones[i, 0] = np.array([1, 0], dtype=np.double)
    else:
      hot_ones[i, 0] = np.array([0, 1], dtype=np.double)
  
  data[:, :, 0] = ( ( data[:, :, 0] % 2**20 ) * 2 ) / float( 2**20 - 1) - 1
  data[:, :, 1] = ( ( data[:, :, 1] % 2**20 ) * 2 ) / float( 7        ) - 1 
  data[:, :, 2] = ( ( data[:, :, 2] % 2**20 ) * 2 ) / float( 2**20 - 1) - 1
    

  x_out = data[:, :, 0:3]
  y_out = hot_ones#data[:, :, 3]#.reshape((Np, 1, 1))


  return x_out, y_out
  #return data[:, :, 0:3], data[:, :, 3].reshape((Np, 1, 1))
   


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        '''
          self.input_layer = layers.Input(input_shape)
          # Expansion FC layer
          self.expan_fc1 = layers.Dense(units=16,
                                        activation=ACT_FUNC)
          #self.expan_fc1.build(input_shape=(1, 3))

          self.reshape1 = layers.Reshape((16, 1),
                                        input_shape=(1, 16))

          # Conv layer 1 - consists of 32x (1x16) kernels to form an output of the shape (1, 32, 16)        
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
          
          # Reshape - (1, 64, 4) --> (1, 1, 256)
          self.reshape2 = layers.Reshape((1, 256),
                                        input_shape=(1, 4, 64))
          
          # Compression FC layer 1 - compresses the input from(1, 1, 256) down to (1, 1, 16)
          self.comp_fc1 = layers.Dense(units=16,
                                      activation=ACT_FUNC)
          
          # Compression FC layer 2 - compresses the input from(1, 1, 16) down to (1, 1, 1)
          self.comp_fc2 = layers.Dense( units=1,
                                        activation=ACT_FUNC)
        '''
        self.expan_fc1 = layers.Dense(units=16,
                                      activation=ACT_FUNC)

        self.reshape1 = layers.Reshape((16, 1),
                                        input_shape=(1, 16))

        # Conv layer 1 - consists of 32x (1x16) kernels to form an output of the shape (1, 32, 16)        
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
        
        # Reshape - (1, 64, 4) --> (1, 1, 256)
        self.reshape2 = layers.Reshape((1, 256),
                                      input_shape=(1, 4, 64))

        self.rnn = layers.SimpleRNN(256,
                                    input_shape=(256, 1),
                                    activation=ACT_FUNC)

        # Compression FC layer 1 - compresses the input from(1, 1, 256) down to (1, 1, 16)
        #self.comp_fc1 = layers.Dense(units=256,
        #                            activation=ACT_FUNC)
        
        # Compression FC layer 2 - compresses the input from(1, 1, 16) down to (1, 1, 1)
        self.comp_fc2 = layers.Dense(units=16,
                                    activation=ACT_FUNC)

        # Compression FC layer 2 - compresses the input from(1, 1, 16) down to (1, 1, 1)
        self.comp_fc3 = layers.Dense(units=2,
                                    activation=tf.keras.activations.softmax)
        '''
          self.fc1 = layers.Dense(units=16,
                                  activation=ACT_FUNC)

          self.fc2 = layers.Dense(units=32,
                                  activation=ACT_FUNC)
          
          self.fc3 = layers.Dense(units=16,
                                  activation=ACT_FUNC)
          
          self.fc4 = layers.Dense(units=2,
                                  activation=ACT_FUNC)
        '''

    # The forward pass
    def call(self, x):
      print("##### - Starting forward pass... - #####"); print("input", x.shape)
      '''
        x = self.expan_fc1(x); print("expan_fc1", x.shape)
        x = self.reshape1(x); print("reshape1", x.shape)
        
        x = self.conv1(x); print("conv1", x.shape)        
        x = self.pool1(x); print("pool1", x.shape)
        
        x = self.conv2(x); print("conv2", x.shape)
        x = self.pool2(x); print("pool2", x.shape)

        x = self.reshape2(x); print("reshape2", x.shape)      

        x = self.comp_fc1(x); print("comp_fc1", x.shape)
        x = self.comp_fc2(x); print("comp_fc2", x.shape, x)
      '''
      
      x = self.expan_fc1(x); print("expan_fc1", x.shape)
      x = self.reshape1(x); print("reshape1", x.shape)
      
      x = self.conv1(x); print("conv1", x.shape)        
      x = self.pool1(x); print("pool1", x.shape)
      
      x = self.conv2(x); print("conv2", x.shape)
      x = self.pool2(x); print("pool2", x.shape)

      x = self.reshape2(x); print("reshape2", x.shape)

      x = self.rnn(x); print("rnn", x.shape)

      #x = self.comp_fc1(x); print("comp_fc1", x.shape)
      x = self.comp_fc2(x); print("comp_fc2", x.shape)
      x = self.comp_fc3(x); print("comp_fc3", x.shape)
      '''
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
      '''
      #x = tf.keras.activations.softmax(x)
      print("##### - Ending forward pass... - #####")
      return x


###################################################################################################

x_train, y_train = read_data(sys.argv[1])

#x_test, y_test   = read_data(sys.argv[2])

'''
print("shape", x_train.shape)
print("Single output", x_train[0].shape)
print("shape", y_train.shape)
print("Single output", y_train[0].shape)
'''
#for i in range(Np):
  #print(x_train[i][0], y_train[i][0])

model = Model()
model.compile(
    # By default, fit() uses tf.function().  You can
    # turn that off for debugging, but it is on now.
    run_eagerly=False,

    # Using a built-in optimizer, configuring as an object
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),

    metrics=["accuracy"],
    # Keras comes with built-in MSE error
    # However, you could use the loss function
    # defined above

    #loss=tf.keras.losses.mean_squared_error,
    loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
    #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
)

model.build(input_shape=(1, 3))
model.summary()

model.fit(x_train,
          y_train, 
          epochs=10,
          batch_size=1000,
          shuffle=False
          #validation_data=(x_test, y_test)
)


branch_test =  np.array([[[140084140304547 * 2/(2**64 - 1), 1 * 2/7, 140084140304566 * 2/(2**64 - 1)]]], dtype=np.double) - 1
non_branch_test =  np.array([[[140084140304600 * 2/(2**64 - 1), 0 , 0 * 2/(2**64 - 1)]]], dtype=np.double) - 1

print("Shape of branch test", branch_test.shape)
print("\nThe BRANCH test for {2} was: {0}\nThe NO_BRANCH test for {3} was: {1}.".format(model.predict(branch_test), model.predict(non_branch_test), branch_test, non_branch_test))

