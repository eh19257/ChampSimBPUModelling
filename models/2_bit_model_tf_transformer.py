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


# Applies the positional encoding of the input into the data
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)
  #print("angle_rads:", angle_rads.shape)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  #print("pos_encoding.shape:", pos_encoding.shape)
  return tf.cast(pos_encoding, dtype=tf.float32)


# Used to encode/embed positional data into each input - check out applied deep learning week 8 lecture
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_dims, start=0):
    super().__init__()
    self.d_dims = d_dims
    self.start = start
    self.vocab_size = vocab_size
    #self.embedding = tf.keras.layers.Embedding(vocab_size, d_dims, mask_zero=True)
    self.expand = layers.Dense(d_dims)
    self.pos_encoding = positional_encoding(length=2048, depth=d_dims)#depth=4)

  #def compute_mask(self, *args, **kwargs):
    #return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = self.vocab_size#tf.shape(x)[1]
    #x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_dims, tf.float32))
    #print("X SHAPE:", x.shape)
    #print("POSENCODING SHAPE:", self.pos_encoding[tf.newaxis, :length, :].shape)
    #print("POSENCODING:", self.pos_encoding.shape)
    #foo = self.pos_encoding[tf.newaxis, :length, :]
    #print("POSENCODING PART:", foo)
    x = self.expand(x)
    #x = x + self.pos_encoding[tf.newaxis, :length, :]
    #tf.print("LENGTH:", length)

    x = x + self.pos_encoding[tf.newaxis, self.start : self.start + length, :]
    return x



# The base attention layer
class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.add = layers.Add()
        self.norm = layers.LayerNormalization()        


# Cross attention component of the Transformer - takes in the context from the encoder and is found in the decoder
class CrossAttention(BaseAttention):
    #def __init__(self):
    #    super().__init__()
    
    def call(self, x, context):
        # "Forward feed" on the attention. In order to make it cross attention the key and value are = to the context
        attention_output, attention_scores = self.mha(query=x,
                                                    key=context,
                                                    value=context,
                                                    return_attention_scores=True
        )
        self.last_attn_scores = attention_scores
        
        # Add & Norm Layers
        x = self.add([x, attention_output])
        x = self.norm(x)
        return x


# Found at the input side of the encoder, here we extract information from the input and take in information from all the inputs
class GlobalSelfAttention(BaseAttention):
    #def __init__(self):
    #    super.__init__()
    
    def call(self, x):
        # "Feed Forward" on the attention. Here we take information from the entire input to the encoder
        attention_output = self.mha(query=x,
                                    key=x,
                                    value=x
        )
        
        # Add & Norm Layers
        x = self.add([x, attention_output])
        x = self.norm(x)

        return x


# Found at the input side of he decoder, takes in information from previous decoder inputs
class CausalSelfAttention(BaseAttention):
    #def __init__(self):
    #    super(self).__init__()
    
    def call(self, x):
        # "Feed Forward" on the attention. Here we take information from the entire input to the encoder
        # NOTE: we also use the "user_causal_mask", this makes it so that the decoder only uses information 
        # from inputs that were inputted before the current on i.e. that caused the current input.
        attention_output = self.mha(query=x,
                                    key=x,
                                    value=x
                                    #use_causal_mask = True
        )
        
        # Add & Norm Layers
        x = self.add([x, attention_output])
        x = self.norm(x)

        return x


# Here we have the FC/Dense part of the encoder/decoder - this part is for classification and there is NO point of doing any feature 
# extraction here, this is becuase the attention layers do this way better than a standard CNN
class FeedForward(layers.Layer):
    def __init__(self, fc_1_units, fc_2_units, dropout_rate=0.1):
        super().__init__()
        
        #self.fc_1 = layers.Dense(fc_1_units, activation=ACT_FUNC)
        #self.fc_2 = layers.Dense(fc_2_nits, )
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(fc_2_units, activation=ACT_FUNC),
            tf.keras.layers.Dense(fc_1_units),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        
        self.add = layers.Add()
        self.norm = layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.norm(x)

        return x


class EncoderLayer(layers.Layer):
    def __init__(self, *, d_dims, num_heads, ff_fc, dropout_rate):
        super().__init__()
        
        self.self_attention = GlobalSelfAttention(num_heads=num_heads,
                                                key_dim=d_dims,
                                                dropout=dropout_rate
        )

        self.ff = FeedForward(d_dims,
                            ff_fc
        )
    
    def call(self, x):
        x = self.self_attention(x); print("POST SELF ATTENTION:", x.shape)
        x = self.ff(x); print("POST FF:", x.shape)

        return x

class Encoder(layers.Layer):
    def __init__(self, *, num_layers, d_dims, num_heads, ff_fc, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_dims     = d_dims
        self.num_layers = num_layers
        
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                d_dims=d_dims
        )

        self.dropout = layers.Dropout(dropout_rate)

        self.enc_layers = [
            EncoderLayer(d_dims=self.d_dims,
                        num_heads=num_heads,
                        ff_fc=ff_fc,
                        dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ]

    def call(self, x):
        x = self.pos_embedding(x); print("POST POS_EMBEDDING:", x.shape)
        x = self.dropout(x); print("POST DROPOUT:", x.shape)

        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x); print("POST ENC_LAYERS:", x.shape)
        return x


class DecoderLayer(layers.Layer):
    def __init__(self, *, d_dims, num_heads, ff_fc, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads,
                                                        key_dim=d_dims,
                                                        dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(num_heads=num_heads,
                                            key_dim=d_dims,
                                            dropout=dropout_rate
        )

        self.ff = FeedForward(d_dims,
                            ff_fc
        )

        
    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ff(x)
        return x

class Decoder(layers.Layer):
    def __init__(self, *, num_layers, d_dims, num_heads, ff_fc, vocab_size, dropout_rate=0.1, enc_vocab_size=0):
        super(Decoder, self).__init__()

        self.d_dims = d_dims
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                d_dims=d_dims,
                                                start=enc_vocab_size
        )

        self.dropout = layers.Dropout(dropout_rate)

        self.dec_layers = [
            DecoderLayer(d_dims=d_dims,
                        num_heads=num_heads,
                        ff_fc=ff_fc,
                        dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ]

        self.last_attn_scores = None
    

    def call(self, x, context):
        x = self.pos_embedding(x); print("DEC POST POS_EMBED:", x.shape)
        x = self.dropout(x); print("DEC POST DROPOUT:", x.shape)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context); print("DEC POST DEC_LAYERS:", x.shape)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        
        return x
    

class Transformer(keras.Model):
    def __init__(self, *, num_layers, d_dims, num_heads, ff_fc, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers=num_layers,
                            d_dims=d_dims,
                            num_heads=num_heads,
                            ff_fc=ff_fc,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate)
        
        self.decoder = Decoder(num_layers=num_layers,
                            d_dims=d_dims,
                            num_heads=num_heads,
                            ff_fc=ff_fc,
                            vocab_size=target_vocab_size,
                            dropout_rate=dropout_rate,
                            enc_vocab_size=input_vocab_size)
        
        self.reshape = layers.Reshape((None, 1, d_dims))
                                      #input_shape=(1, 1, 4, 256))

        self.final_fc_1 = layers.Dense(int(d_dims / 32))

        self.final_fc_2 = layers.Dense(2)

        self.softmax = layers.Softmax()
    

    def call(self, inputs):
        context, x = inputs
        #tf.print("Inputs:", inputs)
        #tf.print(context)
        print("CONTEXT:", context.shape); print("DECODER:", x.shape)
        context = self.encoder(context); print("POST ENCODER:", context.shape)

        x = self.decoder(x, context); #print("POST DECODER:", x)

        #logits = self.reshape(x)

        print("PENULTIMATE SHAPE:", x.shape)

        logits = self.final_fc_1(x)

        print("ULTIMATE SHAPE:", logits.shape)

        logits = self.final_fc_2(logits)

        print("ULTIMATE ULTIMATE SHAPE:", logits.shape)


        # SOME TF thing - go and understand it!!!
        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        probs = self.softmax(logits)
        # Return the final output and the attention weights.
        return probs


###################################################################################################

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_dims, warmup_steps=4000):
        super().__init__()

        self.d_dims = d_dims
        self.d_dims = tf.cast(self.d_dims, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_dims) * tf.math.minimum(arg1, arg2) * 0.1


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


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


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, h):
        self.x = x_set
        self.y = y_set

        self.Np = len(x_set) - h - 1
        self.batch_size = batch_size

        self.h = h
        #self.dec_dims = dec_dims

    def __len__(self):
        return int(np.floor( self.Np / float(self.batch_size) ) )
        #return int(np.floor( (len(self.x) - (self.h + 1) ) / float(self.batch_size)))


    # This should return a single batch
    def __getitem__(self, idx):
        '''
        #tf.print("look at self.x:", self.x)
        enc = self.x[ 0 ][ idx * self.batch_size : (idx + 1) * self.batch_size ]#.reshape((self.batch_size, ))
        dec = self.x[ 1 ][ idx * self.batch_size : (idx + 1) * self.batch_size ]

        label = self.y[ idx * self.batch_size : (idx + 1) * self.batch_size ]
        #batch_x = (self.x[0][idx * self.batch_size : (idx + 1) * self.batch_size], self.x[1][idx * self.batch_size : (idx + 1) * self.batch_size])
        #batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        enc = enc.reshape( (self.batch_size, self.enc_dims, 4) )
        dec = dec.reshape( (self.batch_size, self.dec_dims, 4) )

        label = label.reshape( (self.batch_size, 1, 2) )
        '''

        x = self.__gen_x(idx)
        y = self.__gen_y(idx)
    
        return x, y


    def __gen_x(self, idx):
        enc   = np.zeros( ( self.batch_size, self.h, 4), dtype=np.double)
        dec   = np.zeros( ( self.batch_size, 1     , 4), dtype=np.double)

        for i in range(self.batch_size):
            enc[i] = self.x[ idx * self.batch_size + i          : idx * self.batch_size + i + self.h    ].reshape( (self.h, 4) )
            enc[i] = self.x[ idx * self.batch_size + i + self.h : idx * self.batch_size + i + self.h + 1].reshape( (1     , 4) )

            dec[i][0][3] = 0.5
      
        return (enc, dec)


    def __gen_y(self, idx):
        y_out = np.zeros( ( self.batch_size, 1, 2), dtype=np.double)

        for i in range(self.batch_size):
            y_out[i] = self.y[ idx * self.batch_size + i + self.h : idx * self.batch_size + i + self.h + 1 ].reshape((1, 2))
        
        return y_out


###################################################################################################

num_layers = 4
d_dims = 64#128
ff_fc = 64#128
num_heads = 4
dropout_rate = 0.1
HISTORY_TABLE_SIZE = 128

BATCH_SIZE = 128
#BUFFER_SIZE = 1000 # The number of elements and NOT the number of bytes for the buffer


transformer = Transformer(
    num_layers=num_layers,
    d_dims=d_dims,
    num_heads=num_heads,
    ff_fc=ff_fc,
    input_vocab_size=HISTORY_TABLE_SIZE,
    target_vocab_size=1,
    dropout_rate=dropout_rate)


transformer.compile(
    #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none'),
    #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    #optimizer=tf.keras.optimizers.SGD(learning_rate= CustomSchedule(d_dims=d_dims) ),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),#CustomSchedule(d_dims=d_dims) ),
    metrics=["accuracy"]
)

x_train_raw, y_train_raw = read_data(sys.argv[1])
#x_test_raw , y_test_raw  = read_data(sys.argv[2])

print("Size of x_train_raw:", x_train_raw.shape)

#x_train, y_train = make_batches(x_train_raw, y_train_raw, h=HISTORY_TABLE_SIZE)
#x_test,  y_test  = make_batches(x_test_raw,  y_test_raw, h=HISTORY_TABLE_SIZE)

print("Data is being converted into a data generator...")

#train = tf.data.Dataset.from_generator(DataGenerator(x_train, y_train, BATCH_SIZE))
train = DataGenerator(x_train_raw, y_train_raw, BATCH_SIZE, HISTORY_TABLE_SIZE)
#test  = DataGenerator(x_test, y_test, BATCH_SIZE)


#print("Example Line:", train.__getitem__(1))
(foo, bar), baz = train.__getitem__(1)
print("enc:", foo.shape, ". Dec:", bar.shape, ". Labels:", baz.shape, ".")


#for i in train: print("enc:", i[0][0].shape, ". Dec:", i[0][1].shape, ". Labels:", i[1].shape, ".")
#transformer(train.__getitem__(0)[0])
#transformer.summary()

print("STARTING TRAINING!!!")
transformer.fit(
    train,
    #x=x_train,
    #y=y_train, 
    epochs=10,
    #batch_size=BATCH_SIZE,
    shuffle=True
    #validation_data=test
    #verbose='auto'
)

transformer.save(sys.argv[3])