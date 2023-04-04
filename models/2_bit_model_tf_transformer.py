#import tensorflow_datasets as tfds
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
        data[i, 0, 0] = raw_data[i][0] % 2**20
        data[i, 0, 1] = raw_data[i][1]
        data[i, 0, 2] = raw_data[i][2] % 2**20
        #data[i, 0, 3] = float(raw_data[i][3])

        if (raw_data[i][3] == 1): 
            hot_ones[i, 0] = np.array([1, 0], dtype=np.double)
        else:
            hot_ones[i, 0] = np.array([0, 1], dtype=np.double)

    data[:, :, 0] = (data[:, :, 0] * 2 ) / float(2**20 - 1) - 1
    data[:, :, 1] = (data[:, :, 1] * 2 ) / float(7    ) - 1 
    data[:, :, 2] = (data[:, :, 2] * 2 ) / float(2**20 - 1) - 1

    x_out = data[:, :, 0:3]
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

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)


# Used to encode/embed positional data into each input - check out applied deep learning week 8 lecture
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_dims):
    super().__init__()
    self.d_dims = d_dims
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_dims, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_dims)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_dims, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
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
        self.last_attn_scores = attn_scores
        
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
    #    super().__init__()
    
    def call(self, x):
        # "Feed Forward" on the attention. Here we take information from the entire input to the encoder
        # NOTE: we also use the "user_causal_mask", this makes it so that the decoder only uses information 
        # from inputs that were inputted before the current on i.e. that caused the current input.
        attention_output = self.mha(query=x,
                                    key=x,
                                    value=x,
                                    use_causal_mask=True
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
        x = self.self_attention(x)
        x = self.ff(x)

        return x

class Encoder(layers.Layer):
    def __init__(self, *, num_layers, d_dims, num_heads, ff_fc, vocab_size=None, dropout_rate=0.1):
        super().__init__()

        self.d_dims     = d_dims
        self.num_layers = num_layers
        
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                d_model=d_model
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
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x)
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
    def __init__(self, *, num_layers, d_dims, num_heads, ff_fc, vocab_size=None, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_dims = d_dims
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                d_model=d_model
        )

        self.dropout = layers.Dropout(dropout_rate)

        self.dec_layer = [
            DecoderLayer(d_dims=d_dims,
                        num_heads=num_heads,
                        ff_fc=ff_fc,
                        dropout_rate=dropout_rate
            ) for _ in range(self.num_layers)
        ]

        self.last_attn_scores = None
    

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layer[i](x, context)

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
                            dropout_rate=dropout_rate)
        
        self.final = layers.Dense(target_size)
    

    def call(self, inputs):
        context, x = inputs

        context = self.encoder(context)
        x = self.decoder(x, context)

        logits = self.final(x)


        # SOME TF thing - go and understand it!!!
        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


###################################################################################################

x_train, y_train = read_data(sys.argv[1])

num_layers = 4
d_dims = 128
ff_fc = 512
num_heads = 8
dropout_rate = 0.1
BATCH_SIZE = 10

for i in range(Np):
    print(x_train[i], y_train[i])


'''
transformer = Transformer(
    num_layers=num_layers,
    d_dims=d_dims,
    num_heads=num_heads,
    ff_fc=ff_fc,
    input_vocab_size=BATCH_SIZE,
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)

#transformer.build(input_shape=(1, 3))
transformer((np.zeros(x_train.shape), x_train))
transformer.summary()
# usage transformer((context, inputs))
'''