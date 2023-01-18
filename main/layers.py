#
# import tensorflow as tf
#
# class EncoderLSTM(tf.keras.layers.Layer):
#     def __init__(self, embedding_size, hidden_size, dropout_ratio, embeddings, bidirectional=False, num_layers=1):
#         super(EncoderLSTM, self).__init__()
#         self.embedding_size = embedding_size
#         self.hidden_size = hidden_size
#         self.dropout = tf.keras.layers.Dropout(rate=dropout_ratio)
#         self.num_directions = 2 if bidirectional else 1
#         self.num_layers = num_layers
#         self.embedding = embeddings
#         if bidirectional:
#             self.lstm = tf.keras.layers.Bidirectional(
#                 tf.keras.layers.LSTM(units=self.hidden_size)
#             )
#         else:
#             self.lstm = tf.keras.layers.LSTM(units=self.hidden_size)
#         self.encoder2decoder = tf.keras.layers.Dense(units=hidden_size*self.num_directions, activation="tanh")
#
#     def init_state(self, inputs):
#         batch_size = tf.shape(inputs)[0]
#         h0 = tf.Variable(tf.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size))
#         c0 = tf.Variable(tf.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size))
#         return h0, c0
#
#     def call(self, inputs):
#         embeds = self.embedding(inputs)  # (batch_size, seq_len, embedding_size)
#         embeds = self.dropout(embeds)
#         h0, c0 = self.init_state(inputs)
#
#         enc_h, (enc_c_t, enc_h_t) = self.lstm(embeds, initial_state=[h0, c0])
#
#         if self.num_directions == 2:
#             c_t = tf.concat([enc_c_t[-1], enc_c_t[-2]], axis=1)
#             h_t = tf.concat([enc_h_t[-1], enc_h_t[-2]], axis=1)
#         else:
#             c_t = enc_c_t[-1]
#             h_t = enc_h_t[-1]  # (batch_size, hidden_size)
#
#         decoder_init = self.encoder2decoder(h_t)
#
#         ctx = self.dropout(enc_h)
#
#         return ctx, decoder_init, c_t # (batch_size, seq_len, hidden_size*num_directions)
#                                     # (batch_size, hidden_size)
