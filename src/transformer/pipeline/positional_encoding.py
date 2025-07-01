import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.pos_encoding = self._calculate_positional_encoding(max_sequence_length, d_model)

    def _calculate_positional_encoding(self, max_sequence_length, d_model):
        positions = tf.range(max_sequence_length, dtype=tf.float32)[:, tf.newaxis]
        div_term_exponent = tf.range(0, d_model, 2, dtype=tf.float32) / d_model
        div_term = tf.pow(10000.0, div_term_exponent)
        angle_rads = positions / div_term

        sine_vals = tf.math.sin(angle_rads)
        cosine_vals = tf.math.cos(angle_rads)
        pos_encoding_pairs = tf.stack([sine_vals, cosine_vals], axis=-1) #(max_len, d_model/2, 2)
        pos_encoding_matrix = tf.reshape(pos_encoding_pairs, (max_sequence_length, d_model)) #(max_len, d_model)

        pos_encoding_matrix = pos_encoding_matrix[tf.newaxis, :, :] #(1, max_len, d_model)
        return tf.cast(pos_encoding_matrix, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "max_sequence_length": self.max_sequence_length,
            "d_model": self.d_model,
        })
        return config
