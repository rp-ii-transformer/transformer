import tensorflow as tf

class PositionWiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, **kwargs):
        super(PositionWiseFeedForwardNetwork, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff

        self.dense_1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x_ff = self.dense_1(x)
        output = self.dense_2(x_ff)
        return output

    def get_config(self):
        config = super(PositionWiseFeedForwardNetwork, self).get_config()
        config.update({
            "d_model": self.d_model,
            "d_ff": self.d_ff,
        })
        return config