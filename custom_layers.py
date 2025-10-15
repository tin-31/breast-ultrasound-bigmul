import tensorflow as tf

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(channels // reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, x):
        se = self.avg_pool(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = tf.reshape(se, (-1, 1, 1, x.shape[-1]))
        return x * se

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')

    def call(self, x):
        avg = tf.reduce_mean(x, axis=3, keepdims=True)
        max = tf.reduce_max(x, axis=3, keepdims=True)
        concat = tf.concat([avg, max], axis=3)
        return x * self.conv(concat)

class CBAM(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.se = SEBlock(channels)
        self.sa = SpatialAttention()

    def call(self, x):
        x = self.se(x)
        return self.sa(x)

class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(AttentionGate, self).__init__()
        self.W_g = tf.keras.layers.Conv2D(filters, 1, padding='same')
        self.W_x = tf.keras.layers.Conv2D(filters, 1, padding='same')
        self.psi = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')

    def call(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        att = tf.nn.relu(g1 + x1)
        return x * self.psi(att)
