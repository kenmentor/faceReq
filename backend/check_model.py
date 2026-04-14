import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)
        self.axis = axis
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)
    def call(self, inputs):
        a, b = inputs
        return tf.math.abs(a - b)
    def get_config(self):
        return super().get_config()

model = tf.keras.models.load_model('models/model.h5', custom_objects={"L2Normalize": L2Normalize, "L1Dist": L1Dist}, compile=False)

print("\nLooking for embedding layers...")
for i, l in enumerate(model.layers):
    try:
        print(f"{i}: {l.name} -> {l.output_shape}")
    except:
        print(f"{i}: {l.name} -> error")

# Try to find the embedding layer (before l1_dist or after functional)
for l in model.layers:
    if 'dense' in l.name.lower() or 'l2' in l.name.lower() or 'embedding' in l.name.lower():
        print(f"\nFound: {l.name} -> {l.output_shape}")
