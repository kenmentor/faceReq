import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
import numpy as np


class L2Normalize(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
    
    def get_config(self):
        return super().get_config()


class L1Dist(layers.Layer):
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)
    
    def call(self, inputs):
        a, b = inputs
        return tf.math.abs(a - b)
    
    def get_config(self):
        return super().get_config()


class CosineSimilarity(layers.Layer):
    def __init__(self, **kwargs):
        super(CosineSimilarity, self).__init__(**kwargs)
    
    def call(self, inputs):
        a, b = inputs
        normalize_a = tf.nn.l2_normalize(a, axis=-1)
        normalize_b = tf.nn.l2_normalize(b, axis=-1)
        cosine = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1, keepdims=True)
        return cosine
    
    def get_config(self):
        return super().get_config()


def residual_block(x, filters, kernel_size=3):
    """Residual block with skip connection"""
    shortcut = x
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x


def create_embedding_network(input_shape=(100, 100, 3), embedding_dim=256, use_pretrained=True, freeze_pretrained=True):
    """
    Creates an embedding network using pretrained MobileNetV2 backbone.
    """
    inputs = keras.Input(shape=input_shape, name='input')
    
    if use_pretrained:
        mobilenet = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(96, 96, 3),
            pooling='avg'
        )
        
        if freeze_pretrained:
            mobilenet.trainable = False
        
        x = layers.Resizing(96, 96)(inputs)
        x = layers.Conv2D(3, (1, 1))(x)
        
        x = mobilenet(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(embedding_dim, activation=None)(x)
        outputs = L2Normalize(axis=-1)(x)
    else:
        x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = residual_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = residual_block(x, 256)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = residual_block(x, 512)
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(embedding_dim, activation=None)(x)
        outputs = L2Normalize(axis=-1)(x)
    
    return Model(inputs, outputs, name='embedding_network')


def create_siamese_network(input_shape=(100, 100, 3), embedding_dim=256, use_pretrained=False, freeze_pretrained=True):
    """
    Creates a complete Siamese network with improved architecture.
    Uses both L1 distance and cosine similarity for better accuracy.
    """
    embedding_network = create_embedding_network(input_shape, embedding_dim, use_pretrained=use_pretrained, freeze_pretrained=freeze_pretrained)
    
    input_a = keras.Input(shape=input_shape, name='input_a')
    input_b = keras.Input(shape=input_shape, name='input_b')
    
    embedded_a = embedding_network(input_a)
    embedded_b = embedding_network(input_b)
    
    l1_distance = L1Dist()([embedded_a, embedded_b])
    cosine_sim = CosineSimilarity()([embedded_a, embedded_b])
    
    concat = layers.Concatenate()([l1_distance, cosine_sim])
    
    x = layers.Dense(128, activation='relu')(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model([input_a, input_b], outputs, name='siamese_network')
    
    return model


class ContrastiveLoss(keras.losses.Loss):
    def __init__(self, margin=0.3, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        distance = 1 - y_pred
        
        positive_loss = y_true * tf.square(distance)
        negative_loss = (1 - y_true) * tf.square(tf.maximum(self.margin - distance, 0))
        
        return tf.reduce_mean(positive_loss + negative_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin})
        return config


class BinaryCrossEntropyLoss(keras.losses.Loss):
    """Standard binary cross-entropy for direct similarity prediction"""
    def __init__(self, **kwargs):
        super(BinaryCrossEntropyLoss, self).__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        return keras.losses.binary_crossentropy(y_true, y_pred)


def triplet_loss(y_true, y_pred, margin=0.3):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    
    loss = tf.maximum(pos_dist - neg_dist + margin, 0)
    
    return tf.reduce_mean(loss)


def load_pretrained_siamese(model_path, compile_model=True):
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "L2Normalize": L2Normalize,
            "L1Dist": L1Dist,
            "ContrastiveLoss": ContrastiveLoss,
            "BinaryCrossEntropyLoss": BinaryCrossEntropyLoss
        },
        compile=compile_model
    )
    return model


if __name__ == "__main__":
    model = create_siamese_network(input_shape=(100, 100, 3), embedding_dim=256)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=BinaryCrossEntropyLoss(),
        metrics=['accuracy']
    )
    
    print("Siamese Network Architecture:")
    print("=" * 50)
    model.summary()
    
    print("\nEmbedding Network:")
    print("=" * 50)
    embedding_model = create_embedding_network(input_shape=(100, 100, 3), embedding_dim=256)
    embedding_model.summary()
