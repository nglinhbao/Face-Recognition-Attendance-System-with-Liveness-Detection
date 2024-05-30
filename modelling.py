from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Model
from keras import layers
from keras import ops
from tensorflow.keras.applications import resnet
import numpy as np
from face_recognition import face_locations, face_encodings
import cv2
from io import BytesIO
from PIL import Image
import base64
from pose_detection import predFacePoseCV2

# Load the saved Siamese model
embedding = load_model('models/embedding_metric.h5')  # Replace 'embedding_metric.h5' with your actual file name
target_shape = (200, 200)

# Define the PairDistanceLayer and create the pair_model
class PairDistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, image1, image2):
        distance = 1 - ops.sum(tf.square(tf.subtract(image1, image2)), -1)
        return distance


def create_pair_model(embedding, target_shape):
    # Define the input layers
    image1_input = layers.Input(name="image1", shape=target_shape + (3,))
    image2_input = layers.Input(name="image2", shape=target_shape + (3,))

    # Assuming 'embedding' and 'resnet' are defined elsewhere
    embedding1 = embedding(resnet.preprocess_input(image1_input))
    embedding2 = embedding(resnet.preprocess_input(image2_input))

    # Apply the PairDistanceLayer
    similarity_scores = PairDistanceLayer()(
        image1=embedding1,
        image2=embedding2
    )

    # Create the model
    pair_model = Model(inputs=[image1_input, image2_input], outputs=similarity_scores)

    return pair_model