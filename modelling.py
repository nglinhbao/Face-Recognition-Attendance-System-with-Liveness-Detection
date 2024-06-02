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

# Load the saved Siamese model (assuming this model outputs embeddings)
embedding = load_model('models/embedding_metric.h5') 
target_shape = (200, 200) 

# Define the PairDistanceLayer 
class PairDistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, embedding1, embedding2):
        distance = 1 - ops.sum(tf.square(tf.subtract(embedding1, embedding2)), -1)
        return distance

def create_pair_model(embedding_model, embedding_dim): 
    # Define the input layers - now taking embeddings as input
    embedding1_input = layers.Input(name="embedding1", shape=(embedding_dim,))
    embedding2_input = layers.Input(name="embedding2", shape=(embedding_dim,))

    # Apply the PairDistanceLayer directly on embeddings
    similarity_scores = PairDistanceLayer()(
        embedding1=embedding1_input, 
        embedding2=embedding2_input
    )

    # Create the model
    pair_model = Model(inputs=[embedding1_input, embedding2_input], outputs=similarity_scores)

    return pair_model