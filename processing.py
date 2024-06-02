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
from modelling import target_shape

def preprocess_image(image_data):
    """
    Load the specified file as a JPEG image, preprocess it, resize it to the target shape,
    and normalize it.
    """
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = tf.image.decode_image(image_bytes, channels=3, dtype=tf.uint8)
    image = tf.expand_dims(image, axis=0)

    return image

def extract_face(image):
    """
    Extract the face from the given image using face_recognition library,
    returning a cropped face image.
    """
    # Find face locations in the image
    face_locations_list = face_locations(image)

    if len(face_locations_list) == 0:
        return None  # No face found
    
    # Get the top, right, bottom, left coordinates of the first detected face
    top, right, bottom, left = face_locations_list[0]

    # Convert the image to a PIL Image
    pil_image = Image.fromarray(image.astype(np.uint8))

    # Crop the face from the PIL image
    cropped_face = pil_image.crop((left, top, right, bottom))

    # Convert the cropped face back to a TensorFlow image
    cropped_face_tf = tf.convert_to_tensor(np.array(cropped_face))
    shown_image = cropped_face_tf

    cropped_face_tf = tf.expand_dims(cropped_face_tf, axis=0)  # Add batch dimension

    cropped_face_tf = tf.image.convert_image_dtype(cropped_face_tf, tf.float32)
    cropped_face_tf = tf.image.resize(cropped_face_tf, target_shape)

    # Encode the cropped face as base64
    shown_image = tf.image.resize(shown_image, target_shape)
    jpg_image = tf.io.encode_jpeg(tf.cast(shown_image, tf.uint8))

    encoded_image = base64.b64encode(jpg_image.numpy()).decode('utf-8')
        
    return cropped_face_tf, encoded_image