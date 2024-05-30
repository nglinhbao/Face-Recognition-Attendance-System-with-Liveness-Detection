from flask import Flask, request, jsonify, render_template
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
from modelling import create_pair_model, embedding, target_shape
from processing import preprocess_image, extract_face

app = Flask(__name__)

target_shape = (200, 200)

@app.route('/')
def index():
    print("Index route accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict the similarity between two images.
    """
    image1_file = request.files.get('image1')
    image2_data = request.form.get('image2')

    if not image1_file or not image2_data:
        return jsonify({'error': 'Both image1 and image2 are required.'}), 400

    # Process image1 (from file upload)
    image1 = preprocess_image(image1_file)

    # Process image2 (from data URL)
    image2_data = image2_data.split(',')[1]
    image2_bytes = base64.b64decode(image2_data)
    image2 = tf.image.decode_image(image2_bytes, channels=3, dtype=tf.uint8)
    image2 = tf.expand_dims(image2, axis=0)

    # Extract face images 
    cropped_face1, img_str1 = extract_face(image1.numpy()[0])
    cropped_face2, img_str2 = extract_face(image2.numpy()[0])  # Pass the correctly formatted image

    # Extract face images
    cropped_face1, img_str1 = extract_face(image1.numpy()[0])
    cropped_face2, img_str2 = extract_face(image2.numpy()[0])

    print("Cropped face 1:", cropped_face1.shape)
    print("Cropped face 2:", cropped_face2.shape)

    if cropped_face1 is None or cropped_face2 is None:
        return jsonify({'error': 'No faces detected in one or both images'})

    model = create_pair_model(embedding, target_shape)

    # Make the prediction using the face embeddings
    similarity_score = model.predict([cropped_face1, cropped_face2])

    # Return the similarity score as JSON
    return jsonify({'similarity_score': float(similarity_score[0][0]), 'cropped_face1': img_str1, 'cropped_face2': img_str2})


@app.route('/detect_pose', methods=['POST'])
def detect_pose():
    # Receive image data from JavaScript
    image_data = request.form.get('image')
    if image_data:
        # Decode base64 image data
        image_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform pose detection
        landmarks_, _, _, predLabelList = predFacePoseCV2(frame)
        
        # Return detection result
        is_frontal = predLabelList and predLabelList[0] == 'Frontal'
        return jsonify({'is_frontal': is_frontal})
    else:
        return jsonify({'error': 'No image data received'}), 400
    

if __name__ == '__main__':
    app.run(debug=True)