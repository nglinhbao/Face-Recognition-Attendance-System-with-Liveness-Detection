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
from anti_spoofing import spoofing_detection
import os
import io

app = Flask(__name__)

# Set up embeddings folder
EMBEDDINGS_FOLDER = 'face_embeddings'
if not os.path.exists(EMBEDDINGS_FOLDER):
    os.makedirs(EMBEDDINGS_FOLDER)

USER_IMAGES_FOLDER = 'user_images'
if not os.path.exists(USER_IMAGES_FOLDER):
    os.makedirs(USER_IMAGES_FOLDER)

target_shape = (200, 200)
embedding_dim = 256
model = create_pair_model(embedding, embedding_dim)

def get_embedding_path(filename):
    return os.path.join(EMBEDDINGS_FOLDER, filename)

def load_embedding(filename):
    embedding_path = get_embedding_path(filename)
    if os.path.exists(embedding_path):
        return np.load(embedding_path)
    return None

def save_embedding(username, embedding_vector):
    embedding_path = get_embedding_path(f'{username}.npy')
    np.save(embedding_path, embedding_vector)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'login' in request.form:
            return handle_login() 
        elif 'register' in request.form:
            username = request.form.get('username')
            return handle_registration(username)

    return render_template('index.html')

def handle_login():
    captured_image_data = request.form.get('capturedImage')
    if not captured_image_data:
        return jsonify({'error': 'Captured image is missing.'}), 400

    captured_image = preprocess_image(captured_image_data)

    cropped_face, cropped_face_base64 = extract_face(captured_image.numpy()[0])
    if cropped_face is None:
        return jsonify({'error': 'No face detected in the captured image.'}), 400

    cropped_face_vector = embedding.predict(cropped_face)[0]

    max_similarity = 0
    most_similar_user = None
    most_similar_face_base64 = None

    for filename in os.listdir(EMBEDDINGS_FOLDER):
        if filename.endswith('.npy'):
            username = filename[:-4]
            user_embedding = load_embedding(filename)
            if user_embedding is not None:
                similarity_score = model.predict([tf.expand_dims(cropped_face_vector, axis=0), tf.expand_dims(user_embedding, axis=0)])
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    most_similar_user = username

                    matched_image_path = os.path.join(USER_IMAGES_FOLDER, f"{username}.jpg")
                    with open(matched_image_path, "rb") as img_file:
                        matched_image_bytes = img_file.read()
                        most_similar_face_base64 = base64.b64encode(matched_image_bytes).decode("utf-8")

    threshold = 0.8
    print(float(max_similarity))
    if max_similarity > threshold and most_similar_user:
        return jsonify({
            'message': f'Welcome, {most_similar_user}!',
            'input_face': cropped_face_base64,  # Add input face to response
            'matched_face': most_similar_face_base64  # Add matched face to response
        })
    else:
        return jsonify({'error': 'Face not recognized. Please register.'}), 401

def handle_registration(username):
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    captured_image_data = request.form.get('capturedImage')
    if not captured_image_data:
        return jsonify({'error': 'Captured image is missing.'}), 400

    captured_image = preprocess_image(captured_image_data)

    cropped_face, cropped_face_base64 = extract_face(captured_image.numpy()[0])
    if cropped_face is None:
        return jsonify({'error': 'No face detected in the captured image.'}), 400

    image_filename = os.path.join(USER_IMAGES_FOLDER, f"{username}.jpg")  
    with open(image_filename, 'wb') as f:
        f.write(base64.b64decode(cropped_face_base64))

    # Generate embedding for the captured face
    embedding_vector = embedding.predict(cropped_face)[0]
    save_embedding(username, embedding_vector)

    return jsonify({
        'message': f'Registration successful, {username}!',
        'cropped_face': cropped_face_base64
    })


@app.route('/detect_pose', methods=['POST'])
def detect_pose():
    # Receive image data from JavaScript
    image_data = request.form.get('image')
    if image_data:
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Open the image using BytesIO
        image = Image.open(io.BytesIO(image_bytes))

        image_spoofing = image.resize((80, 80))
         # Perform spoofing detection
        is_real = spoofing_detection(image_spoofing)

        # Resize the image 
        image = image.resize(target_shape) 

        # Convert the image to a numpy array
        image_array = np.array(image)
        # Normalize the image array
        image_array = image_array.astype('float32') / 255.0
        # Expand dimensions 
        image_array = np.expand_dims(image_array, axis=0)

        # Convert PIL Image to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

        # Perform pose detection
        landmarks_, _, _, predLabelList = predFacePoseCV2(frame)
        
        # Return detection result
        is_frontal = predLabelList and predLabelList[0] == 'Frontal'

        return jsonify({'is_frontal': str(is_frontal), 'is_real': str(is_real)})
    else:
        return jsonify({'error': 'No image data received'}), 400
    

if __name__ == '__main__':
    app.run(debug=True)