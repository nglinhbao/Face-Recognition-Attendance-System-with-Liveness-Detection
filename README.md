# Face Recognition Attendance System with Liveness & Pose Detection

This repository contains the code for a robust face recognition attendance system developed as part of the Applied Machine Learning (COS30082) project at Swinburne University of Technology. 

## Project Overview

The system utilizes face verification techniques to manage employee access control, ensuring only registered individuals are granted entry. Key features include:

* **Face Recognition:** Utilizes a Siamese Network based on ResNet50 for generating discriminative face embeddings, enabling accurate identity verification.
* **Liveness Detection:**  Integrates a pre-trained MiniFASNetV2 model ([https://github.com/minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)) to prevent spoofing attempts using photographs or digital displays. 
* **Pose Detection:** Implements MTCNN for face detection and landmark localization, ensuring that only users directly facing the camera can register or log in.
* **User-Friendly Interface:** A web interface built with Flask provides real-time feedback for pose and liveness, facilitating seamless registration and login experiences.

## Dataset

The system was trained and evaluated on the face recognition dataset provided for the course. The dataset is organized as follows:

* `classification_data`:  Contains subfolders for training (`train_data`), validation (`val_data`), and testing (`test_data`) images for classification models.
* `verification_data`: Contains pairs of images and corresponding labels ('same' or 'different') for training and evaluating the Siamese Network. 

## Model Architecture and Training

* **Metric Learning with ResNet:** 
    * The core of the system is a Siamese Network using a shared ResNet50 backbone (pre-trained on ImageNet) for embedding generation. 
    * The network is trained with triplet loss, minimizing the distance between embeddings of similar pairs while maximizing the distance between those of dissimilar pairs.
* **Spoofing Detection:**
    * The pre-trained MiniFASNetV2 model is integrated into the system to analyze captured images and flag potential spoofing attempts.
* **Pose Detection:**
    * MTCNN is used for face detection and landmark localization. Head pose is estimated based on the spatial arrangement of detected landmarks, and a predefined frontal pose range is enforced for authentication. 

## Running the System

1. **Clone the Repository:** `git clone https://github.com/your-username/your-repository-name.git`
2. **Install Dependencies:** `pip install -r requirements.txt` 
3. **Dataset Setup (optional):**
   * Place the `classification_data` and `verification_data` directories in the project's root directory.
   * Dataset link: https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation
4. **Model Training (Optional):** 
   * The pre-trained model weights are included. If you wish to retrain the model, run the `train.py` script (ensure you have sufficient computational resources).
5. **Run the Application:** 
   * Execute `python app.py` to start the Flask web server.
6. **Access the Interface:**
   * Open a web browser and navigate to `http://127.0.0.1:5000/` to access the attendance system interface.

## Evaluation

The Metric Learning model with ResNet demonstrated the best performance for face verification, achieving a high AUC score (0.71) on the test set. The system also demonstrated excellent accuracy in spoofing and pose detection. A detailed analysis of the results and model selection process is available in the project report (report.pdf).

## Innovation

The system's innovative aspect lies in integrating pose detection to enforce frontal face verification, enhancing security by preventing non-frontal image usage during authentication.

## Future Work

* Explore techniques to further reduce the validation loss of the Siamese Network, potentially through regularization and hyperparameter tuning.
* Investigate incorporating additional face recognition models or ensembles for improved accuracy. 
* Implement more sophisticated spoofing attacks and countermeasures to enhance the system's robustness against evolving threats.

## Acknowledgements

* Minivision AI for providing the MiniFASNetV2 model ([https://github.com/minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)).
* Nawaf Al Ageel for their work on side profile detection ([https://github.com/nawafalageel/Side-Profile-Detection](https://github.com/nawafalageel/Side-Profile-Detection)), which offered insights into alternative pose detection methods.

## Author

* Nguyen Linh Bao Nguyen
