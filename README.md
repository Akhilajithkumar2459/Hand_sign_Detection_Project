Hand Sign Recognition using MediaPipe and RandomForest
This project is a real-time hand sign recognition application. It uses Google's MediaPipe library to detect hand landmarks from a webcam feed, a Random Forest classifier to recognize the signs, and a Streamlit web application to provide an interactive user interface.

🌟 Features
Real-time Hand Tracking: Utilizes MediaPipe Hands to accurately detect and track 21 keypoints on the hand in real-time.

Machine Learning Model: Employs a Scikit-learn RandomForestClassifier to classify hand gestures based on the extracted landmark data.

Data Collection: Includes a script to capture hand landmark data and save it to a CSV file for training purposes.

Interactive Web App: A user-friendly interface built with Streamlit that allows users to see the real-time classification of their hand signs via their webcam.

⚙️ How It Works
The project is divided into three main stages:

Data Extraction:

The mediapipe library is used to detect hand landmarks from a live video stream or static images.

For each detected hand, the 3D coordinates (x, y, z) of the 21 landmarks are extracted.

This data is flattened and saved into a CSV file, with each row representing a single frame/image and each column representing a landmark coordinate. A corresponding label for the hand sign is also saved.

Model Training:

The landmark data from the CSV file is loaded using pandas.

The dataset is split into features (landmark coordinates) and labels (the hand signs).

A RandomForestClassifier from scikit-learn is trained on this dataset.

The trained model is then saved as a pickle file (.pkl) for later use in the application.

Real-time Recognition (Streamlit App):

The Streamlit application starts the user's webcam.

For each frame, it uses MediaPipe to detect hand landmarks.

The extracted landmarks are fed into the pre-trained Random Forest model.

The model predicts the corresponding hand sign.

The predicted sign, along with the confidence score, is displayed on the screen over the webcam feed.

🛠️ Technologies Used
Python: The core programming language.

OpenCV: For camera access and image processing.

MediaPipe: For hand landmark detection.

Scikit-learn: For training the Random Forest classification model.

Pandas: For data manipulation and CSV file handling.

Streamlit: To create the interactive web application.

🚀 Setup and Installation
Follow these steps to get the project up and running on your local machine.

Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:
Make sure you have a requirements.txt file in your repository with all the necessary libraries.

pip install -r requirements.txt

If you don't have one, you can create it with the following contents:

opencv-python
mediapipe
scikit-learn
pandas
streamlit

▶️ Usage
1. Data Collection
To train the model on your own custom signs, run the data collection script. (You will likely need to create this script if you haven't already). This script should open your webcam and save the landmark data for a specific sign when you press a key.

2. Model Training
After collecting the data, train the classifier by running the training script:

python train_model.py

This will generate a hand_sign_model.pkl file.

3. Run the Streamlit App
Launch the main application to start recognizing signs in real-time:

streamlit run app.py

This will open a new tab in your web browser with the application running.

📁 Project Structure
.
├── 📄 app.py                 # The main Streamlit application file
├── 📄 train_model.py          # Script to train the ML model
├── 📄 data_collection.py     # Script to collect hand landmark data (optional)
├── 📂 data
│   └── 📄 hand_signs.csv     # CSV file containing the training data
├── 📄 hand_sign_model.pkl    # The saved, pre-trained model file
├── 📄 requirements.txt       # List of Python dependencies
└── 📄 README.md              # This file

🤝 Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please feel free to open an issue or submit a pull request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
