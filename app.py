import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading
from textblob import TextBlob
from gtts import gTTS
import os

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
st.success("Model loaded successfully.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize variables
sentence = ""  # Stores the entire sentence
word = ""  # Stores the current word being formed
prev_char = ""  # Tracks the previous detected character
count_same_char = 0  # Counts how many times the same character is detected consecutively
char_threshold = 10  # Threshold to consider the character finalized

# Define the word processing function
def process_word(word_to_process):
    try:
        corrected_word = str(TextBlob(word_to_process).correct())
        st.write(f"Corrected Word: {corrected_word}")
        tts = gTTS(text=corrected_word, lang='en')
        audio_path = "word.mp3"
        tts.save(audio_path)
        # Play the audio directly in the app
        with open(audio_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")
        os.remove(audio_path)
    except Exception as e:
        st.error(f"Error processing word: {e}")

st.title("Sign Language Detection and Sentence Formation")

# Webcam input
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

try:
    while cap.isOpened():
        data_aux = []  # List to hold processed landmark data
        x_ = []  # List to hold x-coordinates
        y_ = []  # List to hold y-coordinates

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Process the video frame
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Draw landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Extract landmarks for prediction
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Skip if landmarks data is incomplete
            if len(data_aux) > 42:
                continue

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

            # Handle consecutive character detection
            if predicted_character == prev_char:
                count_same_char += 1
            else:
                count_same_char = 0
                prev_char = predicted_character

            if count_same_char == char_threshold:
                if predicted_character == "Space":
                    if word != "":
                        sentence += word + " "
                        threading.Thread(target=process_word, args=(word,)).start()
                        word = ""
                else:
                    word += predicted_character

                count_same_char = 0
                prev_char = ""

            # Draw bounding box and display character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "Sentence: " + sentence + word, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Display the frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")

except Exception as e:
    st.error(f"Error during processing: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()