import cv2
import numpy as np
from tf_keras.models import load_model
from tf_keras.preprocessing import image
import pyttsx3
import streamlit as st

model_best = load_model('face_model.h5')
engine = pyttsx3.init()
# Classes 7 emotional states
class_names = ['Angry', 'Disgusted', 'Fear',
               'Happy', 'Sad', 'Surprise', 'Neutral']
# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)
frame_window = st.image([])
frame_counter = 0
st.title("Real Time Face Emotion Detection Application for Blind therapists")
activiteis = ["Home", "Webcam Face Detection", "About"]
choice = st.sidebar.selectbox("Select Activity", activiteis)
st.sidebar.markdown(f"**Developed By Pauline", unsafe_allow_html=True)

if choice == "Home":
    html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                        <h4 style="color:white;text-align:center;">
                                        Start web cam and check for real time facial emotions.</h4>
                                        </div>
                                        </br>"""
    st.markdown(html_temp_home1, unsafe_allow_html=True)
    st.write("""
             The application has the following functionalities.
             1. Real time face detection using web cam feed.
             2. Real time face emotion recognization.
             3. Text to speech conversion.
             """)
elif choice == "Webcam Face Detection":
    st.header("Webcam Live Feed")
    while True:
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            frame_counter += 1
            print(frame_counter)
            if frame_counter % 10 == 0:
                # Extract the face region
                face_roi = frame[y:y + h, x:x + w]
                # Resize the face image to the required input size for the model
                face_image = cv2.resize(face_roi, (48, 48))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = image.img_to_array(face_image)
                face_image = np.expand_dims(face_image, axis=0)
                face_image = np.vstack([face_image])
                # Predict emotion using the loaded model
                if model_best is not None:
                    predictions = model_best.predict(face_image)
                emotion_label = class_names[np.argmax(predictions)]
                # Display the emotion label on the frame
                cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # speak the emotion

                engine.say("emotion is" + emotion_label)
                engine.runAndWait()
        # Display the resulting frame
        frame_window.image(frame, channels="BGR")
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        # # Release the webcam and close all windows (bug here)
        # cap.release()
        # cv2.destroyAllWindows()

elif choice == "About":
    st.subheader("About this app")
    html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
                                <h4 style="color:white;text-align:center;">
                                Real time face emotion detection application using tensorflow.</h4>
                                </div>
                                </br>"""
    st.markdown(html_temp_about1, unsafe_allow_html=True)
    html_temp4 = """
                         		<div style="background-color:#98AFC7;padding:10px">
                         		<h4 style="color:white;text-align:center;">This Application is developed using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose.  </h4>
                         		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                         		</div>
                         		<br></br>
                         		<br></br>"""
    st.markdown(html_temp4, unsafe_allow_html=True)
else:
    pass
