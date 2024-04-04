import cv2
import streamlit as st
import numpy as np
from tf_keras.models import load_model
from tf_keras.preprocessing import image
import pyttsx3


def facesentiment():
    # st.title("Real-Time Facial Analysis with Streamlit")
    model_best = load_model('face_model.h5')
    engine = pyttsx3.init()
    # Classes 7 emotional states
    class_names = ['Angry', 'Disgusted', 'Fear',
                   'Happy', 'Sad', 'Surprise', 'Neutral']

    # Load the pre-trained face cascade
    # face_cascade = cv2.CascadeClassifier(
    #     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    stframe = st.image([])  # Placeholder for the webcam feed

    while True:
        ret, frame = cap.read()
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        # Process each detected face
        for (x, y, w, h) in faces:
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

            # Convert the BGR frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the frame in Streamlit
            stframe.image(frame_rgb, channels="RGB")

            # speak the emotion
            engine.say("emotion is" + emotion_label)
            engine.runAndWait()
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Face Analysis Application #
    # st.title("Real Time Face Emotion Detection Application")
    activities = ["Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by Shrimanta Satpati    
            Email : satpatishrimanta@gmail.com  
        """)
    if choice == "Webcam Face Detection":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Real time face emotion recognition of webcam feed using OpenCV, DeepFace and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        facesentiment()

    elif choice == "About":
        st.subheader("About this app")

        html_temp4 = """
                                     		<div style="background-color:#98AFC7;padding:10px">
                                     		<h4 style="color:white;text-align:center;">This Application is developed by Shrimanta Satpati. </h4>
                                     		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                     		</div>
                                     		<br></br>
                                     		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
