
def face_landmark():
    import cv2
    import streamlit as st
    import face_recognition as fr
    import time

  
    def webcamSuccess(run):

        status_message = st.empty()
        stframe = st.empty()
        status_message.info("plz wait...")

        cap = cv2.VideoCapture(0)
        time.sleep(1)
        while run:
            ret , frame = cap.read()
            frame = cv2.flip(frame , 1)

            faces = fr.face_landmarks(frame , model="large")

            for face_landmarks in faces:
                for feature_name, points in face_landmarks.items():
                    # for i in range(len(points) - 1):
                    #     pt1 = points[i]
                    #     pt2 = points[i + 1]
                    #     cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                    for point in points:
                    # Draw each landmark as a small filled circle
                        cv2.circle(frame, point, radius=2, color=(0, 255, 0), thickness=-1)

                
                    # if feature_name in ["top_lip", "bottom_lip", "left_eye", "right_eye"]:
                    #     cv2.line(frame, points[-1], points[0], (0, 255, 0), 2)
            stframe.image(frame , channels="BGR")
            status_message.empty()

        cap.release()
        cv2.destroyAllWindows()

    st.title("Face landmark detection")
    st.write(" Using face_recognition library")
    st.markdown("##### open camera ")


    btn = st.button("Open" )
    btnE= st.button("Close")
    
    if btn:
        webcamSuccess(True)


    if btnE:
        webcamSuccess(False)
