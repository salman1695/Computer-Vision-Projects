
def dection():
    import cv2
    import streamlit as st



    stframe = st.empty()


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def webcamSuccess(run):
        cap = cv2.VideoCapture(0)
        while run:
            ret , frame = cap.read()
            frame = cv2.flip(frame , 1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame , "Salman" , (x+5 , y-5) ,cv2.FONT_HERSHEY_PLAIN , 2 , (0 , 255 , 0) , 2  ) 
            stframe.image(frame , channels="BGR")
        cap.release()
        cv2.destroyAllWindows()



    st.title("Face Detection")
    st.write("Harcascade model are use in Face Detection")
    st.markdown("##### open camera ")
    btn = st.button("open " )
    btnE= st.button("close ")
    
    if btn:
        webcamSuccess(True)


    if btnE:
        webcamSuccess(False)
