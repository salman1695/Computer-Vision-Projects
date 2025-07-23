
def recognize():
    import cv2
    import streamlit as st
    import face_recognition as fr
    import time
    import pickle
    import os
    import numpy as np

    def webcamSuccess(run ,detect , capture , name):
        status_message = st.empty()
        stframe = st.empty()
        status_message.info("plz wait...")

        cap = cv2.VideoCapture(0)
     
        
        time.sleep(1)  
        while run:
            ret , frame = cap.read()
            frame = cv2.flip(frame , 1)
            status_message.empty()

            if detect:
                faces = fr.face_locations(frame , model="hog")
                face_embedding = fr.face_encodings(frame , faces)
                for (top , right , bottom , left), face_embedding in zip(faces , face_embedding):
                    cv2.rectangle(frame  , (left , top) , (right , bottom) , (0,0, 255) , 1 )
                    if trained_model:
                        match = fr.compare_faces(train_map["embedding"] ,face_embedding)
                        faceDistance = fr.face_distance(train_map["embedding"] , face_embedding)
                        minFaceDistance = min(faceDistance)
                        minFaceIdx = np.argmin(faceDistance)

                        if match[minFaceIdx] and minFaceDistance < 0.5:
                            label = train_map["labels"][minFaceIdx]
                            os.system(f"say {label} ")
                            cv2.putText(frame , f"{label}" , (left-6 , top - 6) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0, 255 , 0 ) , 1 )
                        else:
                            cv2.putText(frame , "Unknown!" , (left-6 , top - 6) , cv2.FONT_HERSHEY_SIMPLEX, 1 , (0, 255 , 0 ) ,1 )

                        


                    else:
                        cv2.putText(frame , "Unknown!" , (left-6 , top - 6) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0, 255 , 0 ) , 1 )

            stframe.image(frame , channels="BGR")
           
            if capture:
                #filename = f"{name}_{int(time.time())}.jpg" # her time unique name sy img save
                filename = f"{name}.jpg" #  just name sy img save
                cv2.imwrite(filename, frame)
                st.success(f"Saved {filename}")
                break
            # count = 0
            # if capture:
            #     filename = f"{name} {count}.jpg"
            #     cv2.imwrite(filename, frame)  # frame is your captured image
            #     count += 1
            #     break


        cap.release()
        cv2.destroyAllWindows()
    

    def train(name):
        faceimg = fr.load_image_file(f'{name}.jpg')
        facebb = fr.face_locations(faceimg)
        faceemdb = fr.face_encodings(faceimg , facebb)

        if os.path.isfile("trainmodel.pkl"):
            with open("trainmodel.pkl", "rb") as f:
                train_map = pickle.load(f)
        else:
            train_map = {"embedding": [], "labels": []}

        for emdb in faceemdb:
            train_map["embedding"].append(emdb)
            train_map["labels"].append(name)

        st.write(train_map)
        with open("trainmodel.pkl", "wb") as f:
            pickle.dump(train_map, f)



    tabs = ["Real time feed" , "training"]
    choise = st.sidebar.selectbox("Mode! ", tabs)
    trained_model = False
    train_map ={"embedding":[] , 'labels' :[]}
    capture = False


    if os.path.isfile("trainmodel.pkl"):
        with open("trainmodel.pkl", "rb") as f:
            train_map = pickle.load(f)

        trained_model = True
    else:
        trained_model = False

    
    if choise == tabs[0]:
        capture = False

        st.title("Face recognition")
        st.write(" Using face_recognition library")
        st.markdown("##### open camera ")
        btn = st.button("open" )
        btnE= st.button("close")
        
        if btn:
            webcamSuccess(True , True , capture,"")

        if btnE:
            webcamSuccess(False, False , capture,"")
    else:
        capture = False
        st.title(" Training")
        name = st.text_input("Enter your name: ")
        if name != "":
            if st.button("Capture"):
                capture = True
            if st.button("train"):
                train(name)
            webcamSuccess(True , False , capture , name)