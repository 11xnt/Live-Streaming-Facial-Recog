import pickle
from flask_socketio import SocketIO
import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)
socketioApp = SocketIO(app)

# enable the user's camera
camera = cv2.VideoCapture(0)


# checks for if filters has been applied to camera
# if any of the filters have been applied, we run the filter
def gen_frames():  # generate frame by frame from camera
    # loads in the same facial recognition model
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
    # create a face recognizer and reads in trainer.yml that was created in faces-train.py
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    # load the pickle file to attain the labels
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        # reverse the og_labels dictionary
        labels = {v: k for k, v in og_labels.items()}

    while True:
        # gets the total number of frames
        # as it is live, this just gets the most recent frame
        frame_counter = int(camera.get(cv2.CAP_PROP_POS_FRAMES))
        # we read the frame from the camera
        ret, frame = camera.read()
        if (frame_counter % 2 == 0):  # check if it is every other frame by modding it by 2
            # convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect if a face has been found in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

            # loop through the faces var
            for (x, y, w, h) in faces:
                # we get the region of interest in by getting the coordinates of the face
                roi_gray = gray[y:y + h, x:x + w]

                # use the roi_gray to see if it can predict a recognised persons face with the given
                id_, conf = recognizer.predict(roi_gray)
                # if the prediction falls between 30 and 85 inclusive
                if conf >= 30 and conf <= 85:
                    # put text on the with the label's name at above the roi's min Y value
                    cv2.putText(frame,
                                labels[id_] + " Match: " + str(round(conf, 2)) + "%",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA)

                color = (255, 0, 0)  # BGR 0-255 (Blue)
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                # draw a rectangle if we can detect a face
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

                break

            # Display the resulting frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        # else return frame with no processing
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        # if user pressed 'q' break
        if cv2.waitKey(1) == ord('q'):  #
            break

    camera.release()  # turn off camera
    cv2.destroyAllWindows()  # close all windows


# renders the index.html page found in templates directory
@app.route('/')
def index():
    return render_template('index.html')


# sends frames to the user that is within the index.html
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run():
    socketioApp.run(app)


if __name__ == '__main__':
    socketioApp.run(app)
