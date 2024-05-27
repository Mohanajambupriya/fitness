
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__, template_folder='A:\\Deep Learning\\')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

count = 0
position = None

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as pose:

    def generate_frames():
        global count, position
        
        while True:
            success, image = cap.read()
            if not success:
                print("Empty camera")
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            result = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_list = []

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for id, lm in enumerate(result.pose_landmarks.landmark):
                    h, w, _ = image.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    im_list.append([id, x, y])

            if len(im_list) != 0:
                # Assuming specific landmark indices for knees and hips
                left_hip_y = im_list[23][2]
                right_hip_y = im_list[24][2]
                left_knee_y = im_list[25][2]
                right_knee_y = im_list[26][2]

                # Detecting squat position
                if (left_hip_y < left_knee_y and right_hip_y < right_knee_y) or \
                        (left_hip_y < left_knee_y and right_hip_y == right_knee_y) or \
                        (left_hip_y == left_knee_y and right_hip_y < right_knee_y):
                    position = "down"
                if (left_hip_y > left_knee_y and right_hip_y > right_knee_y) and position == "down":
                    color = (0, 255, 0)  # Green for correct squat
                    mp_drawing.draw_landmarks(
                        image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                    )
                    position = "up"
                    count += 1
                    print(count)

            # Overlay squat count on the frame
            cv2.putText(image, f"Squat Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            # Yield the frame as a response to the client
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Route for the video stream page
    @app.route('/')
    def index():
        return render_template('abu.html')

    # Route for accessing the video stream
    @app.route('/squat_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    if __name__ == "__main__":
        app.run(debug=True)

cap.release()
