import numpy as np
from keras.models import load_model
import cv2
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
import mediapipe as mp
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import threading
from datetime import datetime

# Load face detection and emotion classification models
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3D model points for head pose estimation
model_points = np.array([
    [0.0, 0.0, 0.0],           # Nose tip
    [0.0, -330.0, -65.0],      # Chin
    [-225.0, 170.0, -135.0],   # Left eye left corner
    [225.0, 170.0, -135.0],    # Right eye right corner
    [-150.0, -150.0, -125.0],  # Left mouth corner
    [150.0, -150.0, -125.0]    # Right mouth corner
], dtype=np.float64)

# Camera matrix
camera_matrix = np.array([
    [1000, 0, 320],
    [0, 1000, 240],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

# Email configuration (CORRECTED - added missing comma)
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',  # For Gmail
    'smtp_port': 587,
    'sender_email': 'malarvizhi.ramu1986@gmail.com',  # ADDED MISSING COMMA HERE
    'sender_password':'yazyisjpzfsiqwes',  # Use App Password for Gmail
    'receiver_email': 'kowsalyamoorthy16@gmail.com'
}

# Alert tracking variables
alert_start_time = None
current_alert_type = None
alert_active = False
ALERT_DURATION = 5  # seconds

def get_mediapipe_landmarks(frame, face_mesh):
    """Extract facial landmarks using MediaPipe"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None
    
    landmarks = results.multi_face_landmarks[0]
    h, w = frame.shape[:2]
    
    landmark_points = []
    for landmark in landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        landmark_points.append([x, y])
    
    return np.array(landmark_points)

def estimate_head_pose_mediapipe(landmarks, frame):
    """Estimate head pose using MediaPipe landmarks"""
    if landmarks is None:
        return "Unknown", None, None, None
    
    mp_indices = [1, 152, 33, 263, 61, 291]
    
    image_points = np.array([
        landmarks[1],      # Nose tip
        landmarks[152],    # Chin
        landmarks[33],     # Left eye left corner
        landmarks[263],    # Right eye right corner
        landmarks[61],     # Left mouth corner
        landmarks[291]     # Right mouth corner
    ], dtype=np.float64)
    
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        nose_end_point3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        nose_end_point2D, _ = cv2.projectPoints(
            nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs
        )
        
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        x_angle = angles[0] * 360
        y_angle = angles[1] * 360
        z_angle = angles[2] * 360
        
        pose = "Forward"
        if y_angle < -15:
            pose = "Looking Left"
        elif y_angle > 15:
            pose = "Looking Right"
        elif x_angle < -10:
            pose = "Looking Down"
        elif x_angle > 10:
            pose = "Looking Up"
        
        return pose, rotation_vector, translation_vector, nose_end_point2D
        
    except Exception as e:
        print(f"Pose estimation error: {e}")
        return "Unknown", None, None, None

def send_alert_email(image_frame, alert_type, emotion=None, head_pose=None):
    """Send email with captured image"""
    def send_email():
        try:
            # Create message
            msg = MIMEMultipart()
            msg['Subject'] = f'Security Alert: {alert_type} Detected'
            msg['From'] = EMAIL_CONFIG['sender_email']
            msg['To'] = EMAIL_CONFIG['receiver_email']
            
            # Email body
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body = f"""
Security Alert Notification

Alert Type: {alert_type}
Time: {timestamp}
"""
            
            if emotion:
                body += f"Detected Emotion: {emotion}\n"
            if head_pose:
                body += f"Head Pose: {head_pose}\n"
            
            body += "\nThis alert was triggered because the condition persisted for 5 seconds."
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image
            _, img_buffer = cv2.imencode('.jpg', image_frame)
            mime_image = MIMEImage(img_buffer.tobytes())
            mime_image.add_header('Content-Disposition', 'attachment', filename=f'alert_{timestamp.replace(":", "-")}.jpg')
            msg.attach(mime_image)
            
            # Send email
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
            server.quit()
            
            print(f"Alert email sent successfully for {alert_type}!")
            
        except Exception as e:
            print(f"Failed to send email: {e}")
    
    # Send email in a separate thread to avoid blocking
    email_thread = threading.Thread(target=send_email)
    email_thread.daemon = True
    email_thread.start()

def check_alert_condition(current_emotion, current_head_pose):
    """Check if alert conditions are met"""
    global alert_start_time, current_alert_type, alert_active
    
    # Define alert conditions
    emotion_alerts = ['Angry', 'Fear']
    pose_alerts = ['Looking Left', 'Looking Right']
    
    current_condition = None
    alert_type = None
    
    # Check emotion alert
    if current_emotion in emotion_alerts:
        current_condition = current_emotion
        alert_type = f"Emotion: {current_emotion}"
    
    # Check head pose alert (only if no emotion alert)
    elif current_head_pose in pose_alerts:
        current_condition = current_head_pose
        alert_type = f"Head Pose: {current_head_pose}"
    
    # If no alert condition
    if not current_condition:
        alert_start_time = None
        current_alert_type = None
        alert_active = False
        return False, None
    
    # If same alert condition continues
    if current_condition == current_alert_type:
        if alert_start_time and (time.time() - alert_start_time) >= ALERT_DURATION:
            if not alert_active:
                alert_active = True
                return True, alert_type
    else:
        # New alert condition detected
        alert_start_time = time.time()
        current_alert_type = current_condition
        alert_active = False
    
    return False, alert_type

def draw_alert_info(frame, alert_type, time_remaining):
    """Draw alert information on frame"""
    h, w = frame.shape[:2]
    
    # Draw alert box
    cv2.rectangle(frame, (10, h-100), (w-10, h-10), (0, 0, 255), -1)
    cv2.rectangle(frame, (10, h-100), (w-10, h-10), (255, 255, 255), 2)
    
    # Draw alert text
    cv2.putText(frame, f'ALERT: {alert_type}', (20, h-70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Time remaining: {time_remaining:.1f}s', (20, h-40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def draw_head_pose_visualization(frame, landmarks, nose_end_point2D):
    """Draw head pose visualization"""
    if landmarks is not None and nose_end_point2D is not None:
        # Draw pose direction line
        nose_point = tuple(landmarks[1].astype(int))
        nose_end_point = tuple(nose_end_point2D[0][0].astype(int))
        cv2.arrowedLine(frame, nose_point, nose_end_point, (0, 255, 0), 2)
        
        # Draw key landmarks
        key_indices = [1, 152, 33, 263, 61, 291]
        for idx in key_indices:
            point = tuple(landmarks[idx].astype(int))
            cv2.circle(frame, point, 3, (0, 0, 255), -1)

# Main execution
cap = cv2.VideoCapture(0)

print("Security Alert System Started")
print("Monitoring for: Angry/Fear emotions or Left/Right head poses")
print("Alerts will trigger after 5 seconds of continuous detection")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Update camera matrix
    camera_matrix[0, 2] = w // 2
    camera_matrix[1, 2] = h // 2
    
    # Get MediaPipe landmarks for head pose
    landmarks = get_mediapipe_landmarks(frame, face_mesh)
    
    # Initialize variables
    current_emotion = "Neutral"
    current_head_pose = "Forward"
    nose_end_point2D = None
    
    # Head pose detection
    if landmarks is not None:
        pose, rotation_vector, translation_vector, nose_end_point2D = estimate_head_pose_mediapipe(landmarks, frame)
        current_head_pose = pose
        draw_head_pose_visualization(frame, landmarks, nose_end_point2D)
    
    # Emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            prediction = classifier.predict(roi, verbose=0)[0]
            emotion_idx = prediction.argmax()
            current_emotion = emotion_labels[emotion_idx]
            
            # Display emotion with confidence
            confidence = prediction[emotion_idx]
            cv2.putText(frame, f'{current_emotion} ({confidence:.2f})', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display head pose
    cv2.putText(frame, f'Head Pose: {current_head_pose}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Check alert conditions
    should_alert, alert_type = check_alert_condition(current_emotion, current_head_pose)
    
    # Handle alert
    if alert_start_time and current_alert_type:
        time_elapsed = time.time() - alert_start_time
        time_remaining = max(0, ALERT_DURATION - time_elapsed)
        
        if time_remaining > 0:
            draw_alert_info(frame, alert_type, time_remaining)
        
        if should_alert:
            # Send email with current frame
            send_alert_email(frame, alert_type, 
                           emotion=current_emotion if 'Emotion' in alert_type else None,
                           head_pose=current_head_pose if 'Head Pose' in alert_type else None)
            
            # Reset alert after sending
            alert_start_time = None
            current_alert_type = None
            alert_active = False
    
    # Display status
    status_text = "Monitoring..." if not alert_start_time else f"Alert: {current_alert_type}"
    cv2.putText(frame, status_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow('Security Alert System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
face_mesh.close()
cap.release()
cv2.destroyAllWindows()