import cv2
import time
import threading
import pygame

# Function to play the Rick Roll audio
def play_rick_roll():
    pygame.mixer.init()
    pygame.mixer.music.load('rickroll.mp3')  # Make sure you have the audio file in the same directory or provide the correct path
    pygame.mixer.music.play()
    time.sleep(30)  # Play for 30 seconds
    pygame.mixer.music.stop()

# Function to record video in Gaussian blur grayscale
def record_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('motion_blur.avi', fourcc, 20.0, (640, 480), isColor=False)

    start_time = time.time()
    while time.time() - start_time < 10:  # Record for 10 seconds
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        out.write(blur)

    out.release()
    print("Recording saved as 'motion_blur.avi'")

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default camera
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables
ret, frame1 = cap.read()
ret, frame2 = cap.read()

last_played_time = 0  # To track the last time the music was played
cooldown_period = 30  # Cooldown period in seconds

print("Motion detection started. Move in front of the webcam to trigger the Rick Roll!")

try:
    while cap.isOpened():
        # Calculate the difference between two frames
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        # Check for motion
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            motion_detected = True
            break

        # If motion is detected and cooldown period has passed, play the Rick Roll audio and record video
        current_time = time.time()
        if motion_detected and (current_time - last_played_time) > cooldown_period:
            print("Motion detected! Playing Rick Roll and recording video...")
            threading.Thread(target=play_rick_roll).start()
            threading.Thread(target=record_video).start()
            last_played_time = current_time

        # Display the frames (optional)
        cv2.imshow('Motion Detector', frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam and windows closed.")
