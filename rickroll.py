import cv2
import time
import threading
import pygame
import socket
import pickle

# Function to play the Rick Roll audio
def play_rick_roll(start_time=0):
    pygame.mixer.init()
    pygame.mixer.music.load('rickroll.mp3')  # Make sure you have the audio file in the same directory or provide the correct path
    pygame.mixer.music.play(start=start_time)  # Start playback at the given time
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)  # Keep the thread alive while music is playing

# Function to record video in Gaussian blur grayscale
def record_video(cap):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 format
    out = cv2.VideoWriter('motion_blur.mp4', fourcc, 20.0, (1280, 720), isColor=False)  # Set resolution to 1280x720

    start_time = time.time()
    while time.time() - start_time < 10:  # Record for 10 seconds
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))  # Resize frame to 1280x720
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        out.write(blur)

    out.release()
    print("Recording saved as 'motion_blur.mp4'")

# Server function to handle communication between devices
def server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", 9999))  # Listen on all network interfaces, port 9999
    server_socket.listen(3)  # Allow up to 3 connections
    print("Server started. Waiting for connections...")

    clients = []
    current_playback_time = 0

    def handle_client(client_socket):
        nonlocal current_playback_time
        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break

                message = pickle.loads(data)
                if message["type"] == "motion_detected":
                    print("Motion detected on a client. Updating playback time.")
                    current_playback_time = message["playback_time"]
                    threading.Thread(target=play_rick_roll, args=(current_playback_time,)).start()

                    # Notify all clients to pause detection
                    for client in clients:
                        if client != client_socket:
                            client.send(pickle.dumps({"type": "pause_detection"}))

            except Exception as e:
                print(f"Error handling client: {e}")
                break

        client_socket.close()
        clients.remove(client_socket)

    while True:
        client_socket, _ = server_socket.accept()
        clients.append(client_socket)
        print("New client connected.")
        threading.Thread(target=handle_client, args=(client_socket,), daemon=True).start()

# Client function to communicate with the server
def client(server_ip):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, 9999))
    print("Connected to server.")

    def listen_to_server():
        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break

                message = pickle.loads(data)
                if message["type"] == "pause_detection":
                    print("Pausing motion detection as another device is playing.")
                    global paused
                    paused = True
                    time.sleep(30)  # Pause detection for 30 seconds
                    paused = False

            except Exception as e:
                print(f"Error listening to server: {e}")
                break

    threading.Thread(target=listen_to_server, daemon=True).start()

    return client_socket

# Function for motion detection (used by clients)
def motion_detection(client_socket):
    cap = cv2.VideoCapture(0)  # 0 is typically the default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    if not ret or frame1 is None or frame2 is None:
        print("Error: Could not read frames from webcam.")
        cap.release()
        return

    global paused
    paused = False  # To pause detection when another device is playing
    last_played_time = 0  # To track the last time the music was played
    cooldown_period = 30  # Cooldown period in seconds

    print("Motion detection started. Move in front of the webcam to trigger the Rick Roll!")

    try:
        while cap.isOpened():
            if paused:
                time.sleep(1)
                continue

            # Ensure frames are valid before processing
            if frame1 is None or frame2 is None:
                print("Warning: Empty frame detected. Skipping iteration.")
                frame1 = frame2
                ret, frame2 = cap.read()
                continue

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

            # If motion is detected and cooldown period has passed
            current_time = time.time()
            if motion_detected and (current_time - last_played_time) > cooldown_period:
                print("Motion detected! Playing Rick Roll and notifying server...")
                last_played_time = current_time

                playback_time = pygame.mixer.music.get_pos() / 1000 if pygame.mixer.music.get_busy() else 0
                client_socket.send(pickle.dumps({"type": "motion_detected", "playback_time": playback_time}))
                threading.Thread(target=record_video, args=(cap,)).start()

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

if __name__ == "__main__":
    mode = input("Enter mode (server/client): ").strip().lower()
    server_ip = None
    client_socket = None

    if mode == "server":
        threading.Thread(target=server, daemon=True).start()
        print("Server is running without a webcam.")
        while True:
            time.sleep(1)  # Keep the server running

    elif mode == "client":
        server_ip = input("Enter server IP address: ").strip()
        client_socket = client(server_ip)
        motion_detection(client_socket)

    else:
        print("Invalid mode. Exiting.")
