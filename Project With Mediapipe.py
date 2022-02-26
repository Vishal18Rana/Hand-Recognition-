from tkinter import *
import os
import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import tkinter as tk
 
mp_hands = mp.solutions.hands

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils
def detectHandsLandmarks(image, hands, draw=True, display = True):
    '''
    This function performs hands landmarks detection on an image.
    Args:
        image:   The input image with prominent hand(s) whose landmarks needs to be detected.
        hands:   The Hands function required to perform the hands landmarks detection.
        draw:    A boolean value that is if set to true the function draws hands landmarks on the output image. 
        display: A boolean value that is if set to true the function displays the original input image, and the output 
                 image with hands landmarks drawn if it was specified and returns nothing.
    Returns:
        output_image: A copy of input image with the detected hands landmarks drawn if it was specified.
        results:      The output of the hands landmarks detection on the input image.
    '''
    
    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)
    
    # Check if landmarks are found and are specified to be drawn.
    if results.multi_hand_landmarks and draw:
        
        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(image = output_image, landmark_list = hand_landmarks,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),
                                                                                     thickness=2, circle_radius=2))
    
    # Check if the original input image and the output image are specified to be displayed.
    if display:
        
        # Display the original input image and the output image.
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off');
        
    # Otherwise
    else:
        
        # Return the output image and results of hands landmarks detection.
        return output_image, results  
def countFingers(image, results, draw=True, display=True):
    '''
    This function will count the number of fingers up for each hand in the image.
    Args:
        image:   The image of the hands on which the fingers counting is required to be performed.
        results: The output of the hands landmarks detection performed on the image of the hands.
        draw:    A boolean value that is if set to true the function writes the total count of fingers of the hands on the
                 output image.
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image:     A copy of the input image with the fingers count written, if it was specified.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        count:            A dictionary containing the count of the fingers that are up, of both hands.
    '''
    
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()
    
    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}
    
    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    
    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
    
    
    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):
        
        # Retrieve the label of the found hand.
        hand_label = hand_info.classification[0].label
        
        # Retrieve the landmarks of the found hand.
        hand_landmarks =  results.multi_hand_landmarks[hand_index]
        
        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:
            
            # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]
            
            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                
                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1
        
        # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
        
        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
            
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper()+"_THUMB"] = True
            
            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1
     
    # Check if the total count of the fingers of both hands are specified to be written on the output image.
    if draw:

        # Write the total count of the fingers of both hands on the output image.
        cv2.putText(output_image, " Total Fingers: ", (10, 25),cv2.FONT_HERSHEY_COMPLEX, 1, (20,255,155), 2)
        cv2.putText(output_image, str(sum(count.values())), (width//2-150,240), cv2.FONT_HERSHEY_SIMPLEX,
                    8.9, (20,255,155), 10, 10)

    # Check if the output image is specified to be displayed.
    if display:
        
        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:

        # Return the output image, the status of each finger and the count of the fingers up of both hands.
        return output_image, fingers_statuses, count
def recognizeGestures(image, fingers_statuses, count, draw=True, display=True):
    '''
    This function will determine the gesture of the left and right hand in the image.
    Args:
        image:            The image of the hands on which the hand gesture recognition is required to be performed.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands. 
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        draw:             A boolean value that is if set to true the function writes the gestures of the hands on the
                          output image, after recognition.
        display:          A boolean value that is if set to true the function displays the resultant image and 
                          returns nothing.
    Returns:
        output_image:   A copy of the input image with the left and right hand recognized gestures written if it was 
                        specified.
        hands_gestures: A dictionary containing the recognized gestures of the right and left hand.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Store the labels of both hands in a list.
    hands_labels = ['RIGHT', 'LEFT']
    
    # Initialize a dictionary to store the gestures of both hands in the image.
    hands_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}
    
    # Iterate over the left and right hand.
    for hand_index, hand_label in enumerate(hands_labels):
        
        # Initialize a variable to store the color we will use to write the hands gestures on the image.
        # Initially it is red which represents that the gesture is not recognized.
        color = (0, 0, 255)
        
        # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        if count[hand_label] == 2  and fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_INDEX']:
            
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "V SIGN"
            
            # Update the color value to green.
            color=(0,255,0)
            
        ####################################################################################################################            
        
        # Check if the person is making the 'SPIDERMAN' gesture with the hand.
        ##########################################################################################################################################################
        
        # Check if the number of fingers up is 3 and the fingers that are up, are the thumb, index and the pinky finger.
        elif count[hand_label] == 3 and fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_PINKY']:
                
            # Update the gesture value of the hand that we are iterating upon to SPIDERMAN SIGN.
            hands_gestures[hand_label] = "SPIDERMAN SIGN"

            # Update the color value to green.
            color=(0,255,0)
                
        ##########################################################################################################################################################
        
        # Check if the person is making the 'HIGH-FIVE' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 5, which means that all the fingers are up.
        elif count[hand_label] == 5:
            
            # Update the gesture value of the hand that we are iterating upon to HIGH-FIVE SIGN.
            hands_gestures[hand_label] = "HIGH-FIVE SIGN"
            
            # Update the color value to green.
            color=(0,255,0)
        elif fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_PINKY']:
                
            # Update the gesture value of the hand that we are iterating upon to SPIDERMAN SIGN.
            hands_gestures[hand_label] = "IHATEYOU"

            # Update the color value to green.
            color=(0,255,0)
        elif fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_PINKY']  :
            hands_gestures[hand_label] = "PLAY"
            color=(0,255,0)
        elif  fingers_statuses[hand_label+'_THUMB'] and count[hand_label] == 1  :
            hands_gestures[hand_label] = "YES"
            color=(0,255,0)
        elif fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_MIDDLE']:
            hands_gestures[hand_label] = "NO"
            color=(0,255,0)
        
       
            
                
            
       
        ####################################################################################################################  
        
        # Check if the hands gestures are specified to be written.
        if draw:
        
            # Write the hand gesture on the output image. 
            cv2.putText(output_image, hand_label +': '+ hands_gestures[hand_label] , (10, (hand_index+1) * 60),
                        cv2.FONT_HERSHEY_PLAIN, 4, color, 5)
            
    
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:

        # Return the output image and the gestures of the both hands.
        return output_image, hands_gestures
    
    
# Designing window for registration
def register():
    
    global register_screen
    register_screen = Toplevel(main_screen)
    register_screen.title("Register")
    register_screen.geometry("300x250")
 
    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()
    Label(register_screen, text="Please enter details below", bg="blue").pack()
    Label(register_screen, text="").pack()
    username_lable = Label(register_screen, text="Username * ")
    username_lable.pack()
    username_entry = Entry(register_screen, textvariable=username)
    username_entry.pack()
    password_lable = Label(register_screen, text="Password * ")
    password_lable.pack()
    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()
    Label(register_screen, text="").pack()
    Button(register_screen, text="Register", width=10, height=1, bg="blue", command = register_user).pack()
def login():
    global login_screen
    login_screen = Toplevel(main_screen)
    login_screen.title("Login")
    login_screen.geometry("300x250")
    Label(login_screen, text="Please enter details below to login").pack()
    Label(login_screen, text="").pack()
 
    global username_verify
    global password_verify
 
    username_verify = StringVar()
    password_verify = StringVar()
 
    global username_login_entry
    global password_login_entry
 
    Label(login_screen, text="Username * ").pack()
    username_login_entry = Entry(login_screen, textvariable=username_verify)
    username_login_entry.pack()
    Label(login_screen, text="").pack()
    Label(login_screen, text="Password * ").pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show= '*')
    password_login_entry.pack()
    Label(login_screen, text="").pack()
    Button(login_screen, text="Login", width=10, height=1, command = login_verify).pack()
 
# Implementing event on register button
def register():
    global register_screen
    register_screen = Toplevel(main_screen)
    register_screen.title("Register")
    register_screen.geometry("300x250")
 
    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()
 
    Label(register_screen, text="Please enter details below", bg="blue").pack()
    Label(register_screen, text="").pack()
    username_lable = Label(register_screen, text="Username * ")
    username_lable.pack()
    username_entry = Entry(register_screen, textvariable=username)
    username_entry.pack()
    password_lable = Label(register_screen, text="Password * ")
    password_lable.pack()
    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()
    Label(register_screen, text="").pack()
    Button(register_screen, text="Register", width=10, height=1, bg="blue", command = register_user).pack()
 
 
# Designing window for login 
def login():
    global login_screen
    login_screen = Toplevel(main_screen)
    login_screen.title("Login")
    login_screen.geometry("300x250")
    Label(login_screen, text="Please enter details below to login").pack()
    Label(login_screen, text="").pack()
 
    global username_verify
    global password_verify
 
    username_verify = StringVar()
    password_verify = StringVar()
 
    global username_login_entry
    global password_login_entry
 
    Label(login_screen, text="Username * ").pack()
    username_login_entry = Entry(login_screen, textvariable=username_verify)
    username_login_entry.pack()
    Label(login_screen, text="").pack()
    Label(login_screen, text="Password * ").pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show= '*')
    password_login_entry.pack()
    Label(login_screen, text="").pack()
    Button(login_screen, text="Login", width=10, height=1, command = login_verify).pack()
 
# Implementing event on register button
def register_user():
 
    username_info = username.get()
    password_info = password.get()
 
    file = open(username_info, "w")
    file.write(username_info + "\n")
    file.write(password_info)
    file.close()
 
    username_entry.delete(0, END)
    password_entry.delete(0, END)
 
    Label(register_screen, text="Registration Success", fg="green", font=("calibri", 11)).pack()
 
# Implementing event on login button 
 
def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_login_entry.delete(0, END)
    password_login_entry.delete(0, END)
 
    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            login_sucess()
 
        else:
            password_not_recognised()
 
    else:
        user_not_found()
 
# Designing popup for login success
def login_sucess():
    global login_success_screen
    login_success_screen = Toplevel(login_screen)
    login_success_screen.title("Success")
    login_success_screen.geometry("150x100")
    Label(login_success_screen, text="Login Success").pack()
    Button(login_success_screen, text="OK", command=delete_login_success).pack()
 
# Designing popup for login invalid password
 
def password_not_recognised():
    global password_not_recog_screen
    password_not_recog_screen = Toplevel(login_screen)
    password_not_recog_screen.title("Success")
    password_not_recog_screen.geometry("150x100")
    Label(password_not_recog_screen, text="Invalid Password ").pack()
    Button(password_not_recog_screen, text="OK", command=delete_password_not_recognised).pack()
 
# Designing popup for user not found
 
def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("Success")
    user_not_found_screen.geometry("150x100")
    Label(user_not_found_screen, text="User Not Found").pack()
    Button(user_not_found_screen, text="OK", command=delete_user_not_found_screen).pack()
 
# Deleting popups
 
def delete_login_success():
    login_success_screen.destroy()
    login_screen.destroy()
    main_screen.destroy()
    def volume():
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volRange = volume.GetVolumeRange()
        minVol, maxVol, volBar, volPer = volRange[0], volRange[1], 400, 0
        wCam, hCam = 640, 480
        cam = cv2.VideoCapture(0)
        cam.set(3, wCam)
        cam.set(4, hCam)
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cam.isOpened():
                success, image = cam.read()

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:

                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                
                        )
                lmList = []
                if results.multi_hand_landmarks:
                    
                    myHand = results.multi_hand_landmarks[0]
                    for id, lm in enumerate(myHand.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                if len(lmList) != 0:
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    cv2.circle(image, (x1, y1), 15, (255, 255, 255))
                    cv2.circle(image, (x2, y2), 15, (255, 255, 255))
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    length = math.hypot(x2 - x1, y2 - y1)
                    if length < 50:
                        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    vol = np.interp(length, [50, 220], [minVol, maxVol])
                    volume.SetMasterVolumeLevel(vol, None)
                    volBar = np.interp(length, [50, 220], [400, 150])
                    volPer = np.interp(length, [50, 220], [0, 100])
                     # Volume Bar
                    cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                    cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 0), 3)
                cv2.imshow('handDetector', image)
                k = cv2.waitKey(1) & 0xFF

                # Check if 'ESC' is pressed and break the loop.
                if (k == 27):
                    break
        cam.release()
        cv2.destroyAllWindows()

    def qu():
        camera_video = cv2.VideoCapture(0)
        camera_video.set(3, 1280)
        camera_video.set(4, 960)
        cv2.namedWindow('Fingers Counter', cv2.WINDOW_NORMAL)

        # Iterate until the webcam is accessed successfully.
        while camera_video.isOpened():

            ok, frame = camera_video.read()

            # Check if frame is not read properly then continue to the next iteration to read the next frame.
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

        # Perform Hands landmarks detection on the frame.
            frame, results = detectHandsLandmarks(frame, hands_videos, display=False)

        # Check if the hands landmarks in the frame are detected.
            if results.multi_hand_landmarks:
                # Count the number of fingers up of each hand in the frame.
                frame, fingers_statuses, count = countFingers(frame, results, display=False)
            # Display the frame.
            cv2.imshow('Fingers Counter', frame)

        # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
            k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed and break the loop.
            if (k == 27):
                break
        camera_video.release()
        cv2.destroyAllWindows()
    def write_text():
        camera_video = cv2.VideoCapture(0)
        camera_video.set(3, 1280)
        camera_video.set(4, 960)
        cv2.namedWindow('Selfie-Capturing System', cv2.WINDOW_NORMAL)
        num_of_frames = 5

    # Initialize a dictionary to store the counts of the consecutive frames with the hand gestures recognized.
        counter = {'V SIGN': 0, 'SPIDERMAN SIGN': 0, 'HIGH-FIVE SIGN': 0, 'NO': 0, 'PLAY': 0, 'YES': 0, 'IHATEYOU': 0}

    # Initialize a variable to store the captured image.
        captured_image = None

    # Iterate until the webcam is accessed successfully.
        while camera_video.isOpened():
            # Read a frame.
            ok, frame = camera_video.read()

        # Check if frame is not read properly then continue to the next iteration to read the next frame.
            if not ok:
                continue
            # Flip the frame horizontally for natural (selfie-view) visualization.
            frame = cv2.flip(frame, 1)

        # Get the height and width of the frame of the webcam video.
            frame_height, frame_width, _ = frame.shape
            frame, results = detectHandsLandmarks(frame, hands_videos, draw=False, display=False)

        # Check if the hands landmarks in the frame are detected.
            if results.multi_hand_landmarks:
                frame, fingers_statuses, count = countFingers(frame, results, draw=False, display=False)

            # Perform the hand gesture recognition on the hands in the frame.
                _, hands_gestures = recognizeGestures(frame, fingers_statuses, count, draw=False, display=False)
                if any(hand_gesture == "SPIDERMAN SIGN" for hand_gesture in hands_gestures.values()):
                    # Increment the count of consecutive frames with SPIDERMAN hand gesture recognized.
                    counter['SPIDERMAN SIGN'] += 1

                # Check if the counter is equal to the required number of consecutive frames.  
                    if counter['SPIDERMAN SIGN'] == num_of_frames:

                        print('ILOVEYOU')
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,
                                    'ILOVEYOU',
                                    (100, 150),
                                    font, 3,
                                    (0, 255, 255),
                                    5,
                                    cv2.LINE_4)
                        counter['SPIDERMAN SIGN'] = 0
                else:
                    # Update the counter value to zero. As we are counting the consective frames with SPIDERMAN hand gesture.
                    counter['SPIDERMAN SIGN'] = 0
                    if any(hand_gesture == "HIGH-FIVE SIGN" for hand_gesture in hands_gestures.values()):

                        counter['HIGH-FIVE SIGN'] += 1

                # Check if the counter is equal to the required number of consecutive frames.  
                        if counter['HIGH-FIVE SIGN'] == num_of_frames:
                             # Turn off the filter by updating the value of the filter status variable to False.
                            print('HELLO')
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame,
                                        'HELLO',
                                        (100, 150),
                                        font, 3,
                                        (0, 255, 255),
                                        5,
                                        cv2.LINE_4)
                            counter['HIGH-FIVE SIGN'] = 0
                    else:
                        counter['HIGH-FIVE SIGN'] = 0
                    if any(hand_gesture == "NO" for hand_gesture in hands_gestures.values()):
                        counter['NO'] += 1

                        # Check if the counter is equal to the required number of consecutive frames.  
                        if counter['NO'] == num_of_frames:
                             # Turn off the filter by updating the value of the filter status variable to False.
                            print('NO')
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame,
                                        'NO',
                                        (100, 150),
                                        font, 3,
                                        (0, 255, 255),
                                        5,
                                        cv2.LINE_4)

                    # Update the counter value to zero.
                            counter['NO'] = 0
                    else:
                        counter['NO'] = 0
                    if any(hand_gesture == "PLAY" for hand_gesture in hands_gestures.values()):
                        counter['PLAY'] += 1
                        if counter['PLAY'] == num_of_frames:
                            print('PLAY')
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame,
                                        'PLAY',
                                        (100, 150),
                                        font, 3,
                                        (0, 255, 255),
                                        5,
                                        cv2.LINE_4)

                    # Update the counter value to zero.
                            counter['PLAY'] = 0
                    else:
                        counter['PLAY'] = 0
                    if any(hand_gesture == "YES" for hand_gesture in hands_gestures.values()):
                        counter['YES'] += 1

                # Check if the counter is equal to the required number of consecutive frames.  
                        if counter['YES'] == num_of_frames:
                    # Turn off the filter by updating the value of the filter status variable to False.
                            print('YES')
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame,
                                        'YES',
                                        (100, 150),
                                        font, 3,
                                        (0, 255, 255),
                                        5,
                                        cv2.LINE_4)

                    # Update the counter value to zero.
                            counter['YES'] = 0
                    else:
                        counter['YES'] = 0
                    if any(hand_gesture == "IHATEYOU" for hand_gesture in hands_gestures.values()):
                        counter['IHATEYOU'] += 1
                        if counter['IHATEYOU'] == num_of_frames:
                        # Turn off the filter by updating the value of the filter status variable to False.
                            print('IHATEYOU')
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame,
                                        'IHATEYOU',
                                        (100, 150),
                                        font, 3,
                                        (0, 255, 255),
                                        5,
                                        cv2.LINE_4)

                    # Update the counter value to zero.
                            counter['IHATEYOU'] = 0
                    else:
                        counter['IHATEYOU'] = 0
            cv2.imshow('Selfie-Capturing System', frame)

        # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
            k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed and break the loop.
            if (k == 27):
                break
        camera_video.release()
        cv2.destroyAllWindows()


    







                

    import tkinter as tk
    parent = tk.Tk()
    parent.geometry("1000x300")
    parent.title("MP")
    parent.config(bg="blue")

    tk.Label(parent, text="REAL TIME FINGER COUNTER AND HAND GESTURE RECOGNITION", font="normal 20 bold", fg="black",
            bg="blue").pack(pady=20)

    frame = tk.Frame(parent)
    frame.config(bg="blue")
    frame.pack()

    text_disp = tk.Button(frame,
                      text="sign detection",
                      command=write_text, bg="red")

# text_disp.pack(side=tk.LEFT)
    text_disp.pack(side=tk.LEFT, padx=10)

    exit_button = tk.Button(frame,
                        text="Finger counter", bg="red", command=qu)

# exit_button.pack(side=tk.RIGHT)
    exit_button.pack(side=tk.LEFT, padx=10)
    button3 = tk.Button(frame,
                    text="hand gesture", command=volume, bg="red")
# button3.pack(side=tk.BOTTOM)
    button3.pack(side=tk.LEFT, padx=10)
    parent.mainloop()

    
    
    
 
 
def delete_password_not_recognised():
    password_not_recog_screen.destroy()
 
 
def delete_user_not_found_screen():
    user_not_found_screen.destroy()
 
 
# Designing Main(first) window
 
def main_account_screen():
    
    
    global main_screen
    main_screen = Tk()
    main_screen.geometry("300x250")
    main_screen.title("Account Login")
    Label(text="Select Your Choice", bg="blue", width="300", height="2", font=("Calibri", 13)).pack()
    Label(text="").pack()
    Button(text="Login", height="2", width="30", command = login).pack()
    Label(text="").pack()
    Button(text="Register", height="2", width="30", command=register).pack()
 
    main_screen.mainloop()
 
 
main_account_screen()