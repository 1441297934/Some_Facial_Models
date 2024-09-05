from numpy import *
import cv2 as cv
import mediapipe as mp
import time

# variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
# constants
FONTS = cv.FONT_HERSHEY_COMPLEX

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

map_face_mesh = mp.solutions.face_mesh

def rotation_matrix_to_angles(rotation_matrix):
    """
    Calculate Euler angles from rotation matrix.
    :param rotation_matrix: A 3*3 matrix with the following structure
    [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
    [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
    [  -Siny             CosySinx                   Cosy*Cosx         ]
    :return: Angles in degrees for each axis
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi

# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


# Euclidean distance
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)

    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio  # 眨眼比率

# Check if blinked
def checkBlink(ratio, ave,THRESHOLD,CLOSED_EYES_FRAME):
    global CEF_COUNTER, TOTAL_BLINKS
    if ratio > ave + THRESHOLD:
        CEF_COUNTER += 1
    else:
        if CEF_COUNTER > CLOSED_EYES_FRAME:
            TOTAL_BLINKS += 1
            CEF_COUNTER = 0

def CheckBlink(blinktimes=5,camera = cv.VideoCapture(0),THRESHOLD = 0.6,CLOSED_EYES_FRAME = 1 ):
    global frame_counter
    ratiolist = [4, 4, 4, 4, 4]
    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # starting time here
        start_time = time.time()
        # starting Video loop here.
        flag=True
        while flag:
            frame_counter += 1  # frame counter
            ret, frame = camera.read()  # getting frame from camera
            if not ret:
                break  # no more frames break
            #  resizing frame

            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                # Display blink ratio
                temp=round(ratio, 2)

                ave = mean(ratiolist)
                # Check if blinked
                checkBlink(ratio, ave,THRESHOLD,CLOSED_EYES_FRAME)
                ratiolist.pop(0)
                ratiolist.append(ratio)
                # Display total blinks

                print(TOTAL_BLINKS)
                if TOTAL_BLINKS >= blinktimes:
                    flag=False

            # calculating frame per seconds FPS
            end_time = time.time() - start_time
            fps = frame_counter / end_time
            cv.imshow('frame', frame)
            key = cv.waitKey(2)
            if key == ord('q') or key == ord('Q'):
                break
        cv.destroyAllWindows()
        camera.release()
        return True

import math
import cv2
import mediapipe as mp
import numpy as np


def rotation_matrix_to_angles(rotation_matrix):
    """
    Calculate Euler angles from rotation matrix.
    :param rotation_matrix: A 3*3 matrix with the following structure
    [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
    [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
    [  -Siny             CosySinx                   Cosy*Cosx         ]
    :return: Angles in degrees for each axis
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi


def turnright(cap = cv2.VideoCapture(0)):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    flag=0
    exit=True
    while exit and cap.isOpened():
        success, image = cap.read()

        # Convert the color space from BGR to RGB and get Mediapipe results
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Convert the color space from RGB to BGR to display well with Opencv
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)

        h, w, _ = image.shape
        face_coordination_in_image = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_coordination_in_image.append([x, y])

                face_coordination_in_image = np.array(face_coordination_in_image,
                                                      dtype=np.float64)

                # The camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                       [0, focal_length, h / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Use solvePnP function to get rotation vector
                success, rotation_vec, transition_vec = cv2.solvePnP(
                    face_coordination_in_real_world, face_coordination_in_image,
                    cam_matrix, dist_matrix)

                # Use Rodrigues function to convert rotation vector to matrix
                rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

                result = rotation_matrix_to_angles(rotation_matrix)
                if flag==0:
                    iniyaw=result[1]
                    flag=1
                print(abs(iniyaw-result[1]))
                if result[1]<-30 and abs(iniyaw-result[1])>=40:
                    exit=False
                for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                    k, v = info
                    text = f'{k}: {int(v)}'
                    cv2.putText(image, text, (20, i * 30 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

        cv2.imshow('Head Pose Angles', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()


def turnleft(cap = cv2.VideoCapture(0)):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    flag=0
    exit=True
    while exit and cap.isOpened():
        success, image = cap.read()

        # Convert the color space from BGR to RGB and get Mediapipe results
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Convert the color space from RGB to BGR to display well with Opencv
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)

        h, w, _ = image.shape
        face_coordination_in_image = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_coordination_in_image.append([x, y])

                face_coordination_in_image = np.array(face_coordination_in_image,
                                                      dtype=np.float64)

                # The camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                       [0, focal_length, h / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Use solvePnP function to get rotation vector
                success, rotation_vec, transition_vec = cv2.solvePnP(
                    face_coordination_in_real_world, face_coordination_in_image,
                    cam_matrix, dist_matrix)

                # Use Rodrigues function to convert rotation vector to matrix
                rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

                result = rotation_matrix_to_angles(rotation_matrix)
                if flag==0:
                    iniyaw=result[1]
                    flag=1
                print(abs(iniyaw-result[1]))
                if result[1]>30 and abs(iniyaw-result[1])>=40:
                    exit=False
                for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                    k, v = info
                    text = f'{k}: {int(v)}'
                    cv2.putText(image, text, (20, i * 30 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

        cv2.imshow('Head Pose Angles', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

def adjust(cap = cv2.VideoCapture(0)):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    exit=True
    while exit and cap.isOpened():
        success, image = cap.read()

        # Convert the color space from BGR to RGB and get Mediapipe results
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Convert the color space from RGB to BGR to display well with Opencv
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)

        h, w, _ = image.shape
        face_coordination_in_image = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_coordination_in_image.append([x, y])

                face_coordination_in_image = np.array(face_coordination_in_image,
                                                      dtype=np.float64)

                # The camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                       [0, focal_length, h / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Use solvePnP function to get rotation vector
                success, rotation_vec, transition_vec = cv2.solvePnP(
                    face_coordination_in_real_world, face_coordination_in_image,
                    cam_matrix, dist_matrix)

                # Use Rodrigues function to convert rotation vector to matrix
                rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

                result = rotation_matrix_to_angles(rotation_matrix)
                if abs(result[0])<=15 and abs(result[1])<=15 and abs(result[2])<=15:
                    exit=False
                for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                    k, v = info
                    text = f'{k}: {int(v)}'
                    cv2.putText(image, text, (20, i * 30 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

        cv2.imshow('Head Pose Angles', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

def nod(cap = cv2.VideoCapture(0)):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    exit = True
    while exit and cap.isOpened():
        success, image = cap.read()

        # Convert the color space from BGR to RGB and get Mediapipe results
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Convert the color space from RGB to BGR to display well with Opencv
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)

        h, w, _ = image.shape
        face_coordination_in_image = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_coordination_in_image.append([x, y])

                face_coordination_in_image = np.array(face_coordination_in_image,
                                                      dtype=np.float64)

                # The camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                       [0, focal_length, h / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Use solvePnP function to get rotation vector
                success, rotation_vec, transition_vec = cv2.solvePnP(
                    face_coordination_in_real_world, face_coordination_in_image,
                    cam_matrix, dist_matrix)

                # Use Rodrigues function to convert rotation vector to matrix
                rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

                result = rotation_matrix_to_angles(rotation_matrix)
                if result[0] <-20:
                    exit = False
                for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                    k, v = info
                    text = f'{k}: {int(v)}'
                    cv2.putText(image, text, (20, i * 30 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

        cv2.imshow('Head Pose Angles', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    print(1)
    exit = True
    while exit and cap.isOpened():
        success, image = cap.read()

        # Convert the color space from BGR to RGB and get Mediapipe results
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Convert the color space from RGB to BGR to display well with Opencv
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)

        h, w, _ = image.shape
        face_coordination_in_image = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_coordination_in_image.append([x, y])

                face_coordination_in_image = np.array(face_coordination_in_image,
                                                      dtype=np.float64)

                # The camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                       [0, focal_length, h / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Use solvePnP function to get rotation vector
                success, rotation_vec, transition_vec = cv2.solvePnP(
                    face_coordination_in_real_world, face_coordination_in_image,
                    cam_matrix, dist_matrix)

                # Use Rodrigues function to convert rotation vector to matrix
                rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

                result = rotation_matrix_to_angles(rotation_matrix)
                if result[0] > 0 :
                    exit = False
                for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                    k, v = info
                    text = f'{k}: {int(v)}'
                    cv2.putText(image, text, (20, i * 30 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

        cv2.imshow('Head Pose Angles', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()


if __name__ == "__main__":
    CheckBlink()#默认五次眼def CheckBlink(blinktimes=5,camera = cv.VideoCapture(0),THRESHOLD = 0.6,CLOSED_EYES_FRAME = 1 )
    adjust()#恢复正视
    turnleft()#左转头
    turnright()#右转头
    nod()#点头

