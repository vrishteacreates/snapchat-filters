
import cv2
import mediapipe as mp
import numpy as np
import math
import time


# ==========================================
# 1. INITIALIZE MEDIAPIPE
# ==========================================

mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# 2. CREATE FILTER IMAGES
# ==========================================

def create_glasses():

    img = np.zeros((220, 600, 4), dtype=np.uint8)

    cv2.ellipse(img, (165, 110), (130, 75),
                0, 0, 360, (20, 20, 20, 220), -1)

    cv2.ellipse(img, (435, 110), (130, 75),
                0, 0, 360, (20, 20, 20, 220), -1)

    cv2.ellipse(img, (165, 110), (130, 75),
                0, 0, 360, (0, 0, 0, 255), 14)

    cv2.ellipse(img, (435, 110), (130, 75),
                0, 0, 360, (0, 0, 0, 255), 14)

    cv2.line(img, (290, 100), (310, 100),
             (0, 0, 0, 255), 14)

    cv2.line(img, (35, 90), (0, 65),
             (0, 0, 0, 255), 13)

    cv2.line(img, (565, 90), (600, 65),
             (0, 0, 0, 255), 13)

    return img


def create_moustache():

    img = np.zeros((120, 300, 4), dtype=np.uint8)

    cv2.ellipse(img, (105, 55), (85, 35),
                0, 0, 360, (25, 20, 20, 255), -1)

    cv2.ellipse(img, (195, 55), (85, 35),
                0, 0, 360, (25, 20, 20, 255), -1)

    return img


def create_crown():

    img = np.zeros((250, 500, 4), dtype=np.uint8)

    points = np.array([
        [45, 190],
        [20, 65],
        [135, 130],
        [250, 15],
        [365, 130],
        [480, 65],
        [455, 190]
    ], np.int32)

    cv2.fillPoly(img, [points], (0, 190, 255, 255))

    cv2.rectangle(img, (45, 185), (455, 220),
                  (0, 150, 230, 255), -1)

    for x, y in [(20, 65), (250, 15), (480, 65)]:
        cv2.circle(img, (x, y), 17, (0, 0, 255, 255), -1)

    return img


def create_cat():

    img = np.zeros((230, 450, 4), dtype=np.uint8)

    # Left ear
    left = np.array([
        [35, 125], [75, 10], [185, 125]
    ], np.int32)

    cv2.fillPoly(img, [left], (30, 150, 240, 255))

    # Right ear
    right = np.array([
        [265, 125], [375, 10], [415, 125]
    ], np.int32)

    cv2.fillPoly(img, [right], (30, 150, 240, 255))

    # Inner ears
    cv2.fillPoly(
        img,
        [np.array([[75, 100], [82, 40], [145, 105]], np.int32)],
        (180, 120, 220, 255)
    )

    cv2.fillPoly(
        img,
        [np.array([[305, 105], [368, 40], [375, 100]], np.int32)],
        (180, 120, 220, 255)
    )

    # Whiskers
    for y in [150, 175]:
        cv2.line(img, (120, y), (15, y - 20),
                 (255, 255, 255, 255), 5)

        cv2.line(img, (330, y), (435, y - 20),
                 (255, 255, 255, 255), 5)

    # Pink nose
    nose = np.array([
        [205, 155], [245, 155], [225, 180]
    ], np.int32)

    cv2.fillPoly(img, [nose], (180, 100, 255, 255))

    return img


def create_bunny():

    img = np.zeros((350, 450, 4), dtype=np.uint8)

    # Left ear
    cv2.ellipse(img, (130, 105), (55, 115),
                -15, 0, 360, (220, 220, 255, 255), -1)

    # Right ear
    cv2.ellipse(img, (320, 105), (55, 115),
                15, 0, 360, (220, 220, 255, 255), -1)

    # Inner ears
    cv2.ellipse(img, (130, 105), (25, 85),
                -15, 0, 360, (150, 130, 255, 255), -1)

    cv2.ellipse(img, (320, 105), (25, 85),
                15, 0, 360, (150, 130, 255, 255), -1)

    return img


def create_flower_crown():

    img = np.zeros((170, 500, 4), dtype=np.uint8)

    # Curved flower arrangement
    for x in range(35, 480, 55):

        y = int(115 - 45 * math.sin(
            math.pi * x / 500
        ))

        cv2.circle(img, (x, y), 13,
                   (255, 150, 255, 255), -1)

        cv2.circle(img, (x + 10, y - 12), 13,
                   (255, 180, 200, 255), -1)

        cv2.circle(img, (x - 10, y - 12), 13,
                   (255, 120, 220, 255), -1)

        cv2.circle(img, (x, y - 10), 7,
                   (0, 220, 255, 255), -1)

    # Green leaves
    for x in range(55, 470, 75):

        y = int(125 - 45 * math.sin(
            math.pi * x / 500
        ))

        cv2.ellipse(img, (x, y + 20), (15, 6),
                    -35, 0, 360, (40, 180, 40, 255), -1)

    return img


# ==========================================
# 3. INITIALIZE ALL FILTERS
# ==========================================

filters = {
    "GLASSES": create_glasses(),
    "MOUSTACHE": create_moustache(),
    "CROWN": create_crown(),
    "CAT": create_cat(),
    "BUNNY": create_bunny(),
    "FLOWER": create_flower_crown()
}


# ==========================================
# 4. ROTATE FILTER
# ==========================================

def rotate_image(image, angle):

    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center, angle, 1.0
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )


# ==========================================
# 5. OVERLAY FILTER ON WEBCAM
# ==========================================

def overlay_filter(background, overlay, x, y):

    h, w = overlay.shape[:2]

    bg_h, bg_w = background.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)

    x2 = min(x + w, bg_w)
    y2 = min(y + h, bg_h)

    if x1 >= x2 or y1 >= y2:
        return background

    overlay_crop = overlay[
        y1-y:y2-y,
        x1-x:x2-x
    ]

    background_crop = background[y1:y2, x1:x2]

    alpha = overlay_crop[:, :, 3:4] / 255.0

    blended = (
        alpha * overlay_crop[:, :, :3]
        + (1 - alpha) * background_crop
    )

    background[y1:y2, x1:x2] = blended.astype(np.uint8)

    return background


# ==========================================
# 6. APPLY FILTER
# ==========================================

def apply_filter(frame, landmarks, selected):

    h, w = frame.shape[:2]

    # Face reference points
    forehead = landmarks[10]

    left_face = landmarks[234]
    right_face = landmarks[454]

    face_width = math.hypot(
        (right_face.x - left_face.x) * w,
        (right_face.y - left_face.y) * h
    )

    face_angle = math.degrees(
        math.atan2(
            (right_face.y - left_face.y) * h,
            (right_face.x - left_face.x) * w
        )
    )

    fx = int(forehead.x * w)
    fy = int(forehead.y * h)

    # --------------------------------------
    # GLASSES
    # --------------------------------------

    if selected == "GLASSES":

        left = landmarks[33]
        right = landmarks[263]

        lx, ly = int(left.x*w), int(left.y*h)
        rx, ry = int(right.x*w), int(right.y*h)

        eye_distance = math.hypot(rx-lx, ry-ly)

        width = max(1, int(eye_distance * 2.2))
        height = int(width * 220 / 600)

        img = cv2.resize(filters["GLASSES"], (width, height))

        angle = math.degrees(
            math.atan2(ry-ly, rx-lx)
        )

        img = rotate_image(img, angle)

        cx = (lx+rx)//2
        cy = (ly+ry)//2

        frame = overlay_filter(
            frame, img,
            cx-img.shape[1]//2,
            cy-img.shape[0]//2
        )

    # --------------------------------------
    # MOUSTACHE
    # --------------------------------------

    elif selected == "MOUSTACHE":

        left = landmarks[61]
        right = landmarks[291]
        lip = landmarks[13]

        lx, ly = int(left.x*w), int(left.y*h)
        rx, ry = int(right.x*w), int(right.y*h)

        width = max(1, int(math.hypot(rx-lx, ry-ly)*1.5))
        height = int(width*120/300)

        img = cv2.resize(filters["MOUSTACHE"], (width, height))

        angle = math.degrees(
            math.atan2(ry-ly, rx-lx)
        )

        img = rotate_image(img, angle)

        cx = (lx+rx)//2
        cy = int(lip.y*h)

        frame = overlay_filter(
            frame, img,
            cx-img.shape[1]//2,
            cy-img.shape[0]//2
        )

    # --------------------------------------
    # CROWN
    # --------------------------------------

    elif selected == "CROWN":

        width = max(1, int(face_width * 1.35))
        height = int(width*250/500)

        img = cv2.resize(filters["CROWN"], (width, height))
        img = rotate_image(img, face_angle)

        frame = overlay_filter(
            frame, img,
            fx-img.shape[1]//2,
            fy-img.shape[0]+int(height*0.25)
        )

    # --------------------------------------
    # CAT
    # --------------------------------------

    elif selected == "CAT":

        width = max(1, int(face_width * 1.25))
        height = int(width*230/450)

        img = cv2.resize(filters["CAT"], (width, height))
        img = rotate_image(img, face_angle)

        frame = overlay_filter(
            frame, img,
            fx-img.shape[1]//2,
            fy-int(height*0.65)
        )

    # --------------------------------------
    # BUNNY
    # --------------------------------------

    elif selected == "BUNNY":

        width = max(1, int(face_width * 1.2))
        height = int(width*350/450)

        img = cv2.resize(filters["BUNNY"], (width, height))
        img = rotate_image(img, face_angle)

        frame = overlay_filter(
            frame, img,
            fx-img.shape[1]//2,
            fy-int(height*0.8)
        )

    # --------------------------------------
    # FLOWER CROWN
    # --------------------------------------

    elif selected == "FLOWER":

        width = max(1, int(face_width * 1.4))
        height = int(width*170/500)

        img = cv2.resize(filters["FLOWER"], (width, height))
        img = rotate_image(img, face_angle)

        frame = overlay_filter(
            frame, img,
            fx-img.shape[1]//2,
            fy-int(height*0.65)
        )

    return frame


# ==========================================
# 7. BUTTON SETTINGS
# ==========================================

filter_state = {
    "selected": "GLASSES",
    "photo": False
}

buttons = [
    ("GLASSES", 10, 10),
    ("MOUSTACHE", 165, 10),
    ("CROWN", 320, 10),
    ("CAT", 475, 10),
    ("BUNNY", 10, 60),
    ("FLOWER", 165, 60),
    ("OFF", 320, 60),
    ("PHOTO", 475, 60)
]


def mouse_callback(event, x, y, flags, param):

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for name, bx, by in buttons:

        if bx <= x <= bx+145 and by <= y <= by+40:

            if name == "PHOTO":
                param["photo"] = True

            else:
                param["selected"] = name

            break


# ==========================================
# 8. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened!")

else:

    window_name = "MY SNAPCHAT FILTERS"

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        filter_state
    )

    # ======================================
    # 9. MAIN LOOP
    # ======================================

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        # Apply selected filter
        if (
            results.multi_face_landmarks
            and filter_state["selected"] != "OFF"
        ):

            for face in results.multi_face_landmarks:

                frame = apply_filter(
                    frame,
                    face.landmark,
                    filter_state["selected"]
                )

        # ==================================
        # 10. DISPLAY BUTTONS
        # ==================================

        for name, bx, by in buttons:

            if name == filter_state["selected"]:

                color = (40, 180, 40)

            elif name == "PHOTO":

                color = (180, 80, 180)

            else:

                color = (80, 80, 80)

            cv2.rectangle(
                frame,
                (bx, by),
                (bx+145, by+40),
                color,
                -1
            )

            cv2.putText(
                frame,
                name,
                (bx+12, by+27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # Status
        cv2.putText(
            frame,
            "Selected: " + filter_state["selected"],
            (10, frame.shape[0]-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ==================================
        # 11. TAKE PHOTO
        # ==================================

        if filter_state["photo"]:

            filename = "snap_" + str(int(time.time())) + ".jpg"

            cv2.imwrite(filename, frame)

            print("Photo saved:", filename)

            filter_state["photo"] = False

        # Display
        cv2.imshow(window_name, frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ==========================================
# 12. CLOSE APPLICATION
# ==========================================

cap.release()

face_mesh.close()

cv2.destroyAllWindows()

print("Snapchat Filters App Closed!")
