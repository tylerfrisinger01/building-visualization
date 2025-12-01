import cv2
import mediapipe as mp
import threading
from ursina import *

# ---------- Shared hand state between threads ----------
hand_state = {
    "rot_y": 0.0,        # rotation in degrees
    "height_factor": 1.0,  # multiplier for building height
    "pinch": False,      # whether thumb + index are close
}

# ---------- Hand tracking loop (MediaPipe + OpenCV) ----------
def hand_tracking_loop():
    print("[hand] starting hand_tracking_loop")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # On mac sometimes CAP_AVFOUNDATION is more reliable; if that fails, try just 0
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("[hand] ERROR: Could not open camera")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[hand] WARNING: Failed to read frame")
            continue

        # Process frame (no window)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Index fingertip
            idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_norm, y_norm = idx_tip.x, idx_tip.y

            rot_y = (x_norm - 0.5) * 360.0
            inv_y = 1.0 - y_norm
            height_factor = 0.5 + inv_y

            # Thumb–index distance for pinch
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            dx = (idx_tip.x - thumb_tip.x) * w
            dy = (idx_tip.y - thumb_tip.y) * h
            dist = (dx * dx + dy * dy) ** 0.5
            pinch = dist < 40

            hand_state["rot_y"] = rot_y
            hand_state["height_factor"] = height_factor
            hand_state["pinch"] = pinch

            # Debug info so you KNOW it’s tracking
            print(f"[hand] rot_y={rot_y:.1f}, height_factor={height_factor:.2f}, pinch={pinch}")

        # ⚠️ No cv2.imshow and no cv2.waitKey here – that’s what was crashing

    cap.release()
    print("[hand] camera released, thread exiting")


# ---------- 3D Visualization (Ursina) ----------
app = Ursina()

# Simple "building" as a tall cube
BASE_HEIGHT = 5
building = Entity(
    model="cube",
    color=color.gray,
    scale=(2, BASE_HEIGHT, 2),  # width, height, depth
)

camera.position = (0, 10, -25)
camera.look_at(building)

# Some simple textures (Ursina has a few built-ins; you can also load your own)
textures = [
    None,          # plain color
    "white_cube",  # grid-like texture
    "brick",       # if present; otherwise replace with your own path
]
texture_index = 0
last_pinch = False

def update():
    global texture_index, last_pinch

    # Read from shared hand_state
    rot_y = hand_state["rot_y"]
    height_factor = hand_state["height_factor"]
    pinch = hand_state["pinch"]

    # Apply rotation
    building.rotation_y = rot_y

    # Clamp height factor so it doesn't get too crazy
    height_factor = max(0.5, min(2.0, height_factor))
    building.scale_y = BASE_HEIGHT * height_factor

    # Detect "pinch just started" (rising edge)
    if pinch and not last_pinch:
        texture_index = (texture_index + 1) % len(textures)
        building.texture = textures[texture_index]
        print(f"Texture changed to index {texture_index}: {textures[texture_index]}")

    last_pinch = pinch

# ---------- Start everything ----------
# Run hand tracking in background thread
tracking_thread = threading.Thread(target=hand_tracking_loop, daemon=True)
tracking_thread.start()

app.run()
