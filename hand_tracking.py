# hand_tracking_v10.3_StandardSlider_FistGestures.py
# - Auto-starts tracking on launch if camera is available.
# - Allows live updates of control zones via sliders during tracking.
# - Uses direct reading of Tkinter variables in the tracking thread.
# - Reverted to standard slider behavior (only drag works, click on track does nothing).
# - Uses 'command' option for live label update during drag.
# - ADDED: Fist detection for left and right hands, sending L_FIST/R_FIST via UDP.

import cv2
import mediapipe as mp
import socket
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import queue
from PIL import Image, ImageTk
import sys
import time
import traceback
import math
import json
import os
import numpy as np

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
CONFIG_FILE = "config.json"

# --- Default Control Box Settings ---
DEFAULT_RIGHT_CB = {"x": 0.6, "y": 0.2, "w": 0.3, "h": 0.6}
DEFAULT_LEFT_CB = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.6}

# --- Common Resolutions ---
COMMON_RESOLUTIONS = [
    (1920, 1080, "16:9"), (1280, 720, "16:9"), (1024, 768, "4:3"),
    (800, 600, "4:3"), (640, 480, "4:3"), (320, 240, "4:3")
]
DEFAULT_WIDTH, DEFAULT_HEIGHT = 640, 480
DEFAULT_RESOLUTION_STRING = f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT} (Default)"

# --- Global Variables / State Management ---
video_thread = None
stop_event = threading.Event()
frame_queue = queue.Queue(maxsize=2) # Queue for video frames for GUI
sock = None

# --- NEW: Global States for Fist Detection ---
left_hand_fist_state = False
right_hand_fist_state = False
# Threshold for fist detection. This value is CRITICAL and needs tuning.
# It represents the normalized distance between a fingertip and its MCP joint.
# Smaller value = tighter fist required.
# Print distances in is_fist() to help tune this.
FINGER_CLOSE_THRESHOLD = 0.055 # EXAMPLE - ADJUST THIS!

# --- Splash Screen Function ---
def create_splash():
    splash_root = tk.Tk()
    splash_root.title("Loading...")
    splash_root.overrideredirect(True)
    screen_width = splash_root.winfo_screenwidth()
    screen_height = splash_root.winfo_screenheight()
    splash_width = 350; splash_height = 150
    x_pos = (screen_width // 2) - (splash_width // 2)
    y_pos = (screen_height // 2) - (splash_height // 2)
    splash_root.geometry(f'{splash_width}x{splash_height}+{x_pos}+{y_pos}')
    splash_root.configure(bg='lightblue')
    splash_label = tk.Label(splash_root, text="Initializing Hand Tracker...\nPlease Wait.", font=("Helvetica", 14, "bold"), bg='lightblue', fg='navy')
    splash_label.pack(pady=40, expand=True)
    splash_root.attributes('-topmost', True)
    splash_root.update(); splash_root.update_idletasks()
    return splash_root

# --- Camera and Resolution Utilities ---
def find_available_cameras(max_cameras_to_check=5):
    available_cameras = []
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            available_cameras.append((f"Camera {i}", i))
            cap.release()
        else:
            if cap is not None: cap.release()
    return available_cameras

def get_supported_resolutions(camera_index):
    supported = []
    cap_test = None
    try:
        cap_test = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap_test or not cap_test.isOpened():
            print(f"[PY CAM] Error: Could not open camera {camera_index} for probing.")
            return supported
        for width, height, aspect in COMMON_RESOLUTIONS:
            cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            time.sleep(0.05)
            actual_width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if abs(actual_width - width) < 20 and abs(actual_height - height) < 20:
                 res_tuple = (width, height)
                 if res_tuple not in [(w, h) for w, h, a, f in supported]:
                     formatted = f"{width}x{height} ({aspect})"
                     supported.append((width, height, aspect, formatted))
    except Exception as e:
        print(f"[PY CAM] Exception during resolution probing for camera {camera_index}: {e}")
    finally:
        if cap_test is not None and cap_test.isOpened(): cap_test.release()
    supported.sort(key=lambda x: x[0], reverse=True)
    return supported

# --- Helper Functions for Hand Analysis ---
def get_landmark_coords(landmarks, landmark_id, frame_width, frame_height):
    # This function wasn't directly used in your original processing loop for the main logic,
    # but it's a useful utility if needed elsewhere. Keeping it.
    lm = landmarks.landmark[landmark_id]
    return int(lm.x * frame_width), int(lm.y * frame_height), lm.z

def calculate_distance(p1, p2):
    # Calculates Euclidean distance between two MediaPipe Landmark objects
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# --- NEW: Helper function to check for fist ---
def is_fist(hand_landmarks, handedness_label):
    """
    Checks if a hand is likely in a fist position.
    Compares distances of fingertips to their corresponding MCP joints.
    """
    if not hand_landmarks:
        return False

    tip_ids = [ # Order: Thumb, Index, Middle, Ring, Pinky
        mp.solutions.hands.HandLandmark.THUMB_TIP,
        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mp.solutions.hands.HandLandmark.PINKY_TIP
    ]
    mcp_ids = [ # Metacarpophalangeal joints (knuckles at base of fingers)
        mp.solutions.hands.HandLandmark.THUMB_IP, # For thumb, IP (interphalangeal) is closer to "base" when fisted
        mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp.solutions.hands.HandLandmark.RING_FINGER_MCP,
        mp.solutions.hands.HandLandmark.PINKY_MCP
    ]

    closed_fingers = 0
    # We check fingers from Index to Pinky (indices 1 to 4 in tip_ids/mcp_ids)
    # Thumb is a bit more complex to determine "closed" status reliably with this simple method,
    # so we rely on the other four fingers.
    for i in range(1, 5):
        tip = hand_landmarks.landmark[tip_ids[i]]
        mcp = hand_landmarks.landmark[mcp_ids[i]]
        distance = calculate_distance(tip, mcp)

        # --- UNCOMMENT FOR TUNING FINGER_CLOSE_THRESHOLD ---
        if i == 1: # Print only for index finger to reduce spam
            print(f"{handedness_label} Index Tip-MCP Dist: {distance:.4f}")
        # --- END TUNING ---

        if distance < FINGER_CLOSE_THRESHOLD:
            closed_fingers += 1

    # If at least 3 of the main four fingers are closed, we consider it a fist.
    # You can adjust this condition (e.g., `closed_fingers == 4` for a very tight fist)
    if closed_fingers >= 3:
        return True
    return False

# --- Video Processing Thread ---
def video_processing_loop(settings, frame_q, stop_flag):
    global sock
    # --- NEW: Access global fist states ---
    global left_hand_fist_state, right_hand_fist_state
    # --- END NEW ---

    print("[PY THREAD] Video thread starting...")
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1, # Your original setting
        static_image_mode=False,
        max_num_hands=2
    )

    cap = cv2.VideoCapture(settings['camera_index'], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[PY THREAD ERROR] Cannot open camera {settings['camera_index']}") # Changed from raise IOError
        stop_flag.set() # Signal GUI to stop
        return # Exit thread

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
    time.sleep(0.2) # Give camera time to set resolution
    frame_w_actual = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Get actual width
    frame_h_actual = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get actual height
    print(f"[PY THREAD] Camera requested {settings['width']}x{settings['height']}, got {frame_w_actual}x{frame_h_actual}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    capture_queue = queue.Queue(maxsize=1)
    def capture_loop():
        while not stop_flag.is_set():
            ret, raw_frame = cap.read()
            if not ret:
                time.sleep(0.01) # Brief pause if read fails
                continue
            if capture_queue.full():
                try: capture_queue.get_nowait()
                except queue.Empty: pass
            capture_queue.put(raw_frame)
    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    cap_thread.start()
    print("[PY THREAD] Capture thread started.")

    try:
        frame_count = 0
        # Use copies of defaults for last known values to avoid modifying originals
        last_left_cb_values = DEFAULT_LEFT_CB.copy()
        last_right_cb_values = DEFAULT_RIGHT_CB.copy()

        while not stop_flag.is_set():
            try:
                frame = capture_queue.get(timeout=1.0) # Wait up to 1s for a frame
            except queue.Empty:
                # print("[PY THREAD] Capture queue empty, continuing...") # Can be noisy
                continue # No frame available, try again

            frame = cv2.flip(frame, 1) # Flip horizontally

            # Read control box Tkinter variables
            try:
                left_cb = {
                    'x': settings['left_x_var'].get(), 'y': settings['left_y_var'].get(),
                    'w': settings['left_w_var'].get(), 'h': settings['left_h_var'].get()
                }
                right_cb = {
                    'x': settings['right_x_var'].get(), 'y': settings['right_y_var'].get(),
                    'w': settings['right_w_var'].get(), 'h': settings['right_h_var'].get()
                }
                last_left_cb_values = left_cb.copy()
                last_right_cb_values = right_cb.copy()
            except Exception: # If Tkinter vars are gone (e.g. app closing)
                left_cb = last_left_cb_values.copy()
                right_cb = last_right_cb_values.copy()

            # Convert normalized control box coords to pixel coords for drawing
            # Use actual frame dimensions obtained after setting resolution
            current_frame_w_from_shape = frame.shape[1]
            current_frame_h_from_shape = frame.shape[0]

            left_px = {'x': int(left_cb['x'] * current_frame_w_from_shape), 'y': int(left_cb['y'] * current_frame_h_from_shape),
                       'w': int(left_cb['w'] * current_frame_w_from_shape), 'h': int(left_cb['h'] * current_frame_h_from_shape)}
            right_px = {'x': int(right_cb['x'] * current_frame_w_from_shape), 'y': int(right_cb['y'] * current_frame_h_from_shape),
                        'w': int(right_cb['w'] * current_frame_w_from_shape), 'h': int(right_cb['h'] * current_frame_h_from_shape)}

            # Draw control zones
            cv2.rectangle(frame, (left_px['x'], left_px['y']),
                          (left_px['x'] + left_px['w'], left_px['y'] + left_px['h']), (0, 0, 255), 2) # Red for Left
            cv2.rectangle(frame, (right_px['x'], right_px['y']),
                          (right_px['x'] + right_px['w'], right_px['y'] + right_px['h']), (255, 255, 0), 2) # Cyan for Right

            # Hand detection
            rgb_frame_for_mediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame_for_mediapipe.flags.writeable = False
            results = hands.process(rgb_frame_for_mediapipe)
            # frame.flags.writeable = True # Drawing will happen on 'frame', not 'rgb_frame_for_mediapipe'

            # --- NEW: Reset current frame detection flags ---
            current_frame_left_fist_detected = False
            current_frame_right_fist_detected = False
            # --- END NEW ---

            if results.multi_hand_landmarks:
                for idx, hand_landmarks_for_one_hand in enumerate(results.multi_hand_landmarks):
                    handedness_label = results.multi_handedness[idx].classification[0].label
                    mp_drawing.draw_landmarks(frame, hand_landmarks_for_one_hand, mp_hands.HAND_CONNECTIONS)

                    # --- NEW: FIST GESTURE DETECTION ---
                    is_current_hand_a_fist = is_fist(hand_landmarks_for_one_hand, handedness_label)
                    # --- END NEW ---

                    data_to_send_udp = None # UDP message to send for this hand

                    if handedness_label == "Left":
                        if is_current_hand_a_fist:
                            current_frame_left_fist_detected = True
                            # Draw text if it's currently a fist THIS FRAME
                            cv2.putText(frame, "L_FIST (Active)", (left_px['x'], left_px['y'] - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 100), 2)
                            if not left_hand_fist_state: # If it wasn't a fist before, now it is (for UDP)
                                data_to_send_udp = "L_FIST"
                                left_hand_fist_state = True

                        # Original LVAL logic
                        tip_coords = hand_landmarks_for_one_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        wx, wy = tip_coords.x, tip_coords.y
                        if left_cb['x'] <= wx < left_cb['x'] + left_cb['w'] and \
                           left_cb['y'] <= wy < left_cb['y'] + left_cb['h']:
                            rel_y = (wy - left_cb['y']) / left_cb['h'] if left_cb['h'] > 0 else 0.5
                            val = max(0.0, min(1.0, 1.0 - rel_y))
                            if data_to_send_udp is None: # Only send LVAL if no L_FIST
                                data_to_send_udp = f"LVAL:{val:.4f}"
                            cv2.putText(frame, f"L Val: {val:.2f}", (left_px['x'], left_px['y'] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    elif handedness_label == "Right":
                        if is_current_hand_a_fist:
                            current_frame_right_fist_detected = True
                            # Draw text if it's currently a fist THIS FRAME
                            cv2.putText(frame, "R_FIST (Active)", (right_px['x'], right_px['y'] - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
                            if not right_hand_fist_state: # If it wasn't a fist before, now it is (for UDP)
                                data_to_send_udp = "R_FIST"
                                right_hand_fist_state = True

                        # Original RPOS logic
                        tip_coords = hand_landmarks_for_one_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        wx, wy, wz = tip_coords.x, tip_coords.y, tip_coords.z
                        if right_cb['x'] <= wx < right_cb['x'] + right_cb['w'] and \
                           right_cb['y'] <= wy < right_cb['y'] + right_cb['h']:
                            rx = max(0.0, min(1.0, (wx - right_cb['x']) / right_cb['w'])) if right_cb['w'] > 0 else 0.5
                            ry = max(0.0, min(1.0, (wy - right_cb['y']) / right_cb['h'])) if right_cb['h'] > 0 else 0.5
                            invy = 1.0 - ry
                            if data_to_send_udp is None: # Only send RPOS if no R_FIST
                                data_to_send_udp = f"RPOS:{rx:.4f},{invy:.4f},{wz:.4f}"
                            cv2.putText(frame, f"R Rel: {rx:.2f},{invy:.2f}", (right_px['x'], right_px['y'] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Send data if any
                    if data_to_send_udp and sock:
                        try:
                            sock.sendto(data_to_send_udp.encode(), (UDP_IP, UDP_PORT))
                            # print(f"[PY THREAD] Sent UDP: {data_to_send_udp}") # Can be noisy
                        except socket.error as se_udp:
                            print(f"[PY THREAD SOCKET ERROR SENDING] {se_udp}")
                        except Exception as e_udp_send:
                            print(f"[PY THREAD OTHER ERROR SENDING] {e_udp_send}")

            # --- NEW: Update fist states if hands are NOT detected this frame OR if they are no longer fists ---
            if not current_frame_left_fist_detected and left_hand_fist_state:
                left_hand_fist_state = False # Reset if left hand lost or opened
            if not current_frame_right_fist_detected and right_hand_fist_state:
                right_hand_fist_state = False # Reset if right hand lost or opened
            # --- END NEW ---

            # Enqueue frame for GUI display (convert BGR 'frame' to RGB)
            gui_display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_q.full():
                try: frame_q.get_nowait() # Discard oldest if full to prevent lag
                except queue.Empty: pass
            try:
                frame_q.put_nowait(gui_display_frame)
            except queue.Full: # Should be rare after the get_nowait
                pass
                # print("[PY THREAD WARNING] GUI frame queue still full after trying to clear.")

            frame_count += 1

    except Exception as e_loop:
        print(f"[PY THREAD ERROR] Unhandled exception in video processing loop: {e_loop}")
        traceback.print_exc()
    finally:
        print("[PY THREAD] Initiating cleanup for video thread...")
        stop_flag.set() # Ensure other parts of the system know to stop
        if 'cap_thread' in locals() and cap_thread.is_alive():
            print("[PY THREAD] Waiting for capture thread to join...")
            cap_thread.join(timeout=1.0)
            if cap_thread.is_alive(): print("[PY THREAD WARNING] Capture thread did not join cleanly.")
        if 'cap' in locals() and cap and cap.isOpened():
            print("[PY THREAD] Releasing camera capture...")
            cap.release()
        if 'hands' in locals() and hands:
            print("[PY THREAD] Closing MediaPipe Hands...")
            hands.close()
        if sock: # Check if sock was initialized
            print("[PY THREAD] Closing UDP socket...")
            # Temp store and set global to None to prevent race if another thread tries to use it
            temp_sock = sock
            sock = None
            temp_sock.close()
        print("[PY THREAD] Video thread cleaned up.")


# --- Tkinter GUI Application ---
class HandTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.minsize(750, 650) # Your original minsize
        self.selected_camera_index = tk.IntVar(value=-1)
        self.selected_camera_display_name = tk.StringVar()
        self.selected_resolution_str = tk.StringVar(value=DEFAULT_RESOLUTION_STRING)
        self.available_cameras = []
        self.supported_resolutions = []
        self.is_tracking = False
        self.current_width = DEFAULT_WIDTH
        self.current_height = DEFAULT_HEIGHT
        self.update_gui_frame_scheduled = False # For managing GUI updates
        self.placeholder_photo = None # To keep a reference for Tkinter image

        # Control Box Tkinter Variables
        self.left_cb_x_var = tk.DoubleVar()
        self.left_cb_y_var = tk.DoubleVar()
        self.left_cb_w_var = tk.DoubleVar()
        self.left_cb_h_var = tk.DoubleVar()
        self.right_cb_x_var = tk.DoubleVar()
        self.right_cb_y_var = tk.DoubleVar()
        self.right_cb_w_var = tk.DoubleVar()
        self.right_cb_h_var = tk.DoubleVar()

        # Label Variables for Slider Values
        self.left_cb_x_label_var = tk.StringVar()
        self.left_cb_y_label_var = tk.StringVar()
        self.left_cb_w_label_var = tk.StringVar()
        self.left_cb_h_label_var = tk.StringVar()
        self.right_cb_x_label_var = tk.StringVar()
        self.right_cb_y_label_var = tk.StringVar()
        self.right_cb_w_label_var = tk.StringVar()
        self.right_cb_h_label_var = tk.StringVar()

        self._load_config() # Load saved settings or defaults
        # self._update_slider_labels() # Called within _load_config now
        self._update_current_dimensions() # Parse resolution string

        # Main UI Frame
        self.main_frame = ttk.Frame(root, padding="1") # Reduced padding
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Controls Frame (Camera, Resolution, Start/Stop Button)
        self.controls_frame = ttk.LabelFrame(self.main_frame, text="Camera & Tracking", padding="10")
        self.controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        ttk.Label(self.controls_frame, text="Camera:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tk.W)
        self.camera_combobox = ttk.Combobox(self.controls_frame, textvariable=self.selected_camera_display_name, state="readonly", width=15)
        self.camera_combobox.grid(row=0, column=1, padx=(0, 10), pady=5, sticky=tk.W)
        self.camera_combobox.bind("<<ComboboxSelected>>", self.on_camera_selected)
        ttk.Label(self.controls_frame, text="Resolution:").grid(row=0, column=2, padx=(10, 5), pady=5, sticky=tk.W)
        self.resolution_combobox = ttk.Combobox(self.controls_frame, textvariable=self.selected_resolution_str, state="disabled", width=20)
        self.resolution_combobox.grid(row=0, column=3, padx=(0, 10), pady=5, sticky=tk.W)
        self.resolution_combobox.bind("<<ComboboxSelected>>", self.on_resolution_selected)
        self.controls_frame.columnconfigure(4, weight=1) # Make button column expand
        self.toggle_button = ttk.Button(self.controls_frame, text="Start Tracking", command=self.toggle_tracking)
        self.toggle_button.grid(row=0, column=4, padx=10, pady=5, sticky=tk.E) # Align button to East

        # Zone Frame (Sliders for Control Boxes)
        self.zone_frame = ttk.LabelFrame(self.main_frame, text="Control Zones (Normalized 0.0-1.0)", padding="10")
        self.zone_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.zone_frame.columnconfigure(1, weight=1); self.zone_frame.columnconfigure(4, weight=1) # Make slider columns expand

        # Using your original create_slider_row for standard behavior
        def create_slider_row(parent, label_text, double_var, label_var, row_num, col_offset=0):
            ttk.Label(parent, text=label_text).grid(row=row_num, column=0+col_offset, padx=(0, 5), sticky=tk.E)
            scale = ttk.Scale(parent, variable=double_var, from_=0.0, to=1.0, orient=tk.HORIZONTAL, length=150,
                              command=lambda val, dv=double_var, lv=label_var: self._handle_slider_update_label_only(dv, lv))
            scale.grid(row=row_num, column=1+col_offset, padx=(0, 5), sticky=tk.EW)
            label = ttk.Label(parent, textvariable=label_var, width=5) # Fixed width for label
            label.grid(row=row_num, column=2+col_offset, padx=(0, 15), sticky=tk.W)
            return scale, label

        ttk.Label(self.zone_frame, text="Left Hand Zone (Intensity):").grid(row=0, column=0, columnspan=3, pady=(0, 5), sticky=tk.W)
        self.left_x_scale, self.left_x_val_label = create_slider_row(self.zone_frame, "L_X:", self.left_cb_x_var, self.left_cb_x_label_var, 1)
        self.left_y_scale, self.left_y_val_label = create_slider_row(self.zone_frame, "L_Y:", self.left_cb_y_var, self.left_cb_y_label_var, 2)
        self.left_w_scale, self.left_w_val_label = create_slider_row(self.zone_frame, "L_W:", self.left_cb_w_var, self.left_cb_w_label_var, 3)
        self.left_h_scale, self.left_h_val_label = create_slider_row(self.zone_frame, "L_H:", self.left_cb_h_var, self.left_cb_h_label_var, 4)
        ttk.Label(self.zone_frame, text="Right Hand Zone (Movement):").grid(row=0, column=3, columnspan=3, pady=(0, 5), sticky=tk.W)
        self.right_x_scale, self.right_x_val_label = create_slider_row(self.zone_frame, "R_X:", self.right_cb_x_var, self.right_cb_x_label_var, 1, col_offset=3)
        self.right_y_scale, self.right_y_val_label = create_slider_row(self.zone_frame, "R_Y:", self.right_cb_y_var, self.right_cb_y_label_var, 2, col_offset=3)
        self.right_w_scale, self.right_w_val_label = create_slider_row(self.zone_frame, "R_W:", self.right_cb_w_var, self.right_cb_w_label_var, 3, col_offset=3)
        self.right_h_scale, self.right_h_val_label = create_slider_row(self.zone_frame, "R_H:", self.right_cb_h_var, self.right_cb_h_label_var, 4, col_offset=3)
        # List of zone controls (not strictly used in your provided code but good for iterating if needed)
        self.zone_controls = [
            self.left_x_scale, self.left_y_scale, self.left_w_scale, self.left_h_scale,
            self.right_x_scale, self.right_y_scale, self.right_w_scale, self.right_h_scale
        ]

        # Video Display Label
        self.video_label = tk.Label(self.main_frame, bg="black") # For displaying camera feed or placeholder
        self.video_label.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        print("[PY GUI] Initializing camera lists...")
        self.populate_camera_list()
        print("[PY GUI] Camera lists populated.")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close event
        self._draw_placeholder() # Draw initial placeholder with control zones
        print("[PY GUI] Initial placeholder displayed.")
        self.attempt_auto_start() # Attempt to auto-start tracking if configured

    def _load_config(self):
        print("[PY CONFIG] Loading configuration...")
        # Set defaults first
        self.left_cb_x_var.set(DEFAULT_LEFT_CB['x'])
        self.left_cb_y_var.set(DEFAULT_LEFT_CB['y'])
        self.left_cb_w_var.set(DEFAULT_LEFT_CB['w'])
        self.left_cb_h_var.set(DEFAULT_LEFT_CB['h'])
        self.right_cb_x_var.set(DEFAULT_RIGHT_CB['x'])
        self.right_cb_y_var.set(DEFAULT_RIGHT_CB['y'])
        self.right_cb_w_var.set(DEFAULT_RIGHT_CB['w'])
        self.right_cb_h_var.set(DEFAULT_RIGHT_CB['h'])
        self.selected_camera_index.set(-1) # Default to no camera selected
        self.selected_resolution_str.set(DEFAULT_RESOLUTION_STRING)

        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                self.selected_camera_index.set(config_data.get('camera_index', -1))
                self.selected_resolution_str.set(config_data.get('resolution_string', DEFAULT_RESOLUTION_STRING))
                lc = config_data.get('left_control_box', DEFAULT_LEFT_CB)
                rc = config_data.get('right_control_box', DEFAULT_RIGHT_CB)
                self.left_cb_x_var.set(lc.get('x', DEFAULT_LEFT_CB['x']))
                self.left_cb_y_var.set(lc.get('y', DEFAULT_LEFT_CB['y']))
                self.left_cb_w_var.set(lc.get('w', DEFAULT_LEFT_CB['w']))
                self.left_cb_h_var.set(lc.get('h', DEFAULT_LEFT_CB['h']))
                self.right_cb_x_var.set(rc.get('x', DEFAULT_RIGHT_CB['x']))
                self.right_cb_y_var.set(rc.get('y', DEFAULT_RIGHT_CB['y']))
                self.right_cb_w_var.set(rc.get('w', DEFAULT_RIGHT_CB['w']))
                self.right_cb_h_var.set(rc.get('h', DEFAULT_RIGHT_CB['h']))
                print("[PY CONFIG] Configuration loaded.")
            except Exception as e:
                print(f"[PY CONFIG ERROR] Loading configuration: {e}. Using defaults.")
        else:
            print("[PY CONFIG] Configuration file not found. Using default settings.")
        self._update_slider_labels() # Ensure labels are updated after loading

    def _save_config(self):
        print("[PY CONFIG] Saving configuration...")
        config_data = {
            'camera_index': self.selected_camera_index.get(),
            'resolution_string': self.selected_resolution_str.get(),
            'left_control_box': {
                'x': self.left_cb_x_var.get(), 'y': self.left_cb_y_var.get(),
                'w': self.left_cb_w_var.get(), 'h': self.left_cb_h_var.get()
            },
            'right_control_box': {
                'x': self.right_cb_x_var.get(), 'y': self.right_cb_y_var.get(),
                'w': self.right_cb_w_var.get(), 'h': self.right_cb_h_var.get()
            }
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=4)
            print("[PY CONFIG] Configuration saved.")
        except Exception as e:
            print(f"[PY CONFIG ERROR] Saving configuration: {e}")
            traceback.print_exc()

    def _update_slider_labels(self):
        # Updates the text labels next to sliders with their current values
        self.left_cb_x_label_var.set(f"{self.left_cb_x_var.get():.2f}")
        self.left_cb_y_label_var.set(f"{self.left_cb_y_var.get():.2f}")
        self.left_cb_w_label_var.set(f"{self.left_cb_w_var.get():.2f}")
        self.left_cb_h_label_var.set(f"{self.left_cb_h_var.get():.2f}")
        self.right_cb_x_label_var.set(f"{self.right_cb_x_var.get():.2f}")
        self.right_cb_y_label_var.set(f"{self.right_cb_y_var.get():.2f}")
        self.right_cb_w_label_var.set(f"{self.right_cb_w_var.get():.2f}")
        self.right_cb_h_label_var.set(f"{self.right_cb_h_var.get():.2f}")

    def _update_current_dimensions(self):
        # Parses the selected resolution string (e.g., "640x480 (4:3)") to get width and height
        selected_res_str = self.selected_resolution_str.get()
        try:
            res_part = selected_res_str.split(' ')[0] # Get "640x480"
            w_str, h_str = res_part.split('x')
            self.current_width = int(w_str)
            self.current_height = int(h_str)
        except Exception as e:
            print(f"[PY GUI WARNING] Error parsing resolution string '{selected_res_str}'. Using defaults {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}. Error: {e}")
            # Fallback to defaults if parsing fails or results in invalid dimensions
            if self.current_width <= 0 or self.current_height <= 0:
                 self.current_width = DEFAULT_WIDTH
                 self.current_height = DEFAULT_HEIGHT

    def _draw_placeholder(self):
        # Draws a black background with the control zone rectangles if tracking is not active
        if self.is_tracking: return # Don't draw if live video is showing
        try:
            if not self.video_label.winfo_exists(): return # Check if widget still exists
            self._update_current_dimensions() # Ensure width/height are current
            w, h = self.current_width, self.current_height
            if w <= 0 or h <= 0: w, h = DEFAULT_WIDTH, DEFAULT_HEIGHT # Safety for dimensions

            # Create a black numpy array for the image
            placeholder_img_np = np.zeros((h, w, 3), dtype=np.uint8)

            # Get normalized control box values
            lx, ly, lw, lh = self.left_cb_x_var.get(), self.left_cb_y_var.get(), self.left_cb_w_var.get(), self.left_cb_h_var.get()
            rx, ry, rw, rh = self.right_cb_x_var.get(), self.right_cb_y_var.get(), self.right_cb_w_var.get(), self.right_cb_h_var.get()

            # Convert to pixel coordinates
            l_x1, l_y1 = int(lx * w), int(ly * h); l_x2, l_y2 = int((lx + lw) * w), int((ly + lh) * h)
            r_x1, r_y1 = int(rx * w), int(ry * h); r_x2, r_y2 = int((rx + rw) * w), int((ry + rh) * h)

            # Draw rectangles
            cv2.rectangle(placeholder_img_np, (l_x1, l_y1), (l_x2, l_y2), (0, 0, 255), 2) # Left zone in Red
            cv2.rectangle(placeholder_img_np, (r_x1, r_y1), (r_x2, r_y2), (255, 255, 0), 2) # Right zone in Cyan

            # Convert to RGB for Tkinter and display
            rgb_placeholder = cv2.cvtColor(placeholder_img_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_placeholder)
            photo = ImageTk.PhotoImage(image=pil_img, master=self.root) # Specify master for PhotoImage
            self.placeholder_photo = photo # Keep reference
            self.video_label.config(image=self.placeholder_photo, text="") # Clear any text
            self.video_label.image = self.placeholder_photo # Keep reference on widget itself
        except tk.TclError as e:
            print(f"[PY GUI WARNING] TclError drawing placeholder (often during shutdown): {e}")
        except Exception as e:
            print(f"[PY GUI ERROR] Exception drawing placeholder: {e}")
            traceback.print_exc()
            # Attempt to show an error message on the label if drawing fails badly
            try:
                if self.video_label.winfo_exists():
                    self.video_label.config(image='', text="Error drawing zones", fg="red", bg="black")
                    self.video_label.image = None
            except: pass # Ignore errors during error display

    def _handle_slider_update_label_only(self, double_var, label_var):
        # Callback for ttk.Scale's 'command' option to update label live during drag
        try:
            current_value = double_var.get()
            label_var.set(f"{current_value:.2f}")
        except tk.TclError: pass # Can happen if widget is being destroyed
        except Exception as e: print(f"[PY GUI CB ERROR] Error updating label on drag: {e}")

    def _set_control_state(self, new_state_str):
        # Enables or disables UI controls based on whether tracking is active
        # 'normal' for enabled, 'disabled' for disabled
        self.camera_combobox.config(state="disabled" if new_state_str == "disabled" else "readonly")

        # Resolution combobox logic
        res_combo_actual_state = "disabled"
        if new_state_str != "disabled": # If controls should generally be enabled
            # Check if there are valid resolution values (not "N/A" etc.)
            current_res_values = self.resolution_combobox.cget('values')
            if current_res_values and current_res_values[0] not in ["N/A", "Error", "No cameras found"]:
                res_combo_actual_state = "readonly"
        self.resolution_combobox.config(state=res_combo_actual_state)
        # Sliders are always enabled as per your original design
        pass

    def toggle_tracking(self):
        global video_thread, stop_event # These are module-level globals
        if self.is_tracking:
            print("[PY GUI] Stopping tracking...")
            stop_event.set() # Signal the video thread to stop
            self.toggle_button.config(text="Stopping...", state="disabled")
            self.root.update_idletasks() # Update GUI to show "Stopping..."

            if video_thread is not None and video_thread.is_alive():
                 video_thread.join(timeout=2.0) # Wait for thread to finish
                 if video_thread.is_alive(): print("[PY GUI WARNING] Video thread did not join cleanly after 2s.")
                 video_thread = None # Clear thread reference
            self.is_tracking = False
            self.update_gui_frame_scheduled = False # Stop GUI frame updates
            self.toggle_button.config(text="Start Tracking", state="normal")
            self._set_control_state("normal") # Re-enable camera/resolution controls
            self._draw_placeholder() # Show placeholder image
            print("[PY GUI] Tracking stopped.")
        else:
            print("[PY GUI] Starting tracking...")
            cam_index_to_use = self.selected_camera_index.get()
            if cam_index_to_use == -1:
                messagebox.showerror("Error", "No valid camera selected. Cannot start tracking.")
                return

            self._update_current_dimensions() # Ensure width/height are current
            if self.current_width <= 0 or self.current_height <= 0:
                messagebox.showerror("Error", f"Invalid resolution dimensions ({self.current_width}x{self.current_height}). Cannot start tracking.")
                return

            # Prepare settings dictionary for the video thread
            tracking_settings = {
                "camera_index": cam_index_to_use,
                "width": self.current_width, "height": self.current_height,
                "left_x_var": self.left_cb_x_var, "left_y_var": self.left_cb_y_var,
                "left_w_var": self.left_cb_w_var, "left_h_var": self.left_cb_h_var,
                "right_x_var": self.right_cb_x_var, "right_y_var": self.right_cb_y_var,
                "right_w_var": self.right_cb_w_var, "right_h_var": self.right_cb_h_var
            }
            print(f"[PY GUI] Tracking settings prepared for camera {cam_index_to_use} at {self.current_width}x{self.current_height}.")

            self.toggle_button.config(text="Stop Tracking", state="normal") # Update button text
            self._set_control_state("disabled") # Disable camera/resolution controls
            self.placeholder_photo = None # Clear placeholder reference
            if self.video_label.winfo_exists(): # Update video label display
                 self.video_label.config(image='', bg="black", text="Starting Camera...") # Show "Starting..."
            self.root.update_idletasks() # Force GUI update

            stop_event.clear() # Clear stop signal for the new thread
            video_thread = threading.Thread(target=video_processing_loop,
                                            args=(tracking_settings, frame_queue, stop_event),
                                            daemon=True) # Daemon thread will exit with main app
            video_thread.start()
            self.is_tracking = True
            self.update_gui_frame_scheduled = True # Enable GUI frame updates
            self.update_gui_frame() # Start the GUI update loop
            print("[PY GUI] Tracking started.")

    def attempt_auto_start(self):
        print("[PY GUI] Attempting auto-start...")
        if self.selected_camera_index.get() != -1: # If a camera was loaded from config
            self._update_current_dimensions() # Ensure width/height are parsed
            if self.current_width > 0 and self.current_height > 0: # And resolution is valid
                print("[PY GUI] Valid camera and resolution found from config, auto-starting tracking.")
                self.root.after(100, self.toggle_tracking) # Start tracking after a short delay
            else:
                print("[PY GUI] Auto-start skipped: Invalid resolution dimensions from config.")
        else:
            print("[PY GUI] Auto-start skipped: No valid camera selected in config or no config found.")

    def populate_camera_list(self):
        print("[PY GUI] Populating camera list...")
        self.available_cameras = find_available_cameras() # Get list of (name, index) tuples
        if not self.available_cameras:
            # No cameras found: disable relevant controls and show message
            self.camera_combobox['values'] = ["No cameras found"]
            self.camera_combobox.current(0)
            self.camera_combobox['state'] = 'disabled'
            self.toggle_button['state'] = 'disabled'
            self.resolution_combobox['state'] = 'disabled'
            self.resolution_combobox['values'] = ["N/A"]
            self.resolution_combobox.current(0)
            if self.video_label.winfo_exists():
                 self.video_label.config(text="No cameras found. Please connect a camera.", fg="red", image='')
                 self.video_label.image = None # Clear any existing image
            return

        camera_display_names = [name for name, index in self.available_cameras]
        self.camera_combobox['values'] = camera_display_names

        # Try to select the camera loaded from config, or default to the first one
        loaded_cam_device_idx_from_config = self.selected_camera_index.get()
        selected_list_idx_for_combobox = -1
        current_selected_display_name = ""

        for i, (name, device_idx) in enumerate(self.available_cameras):
            if device_idx == loaded_cam_device_idx_from_config:
                selected_list_idx_for_combobox = i
                current_selected_display_name = name
                break

        if selected_list_idx_for_combobox == -1: # If loaded camera not found, default to first
            print(f"[PY GUI WARNING] Saved camera index {loaded_cam_device_idx_from_config} not found/invalid. Selecting first available camera.")
            selected_list_idx_for_combobox = 0
            current_selected_display_name = self.available_cameras[0][0]
            self.selected_camera_index.set(self.available_cameras[0][1]) # Update actual selected device index

        self.camera_combobox.current(selected_list_idx_for_combobox) # Set combobox display
        self.selected_camera_display_name.set(current_selected_display_name) # Update textvariable
        self.camera_combobox.config(state="readonly") # Enable combobox
        self.toggle_button.config(state="normal") # Enable start button
        self.update_resolution_list() # Populate resolutions for the selected camera

    def on_camera_selected(self, event=None): # Bound to <<ComboboxSelected>>
        selected_display_name = self.selected_camera_display_name.get()
        selected_cam_device_index = -1
        # Find the device index corresponding to the selected display name
        for name, index in self.available_cameras:
            if name == selected_display_name:
                selected_cam_device_index = index
                break

        if selected_cam_device_index != -1:
            self.selected_camera_index.set(selected_cam_device_index) # Update the actual device index
            print(f"[PY GUI] Camera changed to: {selected_display_name} (Device Index: {selected_cam_device_index})")
            self.update_resolution_list() # Update resolutions for the new camera
        else:
            # This case should ideally not happen if combobox values are sourced from available_cameras
            print(f"[PY GUI ERROR] Could not find device index for camera name '{selected_display_name}'")
            self.resolution_combobox['state'] = 'disabled'
            self.resolution_combobox['values'] = ["Error selecting camera"]
            self.resolution_combobox.current(0)
            self.toggle_button['state'] = 'disabled'
            self._draw_placeholder() # Show placeholder as camera selection is invalid

    def on_resolution_selected(self, event=None): # Bound to <<ComboboxSelected>>
        print(f"[PY GUI] Resolution selected via combobox: {self.selected_resolution_str.get()}")
        self._update_current_dimensions() # Update internal width/height
        if not self.is_tracking: # If not tracking, redraw placeholder with new dimensions
            self._draw_placeholder()

    def update_resolution_list(self):
        print("[PY GUI] Updating resolution list for selected camera...")
        current_cam_device_idx = self.selected_camera_index.get()
        if current_cam_device_idx == -1: # If no valid camera selected
             self.resolution_combobox['state'] = 'disabled'
             self.resolution_combobox['values'] = ["N/A"]
             self.resolution_combobox.current(0)
             self.toggle_button['state'] = 'disabled'
             self._update_current_dimensions() # Reset dimensions maybe
             self._draw_placeholder()
             return

        self.supported_resolutions = get_supported_resolutions(current_cam_device_idx)
        resolution_display_strings = [DEFAULT_RESOLUTION_STRING] # Default option
        if self.supported_resolutions: # If specific resolutions were found
            resolution_display_strings = [res_tuple[3] for res_tuple in self.supported_resolutions] # Use formatted strings
        self.resolution_combobox['values'] = resolution_display_strings
        self.resolution_combobox['state'] = 'readonly' # Enable combobox

        # Try to select the resolution loaded from config, or default to first available
        loaded_res_str_from_config = self.selected_resolution_str.get()
        selected_idx_for_combobox = -1
        for i, res_str_val in enumerate(resolution_display_strings):
            if res_str_val == loaded_res_str_from_config:
                selected_idx_for_combobox = i
                break

        if selected_idx_for_combobox == -1: # If saved resolution not found/supported for this camera
            print(f"[PY GUI WARNING] Saved resolution '{loaded_res_str_from_config}' not supported for current camera. Selecting first available.")
            selected_idx_for_combobox = 0 # Default to the first resolution in the list
            self.selected_resolution_str.set(resolution_display_strings[selected_idx_for_combobox]) # Update textvariable

        self.resolution_combobox.current(selected_idx_for_combobox) # Set combobox display
        self.toggle_button.config(state="normal") # Ensure start button is enabled
        self._update_current_dimensions() # Update internal width/height based on selection

    def update_gui_frame(self): # Called repeatedly to update video feed in GUI
        if not self.is_tracking or not self.update_gui_frame_scheduled or stop_event.is_set():
            # Stop updating if tracking is off, schedule flag is false, or stop event is set
            self.update_gui_frame_scheduled = False # Ensure it stops if conditions met
            return

        try:
            # Get a frame from the queue (processed by video_thread)
            frame_rgb_from_queue = frame_queue.get_nowait() # Non-blocking get
            pil_image_for_gui = Image.fromarray(frame_rgb_from_queue)
            tkinter_photo_image = ImageTk.PhotoImage(image=pil_image_for_gui, master=self.root)

            if self.video_label.winfo_exists(): # Check if video label widget still exists
                self.video_label.config(image=tkinter_photo_image, text="") # Update image, clear text
                self.video_label.image = tkinter_photo_image # Keep reference to prevent garbage collection
                self.placeholder_photo = None # Clear placeholder reference
            else: # If widget destroyed (e.g., window closing)
                print("[PY GUI WARNING] video_label widget destroyed during GUI frame update.")
                self.update_gui_frame_scheduled = False # Stop scheduling further updates
                if self.is_tracking: self.toggle_tracking() # Attempt to stop tracking cleanly
                return

            if self.root.winfo_exists(): # Check if root window still exists
                self.root.after(15, self.update_gui_frame) # Schedule next update (approx 66 FPS target)
            else: # If root window destroyed
                self.update_gui_frame_scheduled = False # Stop scheduling

        except queue.Empty: # If frame_queue is empty
            if self.root.winfo_exists():
                self.root.after(30, self.update_gui_frame) # Try again a bit later
            else:
                self.update_gui_frame_scheduled = False
        except tk.TclError as e_tk_update: # Catch Tkinter errors (e.g., widget destroyed)
            print(f"[PY GUI ERROR] Tkinter TclError in GUI frame update: {e_tk_update}")
            self.update_gui_frame_scheduled = False
            if self.is_tracking: self.toggle_tracking() # Try to stop tracking
        except RuntimeError as e_rt_update: # Catch other runtime errors
            print(f"[PY GUI ERROR] RuntimeError in GUI frame update: {e_rt_update}")
            traceback.print_exc()
            self.update_gui_frame_scheduled = False
        except Exception as e_gui_update:
            print(f"[PY GUI ERROR] Unexpected error in GUI frame update: {e_gui_update}")
            traceback.print_exc()
            # No self.toggle_tracking() here to avoid potential infinite loops if toggle_tracking also errors

    def on_closing(self): # Called when window 'X' is clicked
        print("[PY GUI] Window close event triggered.")
        self.update_gui_frame_scheduled = False # Stop GUI updates
        if self.is_tracking:
            print("[PY GUI] Tracking active. Attempting to stop tracking before closing...")
            stop_event.set() # Signal video thread
            if video_thread is not None and video_thread.is_alive():
                video_thread.join(timeout=1.0) # Wait briefly for thread
                if video_thread.is_alive(): print("[PY GUI WARNING] Video thread did not stop cleanly on close.")
        self.is_tracking = False # Ensure tracking state is false
        self._save_config() # Save settings
        print("[PY GUI] Destroying Tkinter root window.")

        global sock # Access module-level sock
        if sock: # If socket was initialized and not closed by video_thread
            try:
                temp_sock_close = sock
                sock = None # Prevent video_thread from trying to close it again if it's slow
                temp_sock_close.close()
                print("[PY GUI] Cleaned up UDP socket on application close.")
            except Exception as e_sock_close:
                print(f"[PY GUI ERROR] Error closing UDP socket during on_closing: {e_sock_close}")

        try:
            if self.root and self.root.winfo_exists():
                self.root.destroy() # Destroy the Tkinter window
        except tk.TclError:
            print("[PY GUI INFO] Tkinter window was already destroyed.")
        # No sys.exit(0) here, let mainloop() end naturally or main block handle it.

# --- End of HandTrackingApp Class ---


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Hand Tracking Application v10.3 (FistGestures)...") # Updated version in print
    splash_screen = None # Initialize to None
    main_tk_root = None  # Initialize to None
    app_instance = None  # Initialize to None

    try:
        # splash_screen = create_splash() # Optional: Uncomment to use splash screen

        main_tk_root = tk.Tk()
        main_tk_root.withdraw() # Hide main window initially during setup
        main_tk_root.title("Hand Tracking Control v10.3 (FistGestures)") # Updated title

        print("Main: Initializing HandTrackingApp instance...")
        app_instance = HandTrackingApp(main_tk_root) # Create the application instance
        print("Main: HandTrackingApp instance initialized.")

        # if splash_screen and splash_screen.winfo_exists(): # Close splash screen if it was used
        #     splash_screen.destroy()
        #     splash_screen = None

        print("Main: Showing main application window.")
        if main_tk_root: # If root window was created
            main_tk_root.deiconify() # Show the window
            main_tk_root.lift()      # Bring to front
            main_tk_root.focus_force() # Force focus
            print("Main: Starting Tkinter main event loop.")
            main_tk_root.mainloop() # This line blocks until the window is closed
            print("Main: Tkinter main event loop finished.") # Will print after window closes

    except Exception as e_startup:
        print(f"\n--- FATAL APPLICATION STARTUP ERROR ---")
        traceback.print_exc()
        print(f"Error Details: {e_startup}")
        print("---------------------------------------")
        # if splash_screen and splash_screen.winfo_exists(): splash_screen.destroy()
        # Attempt to show a final error dialog if GUI can be partially initialized
        try:
             error_dialog_root = tk.Tk()
             error_dialog_root.withdraw() # Don't show this root window
             messagebox.showerror("Application Startup Error",
                                  f"A critical error occurred during application startup:\n\n{e_startup}\n\nPlease see the console for more details.")
             error_dialog_root.destroy()
        except Exception as e_msg_dialog:
             print(f"(Additionally, could not display the final error message dialog: {e_msg_dialog})")
        finally: # Ensure main window is destroyed if startup failed
             if main_tk_root and main_tk_root.winfo_exists(): main_tk_root.destroy()

    except KeyboardInterrupt: # Handle Ctrl+C gracefully
         print("\nCtrl+C detected by user. Closing application...")
         # if splash_screen and splash_screen.winfo_exists(): splash_screen.destroy()
         if app_instance: # If app instance was created, use its on_closing method
             app_instance.on_closing()
         elif main_tk_root and main_tk_root.winfo_exists(): # Otherwise, just destroy root
             main_tk_root.destroy()

    finally:
        # This block executes after the try/except/KeyboardInterrupt, or if mainloop() finishes
        print("Main: Application finally block reached.")
        # Final check for socket cleanup, in case it wasn't closed properly elsewhere
        if sock: # Check module-level sock
            try:
                print("Main: Performing final check and closing UDP socket if still open...")
                sock.close()
                sock = None
                print("Main: UDP socket closed in final cleanup.")
            except Exception as e_final_sock:
                print(f"Main: Error closing socket in final cleanup: {e_final_sock}")
        print("Application has finished execution.")
        # sys.exit(0) # Let Python exit naturally after mainloop or if an error occurred