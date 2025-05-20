# hand_tracking_v10.3_StandardSlider_FistGestures.py
# - LVAL/RPOS always sent if in zone.
# - RDIST (Thumb-Index) always sent for Right Hand.
# - GFST:L_FIST/GFST:R_FIST sent as additional info on fist detection.
# - On-screen display for RDIST.

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
frame_queue = queue.Queue(maxsize=2)
sock = None

# Global states for fist detection (to send message only on change)
left_hand_fist_state_previous_frame = False
right_hand_fist_state_previous_frame = False
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
            # print(f"[PY CAM] Error: Could not open camera {camera_index} for probing.") # Less verbose
            return supported
        for width, height, aspect in COMMON_RESOLUTIONS:
            cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            time.sleep(0.05) # Allow camera to attempt setting resolution
            actual_width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Check if the set resolution is close to what was requested
            if abs(actual_width - width) < 20 and abs(actual_height - height) < 20:
                 res_tuple = (width, height)
                 # Ensure no duplicate resolutions (based on w,h) are added
                 if res_tuple not in [(w_s, h_s) for w_s, h_s, _, _ in supported]:
                     formatted = f"{width}x{height} ({aspect})"
                     supported.append((width, height, aspect, formatted))
    except Exception as e:
        print(f"[PY CAM] Exception during resolution probing for camera {camera_index}: {e}")
    finally:
        if cap_test is not None and cap_test.isOpened(): cap_test.release()
    supported.sort(key=lambda x: x[0], reverse=True) # Sort by width, descending
    return supported

# --- Helper Functions for Hand Analysis ---
def calculate_distance(p1, p2): # For 3D MediaPipe Landmark objects
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def is_fist(hand_landmarks, handedness_label):
    if not hand_landmarks:
        return False
    tip_ids = [ # Order: Thumb, Index, Middle, Ring, Pinky
        mp.solutions.hands.HandLandmark.THUMB_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP, mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mp.solutions.hands.HandLandmark.PINKY_TIP
    ]
    mcp_ids = [ # Metacarpophalangeal joints (knuckles at base of fingers)
        mp.solutions.hands.HandLandmark.THUMB_IP, # For thumb, IP (interphalangeal) is closer to "base" when fisted
        mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp.solutions.hands.HandLandmark.RING_FINGER_MCP, mp.solutions.hands.HandLandmark.PINKY_MCP
    ]
    closed_fingers = 0
    # We check fingers from Index to Pinky (indices 1 to 4 in tip_ids/mcp_ids)
    for i in range(1, 5):
        tip = hand_landmarks.landmark[tip_ids[i]]
        mcp = hand_landmarks.landmark[mcp_ids[i]]
        distance = calculate_distance(tip, mcp) # Uses 3D distance
        # if i == 1 and "Left" in handedness_label: print(f"{handedness_label} Index Tip-MCP Dist: {distance:.4f}") # For tuning FINGER_CLOSE_THRESHOLD
        if distance < FINGER_CLOSE_THRESHOLD:
            closed_fingers += 1
    return closed_fingers >= 3 # If at least 3 of the main four fingers are closed

# --- Video Processing Thread ---
def video_processing_loop(settings, frame_q, stop_flag):
    global sock, left_hand_fist_state_previous_frame, right_hand_fist_state_previous_frame
    print("[PY THREAD] Video thread starting...")
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hands_solution = mp_hands.Hands( # Renamed to avoid conflict with 'hands' variable if any
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
        model_complexity=1, static_image_mode=False, max_num_hands=2
    )
    cap = cv2.VideoCapture(settings['camera_index'], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[PY THREAD ERROR] Cannot open camera {settings['camera_index']}")
        stop_flag.set(); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
    time.sleep(0.2) # Give camera time
    # print(f"[PY THREAD] Camera requested {settings['width']}x{settings['height']}, got {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    capture_queue = queue.Queue(maxsize=1) # Reduced queue size for lower latency
    def capture_loop_fn():
        while not stop_flag.is_set():
            ret, raw_frame = cap.read()
            if not ret: time.sleep(0.01); continue
            if capture_queue.full():
                try: capture_queue.get_nowait()
                except queue.Empty: pass
            capture_queue.put(raw_frame)
    cap_thread = threading.Thread(target=capture_loop_fn, daemon=True)
    cap_thread.start()
    # print("[PY THREAD] Capture thread started.") # Less verbose

    try:
        last_left_cb_values = DEFAULT_LEFT_CB.copy()
        last_right_cb_values = DEFAULT_RIGHT_CB.copy()

        while not stop_flag.is_set():
            try: frame = capture_queue.get(timeout=0.5) # Shorter timeout
            except queue.Empty: continue
            frame = cv2.flip(frame, 1)

            try: # Read Tkinter vars for control boxes
                left_cb = {k: settings[f'left_{k}_var'].get() for k in ['x', 'y', 'w', 'h']}
                right_cb = {k: settings[f'right_{k}_var'].get() for k in ['x', 'y', 'w', 'h']}
                last_left_cb_values, last_right_cb_values = left_cb.copy(), right_cb.copy()
            except Exception: # Fallback if Tkinter vars are gone
                left_cb, right_cb = last_left_cb_values.copy(), last_right_cb_values.copy()

            h_frame, w_frame, _ = frame.shape
            left_px = {k: int(left_cb[k] * (w_frame if k in 'xw' else h_frame)) for k in left_cb}
            right_px = {k: int(right_cb[k] * (w_frame if k in 'xw' else h_frame)) for k in right_cb}

            cv2.rectangle(frame, (left_px['x'], left_px['y']), (left_px['x'] + left_px['w'], left_px['y'] + left_px['h']), (0, 0, 255), 2) # Left = Red
            cv2.rectangle(frame, (right_px['x'], right_px['y']), (right_px['x'] + right_px['w'], right_px['y'] + right_px['h']), (255, 255, 0), 2) # Right = Cyan

            rgb_frame_for_mediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame_for_mediapipe.flags.writeable = False # Performance optimization
            results = hands_solution.process(rgb_frame_for_mediapipe)
            # frame.flags.writeable = True # For drawing on 'frame'

            # Track if fists are detected in the current frame (for resetting previous_frame state later)
            current_frame_has_left_fist = False
            current_frame_has_right_fist = False

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness_label = results.multi_handedness[hand_idx].classification[0].label
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    is_current_hand_a_fist_now = is_fist(hand_landmarks, handedness_label)
                    
                    messages_to_send_this_frame = []

                    if handedness_label == "Left":
                        current_frame_has_left_fist = is_current_hand_a_fist_now
                        # LVAL
                        tip_l = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        if left_cb['x'] <= tip_l.x < left_cb['x'] + left_cb['w'] and \
                           left_cb['y'] <= tip_l.y < left_cb['y'] + left_cb['h']:
                            rel_y_l = (tip_l.y - left_cb['y']) / left_cb['h'] if left_cb['h'] > 0 else 0.5
                            val_l = max(0.0, min(1.0, 1.0 - rel_y_l))
                            messages_to_send_this_frame.append(f"LVAL:{val_l:.4f}")
                            cv2.putText(frame, f"L Val: {val_l:.2f}", (left_px['x'], left_px['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
                        # GFST:L_FIST
                        if is_current_hand_a_fist_now:
                            cv2.putText(frame, "L_FIST", (left_px['x'], left_px['y'] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,100),2)
                            if not left_hand_fist_state_previous_frame: # New fist
                                messages_to_send_this_frame.append("GFST:L_FIST")
                        left_hand_fist_state_previous_frame = is_current_hand_a_fist_now


                    elif handedness_label == "Right":
                        current_frame_has_right_fist = is_current_hand_a_fist_now
                        # RPOS
                        tip_r = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        if right_cb['x'] <= tip_r.x < right_cb['x'] + right_cb['w'] and \
                           right_cb['y'] <= tip_r.y < right_cb['y'] + right_cb['h']:
                            rx_r = (tip_r.x - right_cb['x']) / right_cb['w'] if right_cb['w'] > 0 else 0.5
                            ry_r = (tip_r.y - right_cb['y']) / right_cb['h'] if right_cb['h'] > 0 else 0.5
                            messages_to_send_this_frame.append(f"RPOS:{max(0,min(1,rx_r)):.4f},{max(0,min(1,1-ry_r)):.4f},{tip_r.z:.4f}")
                            cv2.putText(frame, f"R Pos: {rx_r:.2f},{1-ry_r:.2f}", (right_px['x'], right_px['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)
                        # RDIST (Thumb-Index)
                        if len(hand_landmarks.landmark) > mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP:
                            try:
                                thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                                index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                                dist_val = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) # 2D screen distance
                                messages_to_send_this_frame.append(f"RDIST:{dist_val:.4f}")
                                cv2.putText(frame, f"R-Dist: {dist_val:.3f}", (right_px['x'], right_px['y'] - 50 if is_current_hand_a_fist_now else right_px['y'] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
                            except Exception: pass # Ignore if landmarks are weird
                        # GFST:R_FIST
                        if is_current_hand_a_fist_now:
                            cv2.putText(frame, "R_FIST", (right_px['x'], right_px['y'] - (70 if len(messages_to_send_this_frame)>1 else 30) ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,0),2)
                            if not right_hand_fist_state_previous_frame: # New fist
                                messages_to_send_this_frame.append("GFST:R_FIST")
                        right_hand_fist_state_previous_frame = is_current_hand_a_fist_now

                    # Send all messages for this hand
                    for msg_to_send in messages_to_send_this_frame:
                        if sock:
                            try: sock.sendto(msg_to_send.encode(), (UDP_IP, UDP_PORT))
                            except Exception as e_send: print(f"[PY SEND ERROR] {e_send} for {msg_to_send}")
            
            # After processing all hands, if a hand that was fisted is no longer detected as fisted
            if not current_frame_has_left_fist and left_hand_fist_state_previous_frame:
                left_hand_fist_state_previous_frame = False
                # Optionally send GFST:L_FIST_OFF if sock: sock.sendto("GFST:L_FIST_OFF".encode(), (UDP_IP, UDP_PORT))
            if not current_frame_has_right_fist and right_hand_fist_state_previous_frame:
                right_hand_fist_state_previous_frame = False
                # Optionally send GFST:R_FIST_OFF if sock: sock.sendto("GFST:R_FIST_OFF".encode(), (UDP_IP, UDP_PORT))


            gui_display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_q.full():
                try: frame_q.get_nowait()
                except queue.Empty: pass
            try: frame_q.put_nowait(gui_display_frame)
            except queue.Full: pass # Should be rare

    except Exception as e_loop:
        print(f"[PY THREAD ERROR] Unhandled exception in video processing loop: {e_loop}")
        traceback.print_exc()
    finally: # Cleanup
        print("[PY THREAD] Initiating cleanup for video thread...")
        stop_flag.set()
        if 'cap_thread' in locals() and cap_thread.is_alive(): cap_thread.join(timeout=0.5)
        if 'cap' in locals() and cap and cap.isOpened(): cap.release()
        if 'hands_solution' in locals() and hands_solution: hands_solution.close() # Use correct variable name
        if sock:
            temp_sock, sock = sock, None # Prevent race conditions on close
            temp_sock.close()
        print("[PY THREAD] Video thread cleaned up.")

# --- Tkinter GUI Application Class (HandTrackingApp) ---
# ... (This class remains the same as your v10.3) ...
class HandTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.minsize(750, 650) 
        self.selected_camera_index = tk.IntVar(value=-1)
        self.selected_camera_display_name = tk.StringVar()
        self.selected_resolution_str = tk.StringVar(value=DEFAULT_RESOLUTION_STRING)
        self.available_cameras = []
        self.supported_resolutions = []
        self.is_tracking = False
        self.current_width = DEFAULT_WIDTH
        self.current_height = DEFAULT_HEIGHT
        self.update_gui_frame_scheduled = False 
        self.placeholder_photo = None 

        self.left_cb_x_var = tk.DoubleVar()
        self.left_cb_y_var = tk.DoubleVar()
        self.left_cb_w_var = tk.DoubleVar()
        self.left_cb_h_var = tk.DoubleVar()
        self.right_cb_x_var = tk.DoubleVar()
        self.right_cb_y_var = tk.DoubleVar()
        self.right_cb_w_var = tk.DoubleVar()
        self.right_cb_h_var = tk.DoubleVar()

        self.left_cb_x_label_var = tk.StringVar()
        self.left_cb_y_label_var = tk.StringVar()
        self.left_cb_w_label_var = tk.StringVar()
        self.left_cb_h_label_var = tk.StringVar()
        self.right_cb_x_label_var = tk.StringVar()
        self.right_cb_y_label_var = tk.StringVar()
        self.right_cb_w_label_var = tk.StringVar()
        self.right_cb_h_label_var = tk.StringVar()

        self._load_config() 
        self._update_current_dimensions() 

        self.main_frame = ttk.Frame(root, padding="1")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

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
        self.controls_frame.columnconfigure(4, weight=1) 
        self.toggle_button = ttk.Button(self.controls_frame, text="Start Tracking", command=self.toggle_tracking)
        self.toggle_button.grid(row=0, column=4, padx=10, pady=5, sticky=tk.E) 

        self.zone_frame = ttk.LabelFrame(self.main_frame, text="Control Zones (Normalized 0.0-1.0)", padding="10")
        self.zone_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.zone_frame.columnconfigure(1, weight=1); self.zone_frame.columnconfigure(4, weight=1)

        def create_slider_row(parent, label_text, double_var, label_var, row_num, col_offset=0):
            ttk.Label(parent, text=label_text).grid(row=row_num, column=0+col_offset, padx=(0, 5), sticky=tk.E)
            scale = ttk.Scale(parent, variable=double_var, from_=0.0, to=1.0, orient=tk.HORIZONTAL, length=150,
                              command=lambda val, dv=double_var, lv=label_var: self._handle_slider_update_label_only(dv, lv))
            scale.grid(row=row_num, column=1+col_offset, padx=(0, 5), sticky=tk.EW)
            label = ttk.Label(parent, textvariable=label_var, width=5) 
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
        
        self.video_label = tk.Label(self.main_frame, bg="black") 
        self.video_label.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.populate_camera_list()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) 
        self._draw_placeholder() 
        self.attempt_auto_start() 

    def _load_config(self):
        # print("[PY CONFIG] Loading configuration...") # Less verbose
        self.left_cb_x_var.set(DEFAULT_LEFT_CB['x']) # Set defaults first
        self.left_cb_y_var.set(DEFAULT_LEFT_CB['y'])
        self.left_cb_w_var.set(DEFAULT_LEFT_CB['w'])
        self.left_cb_h_var.set(DEFAULT_LEFT_CB['h'])
        self.right_cb_x_var.set(DEFAULT_RIGHT_CB['x'])
        self.right_cb_y_var.set(DEFAULT_RIGHT_CB['y'])
        self.right_cb_w_var.set(DEFAULT_RIGHT_CB['w'])
        self.right_cb_h_var.set(DEFAULT_RIGHT_CB['h'])
        self.selected_camera_index.set(-1) 
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
                # print("[PY CONFIG] Configuration loaded.") # Less verbose
            except Exception as e:
                print(f"[PY CONFIG ERROR] Loading configuration: {e}. Using defaults.")
        # else: print("[PY CONFIG] Configuration file not found. Using default settings.") # Less verbose
        self._update_slider_labels() 

    def _save_config(self):
        # print("[PY CONFIG] Saving configuration...") # Less verbose
        config_data = {
            'camera_index': self.selected_camera_index.get(),
            'resolution_string': self.selected_resolution_str.get(),
            'left_control_box': {k: getattr(self, f'left_cb_{k}_var').get() for k in ['x', 'y', 'w', 'h']},
            'right_control_box': {k: getattr(self, f'right_cb_{k}_var').get() for k in ['x', 'y', 'w', 'h']}
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=4)
            # print("[PY CONFIG] Configuration saved.") # Less verbose
        except Exception as e:
            print(f"[PY CONFIG ERROR] Saving configuration: {e}")

    def _update_slider_labels(self):
        for hand_prefix in ['left', 'right']:
            for dim_suffix in ['x', 'y', 'w', 'h']:
                var_instance = getattr(self, f'{hand_prefix}_cb_{dim_suffix}_var')
                label_var_instance = getattr(self, f'{hand_prefix}_cb_{dim_suffix}_label_var')
                label_var_instance.set(f"{var_instance.get():.2f}")

    def _update_current_dimensions(self):
        selected_res_str = self.selected_resolution_str.get()
        try:
            res_part = selected_res_str.split(' ')[0] # "640x480"
            w_str, h_str = res_part.split('x')
            self.current_width = int(w_str)
            self.current_height = int(h_str)
            if self.current_width <= 0 or self.current_height <= 0: # Basic sanity check
                raise ValueError("Dimensions must be positive.")
        except Exception as e:
            # print(f"[PY GUI WARNING] Error parsing resolution string '{selected_res_str}'. Using defaults. Error: {e}") # Less verbose
            self.current_width, self.current_height = DEFAULT_WIDTH, DEFAULT_HEIGHT
            self.selected_resolution_str.set(DEFAULT_RESOLUTION_STRING) # Reset to default string


    def _draw_placeholder(self):
        if self.is_tracking: return
        try:
            if not self.video_label.winfo_exists(): return
            self._update_current_dimensions() # Ensure width/height are current
            w, h = self.current_width, self.current_height
            
            placeholder_img_np = np.zeros((h, w, 3), dtype=np.uint8) # Black image
            # Get normalized control box values
            lx, ly, lw, lh = self.left_cb_x_var.get(), self.left_cb_y_var.get(), self.left_cb_w_var.get(), self.left_cb_h_var.get()
            rx, ry, rw, rh = self.right_cb_x_var.get(), self.right_cb_y_var.get(), self.right_cb_w_var.get(), self.right_cb_h_var.get()
            
            # Convert to pixel coordinates
            l_x1, l_y1, l_x2, l_y2 = int(lx*w), int(ly*h), int((lx+lw)*w), int((ly+lh)*h)
            r_x1, r_y1, r_x2, r_y2 = int(rx*w), int(ry*h), int((rx+rw)*w), int((ry+rh)*h)

            cv2.rectangle(placeholder_img_np, (l_x1, l_y1), (l_x2, l_y2), (0, 0, 255), 2) # Left zone in Red
            cv2.rectangle(placeholder_img_np, (r_x1, r_y1), (r_x2, r_y2), (255, 255, 0), 2) # Right zone in Cyan

            rgb_placeholder = cv2.cvtColor(placeholder_img_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_placeholder)
            photo = ImageTk.PhotoImage(image=pil_img, master=self.root) # Specify master
            self.placeholder_photo = photo 
            self.video_label.config(image=self.placeholder_photo, text="")
            self.video_label.image = self.placeholder_photo # Keep reference on widget
        except tk.TclError: pass # Often during shutdown
        except Exception as e: print(f"[PY GUI ERROR] Exception drawing placeholder: {e}"); traceback.print_exc()

    def _handle_slider_update_label_only(self, double_var, label_var):
        try: label_var.set(f"{double_var.get():.2f}")
        except tk.TclError: pass 
        except Exception as e: print(f"[PY GUI CB ERROR] Updating label on drag: {e}")

    def _set_control_state(self, new_state_str): # 'normal' or 'disabled'
        tk_state = "disabled" if new_state_str == "disabled" else "readonly" # Comboboxes use readonly
        self.camera_combobox.config(state=tk_state)
        
        res_combo_actual_state = "disabled" # Default to disabled
        if tk_state == "readonly": # If controls should generally be enabled
            current_res_values = self.resolution_combobox.cget('values')
            if current_res_values and current_res_values[0] not in ["N/A", "Error", "No cameras found"]:
                res_combo_actual_state = "readonly"
        self.resolution_combobox.config(state=res_combo_actual_state)
        # Sliders remain enabled

    def toggle_tracking(self):
        global video_thread, stop_event # Module-level globals
        if self.is_tracking:
            # print("[PY GUI] Stopping tracking...") # Less verbose
            stop_event.set()
            self.toggle_button.config(text="Stopping...", state="disabled")
            self.root.update_idletasks()
            if video_thread and video_thread.is_alive():
                 video_thread.join(timeout=1.0) # Shorter join timeout
                 # if video_thread.is_alive(): print("[PY GUI WARNING] Video thread did not join cleanly.") # Less verbose
            video_thread = None
            self.is_tracking = False
            self.update_gui_frame_scheduled = False
            self.toggle_button.config(text="Start Tracking", state="normal")
            self._set_control_state("normal")
            self._draw_placeholder()
            # print("[PY GUI] Tracking stopped.") # Less verbose
        else:
            # print("[PY GUI] Starting tracking...") # Less verbose
            cam_idx_to_use = self.selected_camera_index.get()
            if cam_idx_to_use == -1:
                messagebox.showerror("Error", "No valid camera selected. Cannot start tracking.")
                return
            self._update_current_dimensions() # Ensure width/height are current
            if self.current_width <= 0 or self.current_height <= 0:
                messagebox.showerror("Error", f"Invalid resolution dimensions ({self.current_width}x{self.current_height}). Cannot start tracking.")
                return

            tracking_settings = {
                "camera_index": cam_idx_to_use, "width": self.current_width, "height": self.current_height,
                **{f'{h}_{d}_var': getattr(self, f'{h}_cb_{d}_var') for h in ['left', 'right'] for d in ['x', 'y', 'w', 'h']}
            }
            # print(f"[PY GUI] Tracking settings prepared for camera {cam_idx_to_use} at {self.current_width}x{self.current_height}.")

            self.toggle_button.config(text="Stop Tracking", state="normal")
            self._set_control_state("disabled")
            self.placeholder_photo = None # Clear placeholder
            if self.video_label.winfo_exists(): self.video_label.config(image='', bg="black", text="Starting Camera...")
            self.root.update_idletasks()

            stop_event.clear()
            video_thread = threading.Thread(target=video_processing_loop, args=(tracking_settings, frame_queue, stop_event), daemon=True)
            video_thread.start()
            self.is_tracking = True
            self.update_gui_frame_scheduled = True
            self.update_gui_frame()
            # print("[PY GUI] Tracking started.") # Less verbose

    def attempt_auto_start(self):
        # print("[PY GUI] Attempting auto-start...") # Less verbose
        if self.selected_camera_index.get() != -1:
            self._update_current_dimensions()
            if self.current_width > 0 and self.current_height > 0:
                # print("[PY GUI] Valid camera and resolution found from config, auto-starting tracking.") # Less verbose
                self.root.after(100, self.toggle_tracking)
            # else: print("[PY GUI] Auto-start skipped: Invalid resolution dimensions from config.") # Less verbose
        # else: print("[PY GUI] Auto-start skipped: No valid camera selected in config or no config found.") # Less verbose

    def populate_camera_list(self):
        # print("[PY GUI] Populating camera list...") # Less verbose
        self.available_cameras = find_available_cameras()
        if not self.available_cameras:
            self.camera_combobox['values'] = ["No cameras found"]
            self.camera_combobox.current(0); self.camera_combobox['state'] = 'disabled'
            self.toggle_button['state'] = 'disabled'
            self.resolution_combobox['state'] = 'disabled'; self.resolution_combobox['values'] = ["N/A"]; self.resolution_combobox.current(0)
            if self.video_label.winfo_exists(): self.video_label.config(text="No cameras found. Connect a camera.", fg="red", image=''); self.video_label.image = None
            return

        camera_display_names = [name for name, index in self.available_cameras]
        self.camera_combobox['values'] = camera_display_names
        loaded_cam_device_idx_from_config = self.selected_camera_index.get()
        selected_list_idx_for_combobox = next((i for i, (_, dev_idx) in enumerate(self.available_cameras) if dev_idx == loaded_cam_device_idx_from_config), -1)

        if selected_list_idx_for_combobox == -1: # If loaded camera not found, default to first
            # print(f"[PY GUI WARNING] Saved camera index {loaded_cam_device_idx_from_config} not found/invalid. Selecting first.") # Less verbose
            selected_list_idx_for_combobox = 0
            self.selected_camera_index.set(self.available_cameras[0][1]) # Update actual selected device index
        
        self.camera_combobox.current(selected_list_idx_for_combobox)
        self.selected_camera_display_name.set(self.available_cameras[selected_list_idx_for_combobox][0]) # Update textvariable
        self.camera_combobox.config(state="readonly")
        self.toggle_button.config(state="normal")
        self.update_resolution_list()

    def on_camera_selected(self, event=None):
        selected_display_name = self.selected_camera_display_name.get()
        selected_cam_device_index = next((idx for name, idx in self.available_cameras if name == selected_display_name), -1)
        if selected_cam_device_index != -1:
            self.selected_camera_index.set(selected_cam_device_index)
            # print(f"[PY GUI] Camera changed to: {selected_display_name} (Device Index: {selected_cam_device_index})") # Less verbose
            self.update_resolution_list()
        else: # Should not happen if combobox values are from available_cameras
            # print(f"[PY GUI ERROR] Could not find device index for camera name '{selected_display_name}'") # Less verbose
            self.resolution_combobox['state'] = 'disabled'; self.resolution_combobox['values'] = ["Error selecting camera"]; self.resolution_combobox.current(0)
            self.toggle_button['state'] = 'disabled'; self._draw_placeholder()

    def on_resolution_selected(self, event=None):
        # print(f"[PY GUI] Resolution selected via combobox: {self.selected_resolution_str.get()}") # Less verbose
        self._update_current_dimensions()
        if not self.is_tracking: self._draw_placeholder()

    def update_resolution_list(self):
        # print("[PY GUI] Updating resolution list for selected camera...") # Less verbose
        current_cam_device_idx = self.selected_camera_index.get()
        if current_cam_device_idx == -1: # If no valid camera selected
             self.resolution_combobox['state'] = 'disabled'; self.resolution_combobox['values'] = ["N/A"]; self.resolution_combobox.current(0)
             self.toggle_button['state'] = 'disabled'; self._update_current_dimensions(); self._draw_placeholder()
             return

        self.supported_resolutions = get_supported_resolutions(current_cam_device_idx)
        resolution_display_strings = [res_tuple[3] for res_tuple in self.supported_resolutions] if self.supported_resolutions else [DEFAULT_RESOLUTION_STRING]
        self.resolution_combobox['values'] = resolution_display_strings
        self.resolution_combobox['state'] = 'readonly'

        loaded_res_str_from_config = self.selected_resolution_str.get()
        selected_idx_for_combobox = next((i for i, res_str_val in enumerate(resolution_display_strings) if res_str_val == loaded_res_str_from_config), -1)

        if selected_idx_for_combobox == -1: # If saved resolution not found/supported
            # print(f"[PY GUI WARNING] Saved resolution '{loaded_res_str_from_config}' not supported. Selecting first.") # Less verbose
            selected_idx_for_combobox = 0 
            self.selected_resolution_str.set(resolution_display_strings[selected_idx_for_combobox])

        self.resolution_combobox.current(selected_idx_for_combobox)
        self.toggle_button.config(state="normal")
        self._update_current_dimensions()

    def update_gui_frame(self):
        if not self.is_tracking or not self.update_gui_frame_scheduled or stop_event.is_set():
            self.update_gui_frame_scheduled = False; return
        try:
            frame_rgb_from_queue = frame_queue.get_nowait()
            pil_image_for_gui = Image.fromarray(frame_rgb_from_queue)
            tkinter_photo_image = ImageTk.PhotoImage(image=pil_image_for_gui, master=self.root)
            if self.video_label.winfo_exists():
                self.video_label.config(image=tkinter_photo_image, text=""); self.video_label.image = tkinter_photo_image
                self.placeholder_photo = None
            else: # Widget destroyed
                self.update_gui_frame_scheduled = False
                if self.is_tracking: self.toggle_tracking() # Attempt to stop
                return
            if self.root.winfo_exists(): self.root.after(15, self.update_gui_frame) # Approx 66 FPS
            else: self.update_gui_frame_scheduled = False # Root window destroyed
        except queue.Empty: # No frame available
            if self.root.winfo_exists(): self.root.after(30, self.update_gui_frame) # Try again later
            else: self.update_gui_frame_scheduled = False
        except tk.TclError as e_tk_update: # Catch Tkinter errors (e.g., widget destroyed)
            print(f"[PY GUI ERROR] Tkinter TclError in GUI frame update: {e_tk_update}")
            self.update_gui_frame_scheduled = False; self.toggle_tracking()
        except RuntimeError as e_rt_update: # Catch other runtime errors
            print(f"[PY GUI ERROR] RuntimeError in GUI frame update: {e_rt_update}"); traceback.print_exc()
            self.update_gui_frame_scheduled = False
        except Exception as e_gui_update:
            print(f"[PY GUI ERROR] Unexpected error in GUI frame update: {e_gui_update}"); traceback.print_exc()

    def on_closing(self):
        # print("[PY GUI] Window close event triggered.") # Less verbose
        self.update_gui_frame_scheduled = False
        if self.is_tracking:
            # print("[PY GUI] Tracking active. Attempting to stop tracking before closing...") # Less verbose
            stop_event.set()
            if video_thread and video_thread.is_alive(): video_thread.join(timeout=0.5) # Shorter timeout
        self.is_tracking = False
        self._save_config()
        # print("[PY GUI] Destroying Tkinter root window.") # Less verbose
        global sock
        if sock: 
            try: temp_sock_close, sock = sock, None; temp_sock_close.close() # print("[PY GUI] Cleaned up UDP socket on application close.") # Less verbose
            except Exception as e_sock_close: print(f"[PY GUI ERROR] Error closing UDP socket during on_closing: {e_sock_close}")
        try:
            if self.root and self.root.winfo_exists(): self.root.destroy()
        except tk.TclError: pass # print("[PY GUI INFO] Tkinter window was already destroyed.") # Less verbose

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Hand Tracking Application v10.3 (FistGestures, ThumbIndex RDIST, LVAL/RPOS Always)...")
    splash_screen, main_tk_root, app_instance = None, None, None
    try:
        # splash_screen = create_splash() # Optional
        main_tk_root = tk.Tk()
        main_tk_root.withdraw() # Hide main window initially
        main_tk_root.title("Hand Tracking Control v10.3 (Full Send)") # Updated title

        # print("Main: Initializing HandTrackingApp instance...") # Less verbose
        app_instance = HandTrackingApp(main_tk_root) # Create the application instance
        # print("Main: HandTrackingApp instance initialized.") # Less verbose

        # if splash_screen and splash_screen.winfo_exists(): splash_screen.destroy(); splash_screen = None

        # print("Main: Showing main application window.") # Less verbose
        if main_tk_root: 
            main_tk_root.deiconify(); main_tk_root.lift(); main_tk_root.focus_force()
            # print("Main: Starting Tkinter main event loop.") # Less verbose
            main_tk_root.mainloop() 
            # print("Main: Tkinter main event loop finished.") # Less verbose

    except Exception as e_startup:
        print(f"\n--- FATAL APPLICATION STARTUP ERROR ---")
        traceback.print_exc()
        print(f"Error Details: {e_startup}")
        print("---------------------------------------")
        # if splash_screen and splash_screen.winfo_exists(): splash_screen.destroy()
        try: # Attempt to show a final error dialog
             error_dialog_root = tk.Tk(); error_dialog_root.withdraw() # Don't show this root
             messagebox.showerror("Application Startup Error", f"A critical error occurred during application startup:\n\n{e_startup}\n\nPlease see the console for more details.")
             error_dialog_root.destroy()
        except Exception as e_msg_dialog: print(f"(Additionally, could not display the final error message dialog: {e_msg_dialog})")
        finally: 
             if main_tk_root and main_tk_root.winfo_exists(): main_tk_root.destroy()

    except KeyboardInterrupt: # Handle Ctrl+C
         print("\nCtrl+C detected by user. Closing application...")
         # if splash_screen and splash_screen.winfo_exists(): splash_screen.destroy()
         if app_instance: app_instance.on_closing() # Use app's cleanup
         elif main_tk_root and main_tk_root.winfo_exists(): main_tk_root.destroy() # Fallback

    finally:
        # print("Main: Application finally block reached.") # Less verbose
        if sock: # Final check for socket cleanup
            try:
                # print("Main: Performing final check and closing UDP socket if still open...") # Less verbose
                sock.close(); sock = None
                # print("Main: UDP socket closed in final cleanup.") # Less verbose
            except Exception as e_final_sock: print(f"Main: Error closing socket in final cleanup: {e_final_sock}")
        print("Application has finished execution.")