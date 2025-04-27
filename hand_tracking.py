# hand_tracking_v10.3_StandardSlider.py (Python)
# - Auto-starts tracking on launch if camera is available.
# - Allows live updates of control zones via sliders during tracking.
# - Uses direct reading of Tkinter variables in the tracking thread.
# - Reverted to standard slider behavior (only drag works, click on track does nothing).
# - Uses 'command' option for live label update during drag.

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
    # print(f"[DEBUG PY CAM] Found cameras: {available_cameras}")
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
            time.sleep(0.05) # Short delay for setting to apply
            actual_width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Allow slightly larger tolerance
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
    # print(f"[DEBUG PY CAM] Supported resolutions found for camera {camera_index}: {[s[3] for s in supported]}")
    return supported

# --- Helper Functions for Hand Analysis ---
def get_landmark_coords(landmarks, landmark_id, frame_width, frame_height):
    lm = landmarks.landmark[landmark_id]
    return int(lm.x * frame_width), int(lm.y * frame_height), lm.z

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# --- Video Processing Thread ---
# (No changes needed in video_processing_loop itself for this GUI behavior change)
# --- Video Processing with dedicated capture thread ---
def video_processing_loop(settings, frame_q, stop_flag):
    global sock
    print("[PY THREAD] Video thread starting...")
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        static_image_mode=False,
        max_num_hands=2
    )

    # Open camera
    cap = cv2.VideoCapture(settings['camera_index'], cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera {settings['camera_index']}")

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
    time.sleep(0.2)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[PY THREAD] Using resolution {frame_w}x{frame_h}")

    # Prepare UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Dedicated capture thread to always grab latest frame
    capture_queue = queue.Queue(maxsize=1)
    def capture_loop():
        while not stop_flag.is_set():
            ret, raw = cap.read()
            if not ret:
                continue
            # keep only most recent
            if capture_queue.full():
                try: capture_queue.get_nowait()
                except queue.Empty: pass
            capture_queue.put(raw)
    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    cap_thread.start()
    print("[PY THREAD] Capture thread started.")

    try:
        frame_count = 0
        last_left = DEFAULT_LEFT_CB.copy()
        last_right = DEFAULT_RIGHT_CB.copy()

        while not stop_flag.is_set():
            try:
                frame = capture_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            frame = cv2.flip(frame, 1)

            # read control box vars
            try:
                left_cb = {
                    'x': settings['left_x_var'].get(), 'y': settings['left_y_var'].get(),
                    'w': settings['left_w_var'].get(), 'h': settings['left_h_var'].get()
                }
                right_cb = {
                    'x': settings['right_x_var'].get(), 'y': settings['right_y_var'].get(),
                    'w': settings['right_w_var'].get(), 'h': settings['right_h_var'].get()
                }
                last_left = left_cb.copy(); last_right = right_cb.copy()
            except Exception:
                left_cb = last_left.copy(); right_cb = last_right.copy()

            # pixel coords
            left_px = {k: int(v * frame_w) for k, v in left_cb.items()}
            right_px = {k: int(v * frame_w) if k in ['x','w'] else int(v * frame_h) for k,v in right_cb.items()}

            # draw zones
            cv2.rectangle(frame, (left_px['x'], left_px['y']),
( left_px['x']+left_px['w'], left_px['y']+left_px['h']), (0,0,255),2)
            cv2.rectangle(frame, (right_px['x'], right_px['y']),
( right_px['x']+right_px['w'], right_px['y']+right_px['h']), (255,255,0),2)

            # hand detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                for idx, hand in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[idx].classification[0].label
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    # ← use INDEX_FINGER_TIP instead of WRIST
                    tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    wx, wy, wz = tip.x, tip.y, tip.z
                    data = None
                    if hand_label == "Left":
                        if left_cb['x'] <= wx < left_cb['x']+left_cb['w'] and left_cb['y'] <= wy < left_cb['y']+left_cb['h']:
                            rel = (wy - left_cb['y'])/left_cb['h'] if left_cb['h']>0 else 0.5
                            val = max(0.0,min(1.0,1.0-rel))
                            data = f"LVAL:{val:.4f}"
                            cv2.putText(frame, f"L Val: {val:.2f}", (left_px['x'], left_px['y']-10),
cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
                    else:
                        if right_cb['x'] <= wx < right_cb['x']+right_cb['w'] and right_cb['y'] <= wy < right_cb['y']+right_cb['h']:
                            rx = max(0.0,min(1.0,(wx-right_cb['x'])/right_cb['w']))
                            ry = max(0.0,min(1.0,(wy-right_cb['y'])/right_cb['h']))
                            invy = 1.0-ry
                            data = f"RPOS:{rx:.4f},{invy:.4f},{wz:.4f}"
                            cv2.putText(frame, f"R Rel: {rx:.2f},{invy:.2f}", (right_px['x'], right_px['y']-10),
cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
                    if data and sock:
                        try: sock.sendto(data.encode(), (UDP_IP,UDP_PORT))
                        except: pass

            # enqueue for GUI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            while frame_q.qsize()>=frame_q.maxsize:
                try: frame_q.get_nowait()
                except: break
            frame_q.put(rgb_frame, block=False)
            frame_count +=1

    except Exception as e:
        print(f"[PY THREAD ERROR] {e}"); traceback.print_exc()
    finally:
        stop_flag.set()
        if cap_thread.is_alive(): cap_thread.join(timeout=1.0)
        if cap and cap.isOpened(): cap.release()
        if hands: hands.close()
        if sock: sock.close()
        print("[PY THREAD] Cleaning up video thread...")


# --- Tkinter GUI Application ---
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
        self._update_slider_labels()
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

        # --- MODIFIED create_slider_row ---
        def create_slider_row(parent, label_text, double_var, label_var, row_num, col_offset=0):
            ttk.Label(parent, text=label_text).grid(row=row_num, column=0+col_offset, padx=(0, 5), sticky=tk.E)
            # Use 'command' for live label update during drag
            scale = ttk.Scale(parent, variable=double_var, from_=0.0, to=1.0, orient=tk.HORIZONTAL, length=150,
                              command=lambda val, dv=double_var, lv=label_var: self._handle_slider_update_label_only(dv, lv))
            # NO extra bindings needed for standard behavior
            scale.grid(row=row_num, column=1+col_offset, padx=(0, 5), sticky=tk.EW)
            label = ttk.Label(parent, textvariable=label_var, width=5)
            label.grid(row=row_num, column=2+col_offset, padx=(0, 15), sticky=tk.W)
            return scale, label
        # --- END MODIFICATION ---

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
        self.zone_controls = [
            self.left_x_scale, self.left_y_scale, self.left_w_scale, self.left_h_scale,
            self.right_x_scale, self.right_y_scale, self.right_w_scale, self.right_h_scale
        ]

        self.video_label = tk.Label(self.main_frame, bg="black")
        self.video_label.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        print("[PY GUI] Initializing camera lists...")
        self.populate_camera_list()
        print("[PY GUI] Camera lists populated.")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._draw_placeholder()
        print("[PY GUI] Initial placeholder displayed.")
        self.attempt_auto_start()

    def _load_config(self):
        print("[PY CONFIG] Loading configuration...")
        self.left_cb_x_var.set(DEFAULT_LEFT_CB['x'])
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
                with open(CONFIG_FILE, 'r') as f: config_data = json.load(f)
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
        self._update_slider_labels()

    def _save_config(self):
        print("[PY CONFIG] Saving configuration...")
        config_data = {
            'camera_index': self.selected_camera_index.get(),
            'resolution_string': self.selected_resolution_str.get(),
            'left_control_box': {'x': self.left_cb_x_var.get(), 'y': self.left_cb_y_var.get(), 'w': self.left_cb_w_var.get(), 'h': self.left_cb_h_var.get()},
            'right_control_box': {'x': self.right_cb_x_var.get(), 'y': self.right_cb_y_var.get(), 'w': self.right_cb_w_var.get(), 'h': self.right_cb_h_var.get()}
        }
        try:
            with open(CONFIG_FILE, 'w') as f: json.dump(config_data, f, indent=4)
            print("[PY CONFIG] Configuration saved.")
        except Exception as e:
            print(f"[PY CONFIG ERROR] Saving configuration: {e}"); traceback.print_exc()

    def _update_slider_labels(self):
        self.left_cb_x_label_var.set(f"{self.left_cb_x_var.get():.2f}")
        self.left_cb_y_label_var.set(f"{self.left_cb_y_var.get():.2f}")
        self.left_cb_w_label_var.set(f"{self.left_cb_w_var.get():.2f}")
        self.left_cb_h_label_var.set(f"{self.left_cb_h_var.get():.2f}")
        self.right_cb_x_label_var.set(f"{self.right_cb_x_var.get():.2f}")
        self.right_cb_y_label_var.set(f"{self.right_cb_y_var.get():.2f}")
        self.right_cb_w_label_var.set(f"{self.right_cb_w_var.get():.2f}")
        self.right_cb_h_label_var.set(f"{self.right_cb_h_var.get():.2f}")

    def _update_current_dimensions(self):
        selected_res_str = self.selected_resolution_str.get()
        try:
            res_part = selected_res_str.split(' ')[0]
            w_str, h_str = res_part.split('x')
            self.current_width = int(w_str)
            self.current_height = int(h_str)
        except Exception as e:
            print(f"[PY GUI WARNING] Parsing resolution '{selected_res_str}'. Using {self.current_width}x{self.current_height}. Error: {e}")
            if self.current_width <= 0 or self.current_height <= 0:
                 self.current_width = DEFAULT_WIDTH; self.current_height = DEFAULT_HEIGHT

    def _draw_placeholder(self):
        if self.is_tracking: return
        try:
            if not self.video_label.winfo_exists(): return
            self._update_current_dimensions()
            w = self.current_width; h = self.current_height
            if w <= 0 or h <= 0: w, h = DEFAULT_WIDTH, DEFAULT_HEIGHT
            placeholder_img_np = np.zeros((h, w, 3), dtype=np.uint8)
            lx, ly = self.left_cb_x_var.get(), self.left_cb_y_var.get()
            lw, lh = self.left_cb_w_var.get(), self.left_cb_h_var.get()
            rx, ry = self.right_cb_x_var.get(), self.right_cb_y_var.get()
            rw, rh = self.right_cb_w_var.get(), self.right_cb_h_var.get()
            l_x1, l_y1 = int(lx * w), int(ly * h); l_x2, l_y2 = int((lx + lw) * w), int((ly + lh) * h)
            r_x1, r_y1 = int(rx * w), int(ry * h); r_x2, r_y2 = int((rx + rw) * w), int((ry + rh) * h)
            cv2.rectangle(placeholder_img_np, (l_x1, l_y1), (l_x2, l_y2), (0, 0, 255), 2)
            cv2.rectangle(placeholder_img_np, (r_x1, r_y1), (r_x2, r_y2), (255, 255, 0), 2)
            rgb_placeholder = cv2.cvtColor(placeholder_img_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_placeholder)
            photo = ImageTk.PhotoImage(image=pil_img, master=self.root)
            self.placeholder_photo = photo
            self.video_label.config(image=self.placeholder_photo, text="")
            self.video_label.image = self.placeholder_photo
        except tk.TclError as e: print(f"[PY GUI WARNING] TclError drawing placeholder: {e}")
        except Exception as e:
            print(f"[PY GUI ERROR] Drawing placeholder: {e}"); traceback.print_exc()
            try:
                if self.video_label.winfo_exists():
                    self.video_label.config(image='', text="Error drawing zones", fg="red", bg="black")
                    self.video_label.image = None
            except: pass

    # --- Handles slider value changes DURING DRAG (updates label only) ---
    def _handle_slider_update_label_only(self, double_var, label_var):
        """Updates the text label associated with a slider."""
        try:
            current_value = double_var.get()
            label_var.set(f"{current_value:.2f}")
        except tk.TclError as e:
            print(f"[PY GUI CB WARNING] TclError updating label on drag: {e}")
        except Exception as e:
            print(f"[PY GUI CB ERROR] Error updating label on drag: {e}")

    # --- DELETE _handle_slider_click and _handle_slider_drag ---
    # (They are no longer needed with this standard behavior)

    def _set_control_state(self, state):
        """Enable/disable controls based on tracking state."""
        self.camera_combobox.config(state="disabled" if state == "disabled" else "readonly")
        res_state = "disabled"
        if state != "disabled" and self.resolution_combobox['values'] and self.resolution_combobox['values'][0] not in ["N/A", "Error", "No cameras found"]:
            res_state = "readonly"
        self.resolution_combobox.config(state=res_state)
        pass # Sliders always enabled

    def toggle_tracking(self):
        global video_thread, stop_event
        if self.is_tracking:
            print("[PY GUI] Stopping tracking...")
            stop_event.set()
            self.toggle_button.config(text="Stopping...", state="disabled")
            self.root.update_idletasks()
            if video_thread is not None and video_thread.is_alive():
                 video_thread.join(timeout=2.0)
                 if video_thread.is_alive(): print("[PY GUI WARNING] Video thread join timeout.")
                 video_thread = None
            self.is_tracking = False
            self.update_gui_frame_scheduled = False
            self.toggle_button.config(text="Start Tracking", state="normal")
            self._set_control_state("normal")
            self._draw_placeholder()
            print("[PY GUI] Tracking stopped.")
        else:
            print("[PY GUI] Starting tracking...")
            cam_index = self.selected_camera_index.get()
            if cam_index == -1: messagebox.showerror("Error", "No valid camera selected."); return
            self._update_current_dimensions()
            if self.current_width <= 0 or self.current_height <= 0:
                messagebox.showerror("Error", f"Invalid resolution dimensions ({self.current_width}x{self.current_height}). Cannot start tracking.")
                return
            tracking_settings = {
                "camera_index": cam_index, "width": self.current_width, "height": self.current_height,
                "left_x_var": self.left_cb_x_var, "left_y_var": self.left_cb_y_var,
                "left_w_var": self.left_cb_w_var, "left_h_var": self.left_cb_h_var,
                "right_x_var": self.right_cb_x_var, "right_y_var": self.right_cb_y_var,
                "right_w_var": self.right_cb_w_var, "right_h_var": self.right_cb_h_var
            }
            print(f"[PY GUI] Tracking settings prepared.")
            self.toggle_button.config(text="Stop Tracking", state="normal")
            self._set_control_state("disabled")
            self.placeholder_photo = None
            if self.video_label.winfo_exists():
                 self.video_label.config(image='', bg="black", text="Starting Camera...")
            self.root.update_idletasks()
            stop_event.clear()
            video_thread = threading.Thread(target=video_processing_loop,
                                            args=(tracking_settings, frame_queue, stop_event),
                                            daemon=True)
            video_thread.start()
            self.is_tracking = True
            self.update_gui_frame_scheduled = True
            self.update_gui_frame()
            print("[PY GUI] Tracking started.")

    def attempt_auto_start(self):
        print("[PY GUI] Attempting auto-start...")
        if self.selected_camera_index.get() != -1:
            if self.current_width > 0 and self.current_height > 0:
                print("[PY GUI] Valid camera and resolution found, auto-starting.")
                self.root.after(100, self.toggle_tracking)
            else:
                print("[PY GUI] Auto-start skipped: Invalid resolution dimensions.")
        else:
            print("[PY GUI] Auto-start skipped: No valid camera selected.")

    def populate_camera_list(self):
        print("[PY GUI] Populating camera list...")
        self.available_cameras = find_available_cameras()
        if not self.available_cameras:
            self.camera_combobox['values'] = ["No cameras found"]; self.camera_combobox.current(0); self.camera_combobox['state'] = 'disabled'
            self.toggle_button['state'] = 'disabled'
            self.resolution_combobox['state'] = 'disabled'; self.resolution_combobox['values'] = ["N/A"]; self.resolution_combobox.current(0)
            if self.video_label.winfo_exists():
                 self.video_label.config(text="No cameras found.", fg="red", image='')
                 self.video_label.image = None
            return
        camera_display_names = [name for name, index in self.available_cameras]
        self.camera_combobox['values'] = camera_display_names
        loaded_cam_idx = self.selected_camera_index.get()
        selected_list_idx = -1
        selected_display_name = ""
        for i, (name, index) in enumerate(self.available_cameras):
            if index == loaded_cam_idx:
                selected_list_idx = i
                selected_display_name = name
                break
        if selected_list_idx == -1:
            print(f"[PY GUI WARNING] Camera index {loaded_cam_idx} not found/invalid. Selecting default.")
            selected_list_idx = 0
            selected_display_name = self.available_cameras[0][0]
            self.selected_camera_index.set(self.available_cameras[0][1])
        self.camera_combobox.current(selected_list_idx)
        self.selected_camera_display_name.set(selected_display_name)
        self.camera_combobox.config(state="readonly")
        self.toggle_button.config(state="normal")
        self.update_resolution_list()

    def on_camera_selected(self, event=None):
        selected_display_name = self.selected_camera_display_name.get()
        selected_cam_device_index = -1
        for name, index in self.available_cameras:
            if name == selected_display_name:
                selected_cam_device_index = index
                break
        if selected_cam_device_index != -1:
            self.selected_camera_index.set(selected_cam_device_index)
            print(f"[PY GUI] Camera changed to: {selected_display_name} (Index: {selected_cam_device_index})")
            self.update_resolution_list()
        else:
            print(f"[PY GUI ERROR] Could not find index for camera '{selected_display_name}'")
            self.resolution_combobox['state'] = 'disabled'; self.resolution_combobox['values'] = ["Error"]; self.resolution_combobox.current(0)
            self.toggle_button['state'] = 'disabled';
            self._draw_placeholder()

    def on_resolution_selected(self, event=None):
        print(f"[PY GUI] Resolution selected: {self.selected_resolution_str.get()}")
        self._update_current_dimensions()
        if not self.is_tracking:
            self._draw_placeholder()

    def update_resolution_list(self):
        print("[PY GUI] Updating resolution list...")
        cam_index = self.selected_camera_index.get()
        if cam_index == -1:
             self.resolution_combobox['state'] = 'disabled'; self.toggle_button['state'] = 'disabled'
             self._update_current_dimensions(); self._draw_placeholder()
             return
        self.supported_resolutions = get_supported_resolutions(cam_index)
        loaded_res_str = self.selected_resolution_str.get()
        res_values = [DEFAULT_RESOLUTION_STRING]
        if self.supported_resolutions:
            res_values = [res[3] for res in self.supported_resolutions]
        self.resolution_combobox['values'] = res_values
        self.resolution_combobox['state'] = 'readonly'
        selected_idx = -1
        for i, res_str in enumerate(res_values):
            if res_str == loaded_res_str: selected_idx = i; break
        if selected_idx == -1:
            print(f"[PY GUI WARNING] Resolution '{loaded_res_str}' not found/supported. Selecting default/highest.")
            selected_idx = 0
            self.selected_resolution_str.set(res_values[selected_idx])
        self.resolution_combobox.current(selected_idx)
        self.toggle_button['state'] = 'normal'
        self._update_current_dimensions()

    def update_gui_frame(self):
        if not self.is_tracking or not self.update_gui_frame_scheduled or stop_event.is_set():
            return
        try:
            frame_rgb = frame_queue.get_nowait()
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img, master=self.root)
            if self.video_label.winfo_exists():
                self.video_label.config(image=img_tk, text="")
                self.video_label.image = img_tk
                self.placeholder_photo = None
            else:
                print("[PY GUI WARNING] video_label destroyed during update.")
                self.update_gui_frame_scheduled = False
                if self.is_tracking: self.toggle_tracking()
                return
            if self.root.winfo_exists():
                self.root.after(15, self.update_gui_frame)
            else:
                self.update_gui_frame_scheduled = False
        except queue.Empty:
            if self.root.winfo_exists():
                self.root.after(30, self.update_gui_frame)
            else:
                self.update_gui_frame_scheduled = False
        except tk.TclError as e: print(f"[PY GUI ERROR] Tkinter TclError in GUI update: {e}"); self.update_gui_frame_scheduled = False; self.toggle_tracking()
        except RuntimeError as e: print(f"[PY GUI ERROR] RuntimeError in GUI update: {e}"); traceback.print_exc(); self.update_gui_frame_scheduled = False
        except Exception as e: print(f"[PY GUI ERROR] GUI Update Error: {e}"); traceback.print_exc()

    def on_closing(self):
        print("[PY GUI] Window close requested.")
        self.update_gui_frame_scheduled = False
        if self.is_tracking:
            print("[PY GUI] Stopping tracking before closing...")
            stop_event.set()
            if video_thread is not None and video_thread.is_alive():
                video_thread.join(timeout=1.0)
        self.is_tracking = False
        self._save_config()
        print("[PY GUI] Destroying Tkinter window.")
        global sock
        if sock:
            try: sock.close(); print("[PY GUI] Cleaned up dangling UDP socket."); sock = None
            except Exception as e: print(f"[PY GUI ERROR] Error closing dangling socket: {e}")
        try:
            if self.root and self.root.winfo_exists(): self.root.destroy()
        except tk.TclError: print("[PY GUI] Window already destroyed.")

# --- End of HandTrackingApp Class ---


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Hand Tracking Application...")
    splash = None; main_root = None; app = None
    try:
        # splash = create_splash() # Optional splash
        main_root = tk.Tk()
        main_root.withdraw() # Hide while setting up
        main_root.title("Hand Tracking Control v10.3 (AutoStart, LiveUpdate, StandardSlider)") # Updated title
        print("Main: Initializing HandTrackingApp...")
        app = HandTrackingApp(main_root)
        print("Main: HandTrackingApp initialized.")
        # if splash: splash.destroy(); splash = None # Close splash if used
        print("Main: Showing main application window.")
        if main_root:
            main_root.deiconify(); main_root.lift(); main_root.focus_force()
            print("Main: Starting main event loop.")
            main_root.mainloop() # Blocks here until window is closed
    except Exception as e:
        print(f"\n--- FATAL STARTUP ERROR ---"); traceback.print_exc(); print(f"Error: {e}"); print("--------------------------")
        # if splash and splash.winfo_exists(): splash.destroy()
        try: # Attempt to show error dialog
             err_root = tk.Tk(); err_root.withdraw()
             messagebox.showerror("Application Startup Error", f"Failed to initialize.\n\nError:\n{e}\n\nSee console.")
             err_root.destroy()
        except Exception as msg_err: print(f"(Could not display final error: {msg_err})")
        finally:
             if main_root and main_root.winfo_exists(): main_root.destroy() # Ensure main window closes on error
    except KeyboardInterrupt:
         print("\nCtrl+C detected. Closing.")
         # if splash and splash.winfo_exists(): splash.destroy()
         if app: app.on_closing() # Trigger cleanup
         elif main_root and main_root.winfo_exists(): main_root.destroy()
    finally:
        # Final check for socket cleanup
        if sock:
            try: sock.close(); print("Closed socket in final cleanup.")
            except: pass
    print("Application finished.")
    sys.exit(0)