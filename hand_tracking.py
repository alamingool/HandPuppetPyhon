# hand_tracking_v10.6_SlidersForHNF_WithGUI.py
# - Added GUI sliders for Hand Near Face (HNF) zone.
# - LVAL/RPOS always sent if in zone.
# - RDIST (Thumb-Index) always sent for Right Hand (or 0 if not detected).
# - GFST:L_FIST/GFST:R_FIST sent as additional info on fist detection.
# - GFST_OFF:L_FIST/GFST_OFF:R_FIST sent on fist release/loss.
# - On-screen display for RDIST.
# - QTE: HAND_READY_FOR_BOW:1/0 (single hand near face) and HEAD_BOW:1/0 messages.
# - Includes full Tkinter GUI.

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
DEFAULT_RIGHT_CB = {"x": 0.6, "y": 0.2, "w": 0.3, "h": 0.6}
DEFAULT_LEFT_CB = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.6}
DEFAULT_HNF_ZONE = {"x_min": 0.25, "y_min": 0.1, "x_max": 0.75, "y_max": 0.5} # Default HNF Zone

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

left_hand_fist_state_previous_frame = False
right_hand_fist_state_previous_frame = False
FINGER_CLOSE_THRESHOLD = 0.055

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
    cap = None
    for i in range(max_cameras_to_check):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                available_cameras.append((f"Camera {i}", i))
        except Exception:
            continue
        finally:
            if cap is not None and cap.isOpened(): cap.release()
            elif cap is not None: cap.release()
    return available_cameras

def get_supported_resolutions(camera_index):
    supported = []
    cap_test = None
    try:
        cap_test = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap_test or not cap_test.isOpened(): return supported
        for width, height, aspect in COMMON_RESOLUTIONS:
            cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            time.sleep(0.05)
            actual_width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if abs(actual_width - width) < 20 and abs(actual_height - height) < 20:
                 res_tuple = (width, height)
                 if res_tuple not in [(w_s, h_s) for w_s, h_s, _, _ in supported]:
                     formatted = f"{width}x{height} ({aspect})"
                     supported.append((width, height, aspect, formatted))
    except Exception as e:
        print(f"[PY CAM RES ERR] Cam {camera_index}: {e}")
    finally:
        if cap_test is not None and cap_test.isOpened(): cap_test.release()
    supported.sort(key=lambda x: x[0], reverse=True)
    return supported

# --- Helper Functions for Hand Analysis ---
def calculate_distance_3d(p1, p2): return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)
def calculate_distance_2d(p1, p2): return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

def is_fist(hand_landmarks, handedness_label):
    if not hand_landmarks: return False
    tip_ids = [mp.solutions.hands.HandLandmark.THUMB_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP, mp.solutions.hands.HandLandmark.RING_FINGER_TIP, mp.solutions.hands.HandLandmark.PINKY_TIP]
    mcp_ids = [mp.solutions.hands.HandLandmark.THUMB_IP, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP, mp.solutions.hands.HandLandmark.RING_FINGER_MCP, mp.solutions.hands.HandLandmark.PINKY_MCP]
    closed=0
    for i in range(1,5):
        if calculate_distance_3d(hand_landmarks.landmark[tip_ids[i]], hand_landmarks.landmark[mcp_ids[i]]) < FINGER_CLOSE_THRESHOLD:
            closed+=1
    return closed>=3

# --- Video Processing Thread ---
def video_processing_loop(settings, frame_q, stop_flag):
    global sock, left_hand_fist_state_previous_frame, right_hand_fist_state_previous_frame
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    hands_solution = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1, static_image_mode=False, max_num_hands=2)
    pose_solution = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    cap = cv2.VideoCapture(settings['camera_index'], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[PY THREAD ERROR] Cannot open camera {settings['camera_index']}")
        stop_flag.set(); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
    time.sleep(0.2)
    
    if sock is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    capture_queue = queue.Queue(maxsize=1) 
    def capture_loop_fn():
        while not stop_flag.is_set():
            ret, raw_frame = cap.read()
            if not ret: 
                time.sleep(0.01)
                continue
            if capture_queue.full(): 
                try: capture_queue.get_nowait()
                except queue.Empty: pass
            capture_queue.put(raw_frame)
    cap_thread = threading.Thread(target=capture_loop_fn, daemon=True)
    cap_thread.start()

    qte_hand_ready_active = False 
    QTE_HAND_READY_DEBOUNCE_FRAMES = 7 
    qte_hand_ready_true_count = 0
    qte_hand_ready_false_count = 0

    qte_head_bow_active_sent_udp = False
    qte_head_bow_y_neutral = -1.0
    QTE_HEAD_BOW_Y_THRESHOLD = 0.06

    try:
        last_left_cb_values = DEFAULT_LEFT_CB.copy()
        last_right_cb_values = DEFAULT_RIGHT_CB.copy()
        # Initialize HNF zone from settings (which get them from sliders)
        # These will be updated each frame from the sliders
        _HAND_NEAR_FACE_X_MIN = settings.get('hnf_x_min_var', tk.DoubleVar(value=DEFAULT_HNF_ZONE['x_min'])).get()
        _HAND_NEAR_FACE_Y_MIN = settings.get('hnf_y_min_var', tk.DoubleVar(value=DEFAULT_HNF_ZONE['y_min'])).get()
        _HAND_NEAR_FACE_X_MAX = settings.get('hnf_x_max_var', tk.DoubleVar(value=DEFAULT_HNF_ZONE['x_max'])).get()
        _HAND_NEAR_FACE_Y_MAX = settings.get('hnf_y_max_var', tk.DoubleVar(value=DEFAULT_HNF_ZONE['y_max'])).get()


        while not stop_flag.is_set():
            try:
                frame = capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame = cv2.flip(frame, 1)
            
            try:
                left_cb = {k: settings[f'left_{k}_var'].get() for k in ['x','y','w','h']}
                right_cb = {k: settings[f'right_{k}_var'].get() for k in ['x','y','w','h']}
                last_left_cb_values, last_right_cb_values = left_cb.copy(), right_cb.copy()

                # Get Hand Near Face zone values from GUI sliders for this frame
                _HAND_NEAR_FACE_X_MIN = settings['hnf_x_min_var'].get()
                _HAND_NEAR_FACE_Y_MIN = settings['hnf_y_min_var'].get()
                _HAND_NEAR_FACE_X_MAX = settings['hnf_x_max_var'].get()
                _HAND_NEAR_FACE_Y_MAX = settings['hnf_y_max_var'].get()
                if _HAND_NEAR_FACE_X_MAX < _HAND_NEAR_FACE_X_MIN: _HAND_NEAR_FACE_X_MAX = _HAND_NEAR_FACE_X_MIN
                if _HAND_NEAR_FACE_Y_MAX < _HAND_NEAR_FACE_Y_MIN: _HAND_NEAR_FACE_Y_MAX = _HAND_NEAR_FACE_Y_MIN

            except Exception: # Fallback if GUI vars are not accessible
                left_cb, right_cb = last_left_cb_values.copy(), last_right_cb_values.copy()
                # Use last known good HNF values or defaults if settings can't be read
                # This part is tricky as we don't have last_hnf_values like control boxes
                # For simplicity, we'll just rely on the initial _HAND_NEAR_FACE values if this fails
            
            h_frame, w_frame, _ = frame.shape
            left_px = {k: int(left_cb[k] * (w_frame if k in 'xw' else h_frame)) for k in left_cb}
            right_px = {k: int(right_cb[k] * (w_frame if k in 'xw' else h_frame)) for k in right_cb}

            cv2.rectangle(frame, (left_px['x'], left_px['y']), (left_px['x'] + left_px['w'], left_px['y'] + left_px['h']), (0, 0, 255), 2)
            cv2.rectangle(frame, (right_px['x'], right_px['y']), (right_px['x'] + right_px['w'], right_px['y'] + right_px['h']), (255, 255, 0), 2)

            hnf_x1_px = int(_HAND_NEAR_FACE_X_MIN * w_frame)
            hnf_y1_px = int(_HAND_NEAR_FACE_Y_MIN * h_frame)
            hnf_x2_px = int(_HAND_NEAR_FACE_X_MAX * w_frame)
            hnf_y2_px = int(_HAND_NEAR_FACE_Y_MAX * h_frame)
            cv2.rectangle(frame, (hnf_x1_px, hnf_y1_px), (hnf_x2_px, hnf_y2_px), (0, 255, 128), 1)
            cv2.putText(frame, "HandReadyZone", (hnf_x1_px, hnf_y1_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 128), 1)

            rgb_frame_for_mediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame_for_mediapipe.flags.writeable = False
            hands_results = hands_solution.process(rgb_frame_for_mediapipe)
            pose_results = pose_solution.process(rgb_frame_for_mediapipe)
            rgb_frame_for_mediapipe.flags.writeable = True

            current_frame_has_left_fist = False
            current_frame_has_right_fist = False
            right_hand_detected_this_frame = False
            current_frame_hand_ready_raw = False

            if hands_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    handedness_label = hands_results.multi_handedness[hand_idx].classification[0].label
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1))
                    
                    is_current_hand_a_fist_now = is_fist(hand_landmarks, handedness_label)
                    messages_to_send_this_frame_for_hand = []

                    if handedness_label == "Left":
                        current_frame_has_left_fist = is_current_hand_a_fist_now
                        tip_l = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        if left_px['x'] <= tip_l.x * w_frame < left_px['x'] + left_px['w'] and \
                           left_px['y'] <= tip_l.y * h_frame < left_px['y'] + left_px['h']:
                            rel_y_l = (tip_l.y * h_frame - left_px['y']) / left_px['h'] if left_px['h'] > 0 else 0.5
                            val_l = max(0.0, min(1.0, 1.0 - rel_y_l))
                            messages_to_send_this_frame_for_hand.append(f"LVAL:{val_l:.4f}")
                            cv2.putText(frame, f"LVal: {val_l:.2f}", (left_px['x'], left_px['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200),2)
                        if is_current_hand_a_fist_now:
                            cv2.putText(frame, "L_FIST", (left_px['x'], left_px['y'] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,100),2)
                            if not left_hand_fist_state_previous_frame:
                                messages_to_send_this_frame_for_hand.append("GFST:L_FIST")
                    
                    elif handedness_label == "Right":
                        right_hand_detected_this_frame = True
                        current_frame_has_right_fist = is_current_hand_a_fist_now
                        tip_r = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        if right_px['x'] <= tip_r.x * w_frame < right_px['x'] + right_px['w'] and \
                           right_px['y'] <= tip_r.y * h_frame < right_px['y'] + right_px['h']:
                            rx_r = (tip_r.x * w_frame - right_px['x']) / right_px['w'] if right_px['w'] > 0 else 0.5
                            ry_r = (tip_r.y * h_frame - right_px['y']) / right_px['h'] if right_px['h'] > 0 else 0.5
                            messages_to_send_this_frame_for_hand.append(f"RPOS:{max(0,min(1,rx_r)):.4f},{max(0,min(1,1-ry_r)):.4f},{tip_r.z:.4f}")
                            cv2.putText(frame, f"RPos: {rx_r:.2f},{1-ry_r:.2f}", (right_px['x'], right_px['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0),2)
                        if len(hand_landmarks.landmark) > max(mp.solutions.hands.HandLandmark.THUMB_TIP.value, mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value):
                            try:
                                thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                                index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                                dist_val = calculate_distance_2d(thumb_tip, index_tip)
                                messages_to_send_this_frame_for_hand.append(f"RDIST:{dist_val:.4f}")
                                rdist_text_y_offset = 40 if is_current_hand_a_fist_now else 25
                                cv2.putText(frame, f"RDist: {dist_val:.3f}", (right_px['x'], right_px['y'] - rdist_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200),2)
                            except Exception: pass
                        if is_current_hand_a_fist_now:
                            r_fist_text_y_offset = 55 if any("RDIST" in m for m in messages_to_send_this_frame_for_hand) else 25
                            if not any("RPOS" in m for m in messages_to_send_this_frame_for_hand) and not any("RDIST" in m for m in messages_to_send_this_frame_for_hand):
                                r_fist_text_y_offset = 10
                            cv2.putText(frame, "R_FIST", (right_px['x'], right_px['y'] - r_fist_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,0),2)
                            if not right_hand_fist_state_previous_frame:
                                messages_to_send_this_frame_for_hand.append("GFST:R_FIST")

                    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                    if (_HAND_NEAR_FACE_X_MIN < wrist.x < _HAND_NEAR_FACE_X_MAX and \
                        _HAND_NEAR_FACE_Y_MIN < wrist.y < _HAND_NEAR_FACE_Y_MAX):
                        current_frame_hand_ready_raw = True 
                        cv2.circle(frame, (int(wrist.x * w_frame), int(wrist.y * h_frame)), 15, (50, 205, 50), 3)
                        
                    for msg_to_send in messages_to_send_this_frame_for_hand:
                        if sock:
                            try: sock.sendto(msg_to_send.encode(), (UDP_IP, UDP_PORT))
                            except Exception as e_send: print(f"[PY SEND ERROR] {e_send} for {msg_to_send}")
            
            if current_frame_hand_ready_raw:
                qte_hand_ready_true_count += 1
                qte_hand_ready_false_count = 0
                if qte_hand_ready_true_count >= QTE_HAND_READY_DEBOUNCE_FRAMES and not qte_hand_ready_active:
                    qte_hand_ready_active = True
                    if sock: sock.sendto("HAND_READY_FOR_BOW:1".encode(), (UDP_IP, UDP_PORT))
            else:
                qte_hand_ready_false_count += 1
                qte_hand_ready_true_count = 0
                if qte_hand_ready_false_count >= QTE_HAND_READY_DEBOUNCE_FRAMES and qte_hand_ready_active:
                    qte_hand_ready_active = False
                    if sock: sock.sendto("HAND_READY_FOR_BOW:0".encode(), (UDP_IP, UDP_PORT))

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                         mp_drawing.DrawingSpec(color=(0,128,0), thickness=1, circle_radius=1))
                try:
                    landmarks = pose_results.pose_landmarks.landmark
                    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                    if nose.visibility > 0.5 and left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
                        nose_y = nose.y
                        if qte_head_bow_y_neutral < 0 or not qte_head_bow_active_sent_udp:
                            if nose_y < shoulder_mid_y + (QTE_HEAD_BOW_Y_THRESHOLD * 0.3):
                                qte_head_bow_y_neutral = shoulder_mid_y
                        if qte_head_bow_y_neutral > 0 and nose_y > qte_head_bow_y_neutral + QTE_HEAD_BOW_Y_THRESHOLD:
                            if not qte_head_bow_active_sent_udp:
                                if sock: sock.sendto("HEAD_BOW:1".encode(), (UDP_IP, UDP_PORT))
                                qte_head_bow_active_sent_udp = True
                                cv2.putText(frame, "BOW", (int(nose.x*w_frame), int(nose.y*h_frame)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                        elif qte_head_bow_active_sent_udp and qte_head_bow_y_neutral > 0 and \
                             nose_y < qte_head_bow_y_neutral + (QTE_HEAD_BOW_Y_THRESHOLD * 0.5):
                            if sock: sock.sendto("HEAD_BOW:0".encode(), (UDP_IP, UDP_PORT))
                            qte_head_bow_active_sent_udp = False
                            qte_head_bow_y_neutral = -1.0
                except IndexError: print("[PY POSE ERR] Index error accessing pose landmarks.")
                except Exception as e_pose: print(f"[PY POSE ERR] Error in pose processing: {e_pose}")

            if not right_hand_detected_this_frame:
                if sock:
                    try: sock.sendto("RDIST:0.0000".encode(), (UDP_IP, UDP_PORT))
                    except Exception as e_zero: print(f"[PY SEND ERROR] RDIST 0: {e_zero}")
            
            if not current_frame_has_left_fist and left_hand_fist_state_previous_frame:
                if sock: 
                    try: sock.sendto("GFST_OFF:L_FIST".encode(), (UDP_IP, UDP_PORT))
                    except Exception as e_off: print(f"[PY SEND ERROR] GFST_OFF L: {e_off}")
            
            if not current_frame_has_right_fist and right_hand_fist_state_previous_frame:
                if sock: 
                    try: sock.sendto("GFST_OFF:R_FIST".encode(), (UDP_IP, UDP_PORT))
                    except Exception as e_off: print(f"[PY SEND ERROR] GFST_OFF R: {e_off}")

            left_hand_fist_state_previous_frame = current_frame_has_left_fist
            right_hand_fist_state_previous_frame = current_frame_has_right_fist
            
            if qte_hand_ready_active:
                cv2.putText(frame, "HAND_READY (Sent:1)", (10, h_frame - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "HAND_NOT_READY (Sent:0)", (10, h_frame - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if current_frame_hand_ready_raw:
                 cv2.putText(frame, "RAW HAND_READY: YES", (10, h_frame - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            if qte_head_bow_active_sent_udp:
                cv2.putText(frame, "HEAD_BOW (Sent:1)", (10, h_frame - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            gui_display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_q.full():
                try: frame_q.get_nowait()
                except queue.Empty: pass
            try: frame_q.put_nowait(gui_display_frame)
            except queue.Full: pass

    except Exception as e_loop:
        print(f"[PY THREAD ERROR] Unhandled exception in video processing loop: {e_loop}")
        traceback.print_exc()
    finally:
        print("[PY THREAD] Initiating cleanup for video thread...")
        stop_flag.set()
        if 'cap_thread' in locals() and cap_thread.is_alive(): cap_thread.join(timeout=0.5)
        if 'cap' in locals() and cap and cap.isOpened(): cap.release()
        if 'hands_solution' in locals() and hands_solution: hands_solution.close()
        if 'pose_solution' in locals() and pose_solution: pose_solution.close()
        print("[PY THREAD] Video thread cleaned up.")

# --- Tkinter GUI Application Class (HandTrackingApp) ---
class HandTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.minsize(750, 750) # Increased min height for new sliders
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

        # Variables for Hand Near Face Zone sliders
        self.hnf_x_min_var = tk.DoubleVar()
        self.hnf_y_min_var = tk.DoubleVar()
        self.hnf_x_max_var = tk.DoubleVar()
        self.hnf_y_max_var = tk.DoubleVar()

        # Variables for Hand Near Face Zone slider value labels
        self.hnf_x_min_label_var = tk.StringVar()
        self.hnf_y_min_label_var = tk.StringVar()
        self.hnf_x_max_label_var = tk.StringVar()
        self.hnf_y_max_label_var = tk.StringVar()

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
        
        # Hand Near Face Zone Sliders Frame
        self.hnf_zone_frame = ttk.LabelFrame(self.main_frame, text="Hand Ready Zone (QTE)", padding="10")
        self.hnf_zone_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.hnf_zone_frame.columnconfigure(1, weight=1); self.hnf_zone_frame.columnconfigure(4, weight=1)

        create_slider_row(self.hnf_zone_frame, "HNF X Min:", self.hnf_x_min_var, self.hnf_x_min_label_var, 0, col_offset=0)
        create_slider_row(self.hnf_zone_frame, "HNF Y Min:", self.hnf_y_min_var, self.hnf_y_min_label_var, 1, col_offset=0)
        create_slider_row(self.hnf_zone_frame, "HNF X Max:", self.hnf_x_max_var, self.hnf_x_max_label_var, 0, col_offset=3)
        create_slider_row(self.hnf_zone_frame, "HNF Y Max:", self.hnf_y_max_var, self.hnf_y_max_label_var, 1, col_offset=3)

        self.video_label = tk.Label(self.main_frame, bg="black") 
        self.video_label.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.populate_camera_list()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) 
        self._draw_placeholder() 
        self.attempt_auto_start() 

    def _load_config(self):
        self.left_cb_x_var.set(DEFAULT_LEFT_CB['x']) 
        self.left_cb_y_var.set(DEFAULT_LEFT_CB['y'])
        self.left_cb_w_var.set(DEFAULT_LEFT_CB['w'])
        self.left_cb_h_var.set(DEFAULT_LEFT_CB['h'])
        self.right_cb_x_var.set(DEFAULT_RIGHT_CB['x'])
        self.right_cb_y_var.set(DEFAULT_RIGHT_CB['y'])
        self.right_cb_w_var.set(DEFAULT_RIGHT_CB['w'])
        self.right_cb_h_var.set(DEFAULT_RIGHT_CB['h'])
        
        self.hnf_x_min_var.set(DEFAULT_HNF_ZONE['x_min'])
        self.hnf_y_min_var.set(DEFAULT_HNF_ZONE['y_min'])
        self.hnf_x_max_var.set(DEFAULT_HNF_ZONE['x_max'])
        self.hnf_y_max_var.set(DEFAULT_HNF_ZONE['y_max'])

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
                # ... (rest of left_cb and right_cb loading)
                self.left_cb_y_var.set(lc.get('y', DEFAULT_LEFT_CB['y']))
                self.left_cb_w_var.set(lc.get('w', DEFAULT_LEFT_CB['w']))
                self.left_cb_h_var.set(lc.get('h', DEFAULT_LEFT_CB['h']))
                self.right_cb_x_var.set(rc.get('x', DEFAULT_RIGHT_CB['x']))
                self.right_cb_y_var.set(rc.get('y', DEFAULT_RIGHT_CB['y']))
                self.right_cb_w_var.set(rc.get('w', DEFAULT_RIGHT_CB['w']))
                self.right_cb_h_var.set(rc.get('h', DEFAULT_RIGHT_CB['h']))

                hnf_zone_config = config_data.get('hand_near_face_zone', DEFAULT_HNF_ZONE)
                self.hnf_x_min_var.set(hnf_zone_config.get('x_min', DEFAULT_HNF_ZONE['x_min']))
                self.hnf_y_min_var.set(hnf_zone_config.get('y_min', DEFAULT_HNF_ZONE['y_min']))
                self.hnf_x_max_var.set(hnf_zone_config.get('x_max', DEFAULT_HNF_ZONE['x_max']))
                self.hnf_y_max_var.set(hnf_zone_config.get('y_max', DEFAULT_HNF_ZONE['y_max']))

            except Exception as e:
                print(f"[PY CONFIG ERROR] Loading configuration: {e}. Using defaults.")
        self._update_slider_labels() 

    def _save_config(self):
        config_data = {
            'camera_index': self.selected_camera_index.get(),
            'resolution_string': self.selected_resolution_str.get(),
            'left_control_box': {k: getattr(self, f'left_cb_{k}_var').get() for k in ['x', 'y', 'w', 'h']},
            'right_control_box': {k: getattr(self, f'right_cb_{k}_var').get() for k in ['x', 'y', 'w', 'h']},
            'hand_near_face_zone': {
                'x_min': self.hnf_x_min_var.get(),
                'y_min': self.hnf_y_min_var.get(),
                'x_max': self.hnf_x_max_var.get(),
                'y_max': self.hnf_y_max_var.get()
            }
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"[PY CONFIG ERROR] Saving configuration: {e}")

    def _update_slider_labels(self):
        for hand_prefix in ['left', 'right']:
            for dim_suffix in ['x', 'y', 'w', 'h']:
                var_instance = getattr(self, f'{hand_prefix}_cb_{dim_suffix}_var')
                label_var_instance = getattr(self, f'{hand_prefix}_cb_{dim_suffix}_label_var')
                label_var_instance.set(f"{var_instance.get():.2f}")
        
        for dim_suffix in ['x_min', 'y_min', 'x_max', 'y_max']:
            var_instance = getattr(self, f'hnf_{dim_suffix}_var')
            label_var_instance = getattr(self, f'hnf_{dim_suffix}_label_var')
            label_var_instance.set(f"{var_instance.get():.2f}")


    def _update_current_dimensions(self):
        selected_res_str = self.selected_resolution_str.get()
        try:
            res_part = selected_res_str.split(' ')[0]
            w_str, h_str = res_part.split('x')
            self.current_width = int(w_str)
            self.current_height = int(h_str)
            if self.current_width <= 0 or self.current_height <= 0:
                raise ValueError("Dimensions must be positive.")
        except Exception:
            self.current_width, self.current_height = DEFAULT_WIDTH, DEFAULT_HEIGHT
            self.selected_resolution_str.set(DEFAULT_RESOLUTION_STRING)

    def _draw_placeholder(self):
        if self.is_tracking: return
        try:
            if not self.video_label.winfo_exists(): return
            self._update_current_dimensions() 
            w, h = self.current_width, self.current_height
            placeholder_img_np = np.zeros((h, w, 3), dtype=np.uint8)
            lx, ly, lw, lh = self.left_cb_x_var.get(), self.left_cb_y_var.get(), self.left_cb_w_var.get(), self.left_cb_h_var.get()
            rx, ry, rw, rh = self.right_cb_x_var.get(), self.right_cb_y_var.get(), self.right_cb_w_var.get(), self.right_cb_h_var.get()
            l_x1, l_y1, l_x2, l_y2 = int(lx*w), int(ly*h), int((lx+lw)*w), int((ly+lh)*h)
            r_x1, r_y1, r_x2, r_y2 = int(rx*w), int(ry*h), int((rx+rw)*w), int((ry+rh)*h)
            cv2.rectangle(placeholder_img_np, (l_x1, l_y1), (l_x2, l_y2), (0, 0, 255), 2)
            cv2.rectangle(placeholder_img_np, (r_x1, r_y1), (r_x2, r_y2), (255, 255, 0), 2)

            hnf_x1_px = int(self.hnf_x_min_var.get() * w)
            hnf_y1_px = int(self.hnf_y_min_var.get() * h)
            hnf_x2_px = int(self.hnf_x_max_var.get() * w)
            hnf_y2_px = int(self.hnf_y_max_var.get() * h)
            cv2.rectangle(placeholder_img_np, (hnf_x1_px, hnf_y1_px), (hnf_x2_px, hnf_y2_px), (0, 255, 128), 1)

            rgb_placeholder = cv2.cvtColor(placeholder_img_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_placeholder)
            photo = ImageTk.PhotoImage(image=pil_img, master=self.root)
            
            self.placeholder_photo = photo 
            self.video_label.config(image=self.placeholder_photo, text="")
            self.video_label.image = self.placeholder_photo
        except tk.TclError: pass 
        except Exception as e: print(f"[PY GUI ERROR] Exception drawing placeholder: {e}"); traceback.print_exc()

    def _handle_slider_update_label_only(self, double_var, label_var):
        try: label_var.set(f"{double_var.get():.2f}")
        except tk.TclError: pass 
        except Exception as e: print(f"[PY GUI CB ERROR] Updating label on drag: {e}")
        if not self.is_tracking: self._draw_placeholder()

    def _set_control_state(self, new_state_str):
        tk_state = "disabled" if new_state_str == "disabled" else "readonly"
        self.camera_combobox.config(state=tk_state)
        
        res_combo_actual_state = "disabled"
        if tk_state == "readonly":
            current_res_values = self.resolution_combobox.cget('values')
            if current_res_values and current_res_values[0] not in ["N/A", "Error", "No cameras found"]:
                res_combo_actual_state = "readonly"
        self.resolution_combobox.config(state=res_combo_actual_state)

    def toggle_tracking(self):
        global video_thread, stop_event, sock
        if self.is_tracking:
            stop_event.set()
            self.toggle_button.config(text="Stopping...", state="disabled")
            self.root.update_idletasks()
            if video_thread and video_thread.is_alive():
                 video_thread.join(timeout=1.0)
            video_thread = None
            self.is_tracking = False
            self.update_gui_frame_scheduled = False
            self.toggle_button.config(text="Start Tracking", state="normal")
            self._set_control_state("normal")
            self._draw_placeholder()
        else:
            cam_idx_to_use = self.selected_camera_index.get()
            if cam_idx_to_use == -1:
                messagebox.showerror("Error", "No valid camera selected. Cannot start tracking.")
                return
            self._update_current_dimensions()
            if self.current_width <= 0 or self.current_height <= 0:
                messagebox.showerror("Error", f"Invalid resolution dimensions ({self.current_width}x{self.current_height}).")
                return

            tracking_settings = {
                "camera_index": cam_idx_to_use,
                "width": self.current_width,
                "height": self.current_height,
                **{f'{h}_{d}_var': getattr(self, f'{h}_cb_{d}_var') for h in ['left', 'right'] for d in ['x', 'y', 'w', 'h']},
                'hnf_x_min_var': self.hnf_x_min_var, # Pass HNF DoubleVars
                'hnf_y_min_var': self.hnf_y_min_var,
                'hnf_x_max_var': self.hnf_x_max_var,
                'hnf_y_max_var': self.hnf_y_max_var,
            }
            self.toggle_button.config(text="Stop Tracking", state="normal")
            self._set_control_state("disabled")
            self.placeholder_photo = None
            if self.video_label.winfo_exists():
                self.video_label.config(image='', bg="black", text="Starting Camera...")
            self.root.update_idletasks()

            if sock is None:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            stop_event.clear()
            video_thread = threading.Thread(target=video_processing_loop, args=(tracking_settings, frame_queue, stop_event), daemon=True)
            video_thread.start()
            self.is_tracking = True
            self.update_gui_frame_scheduled = True
            self.update_gui_frame()

    def attempt_auto_start(self):
        if self.selected_camera_index.get() != -1:
            self._update_current_dimensions()
            if self.current_width > 0 and self.current_height > 0:
                self.root.after(100, self.toggle_tracking)

    def populate_camera_list(self):
        self.available_cameras = find_available_cameras()
        if not self.available_cameras:
            self.camera_combobox['values'] = ["No cameras found"]
            self.camera_combobox.current(0); self.camera_combobox['state'] = 'disabled'
            self.toggle_button['state'] = 'disabled'
            self.resolution_combobox['state'] = 'disabled'; self.resolution_combobox['values'] = ["N/A"]; self.resolution_combobox.current(0)
            if self.video_label.winfo_exists():
                self.video_label.config(text="No cameras found. Connect a camera.", fg="red", image=''); self.video_label.image = None
            return

        camera_display_names = [name for name, index in self.available_cameras]
        self.camera_combobox['values'] = camera_display_names
        
        loaded_cam_device_idx_from_config = self.selected_camera_index.get()
        selected_list_idx_for_combobox = next((i for i, (_, dev_idx) in enumerate(self.available_cameras) if dev_idx == loaded_cam_device_idx_from_config), 0) 
        
        if self.available_cameras:
            actual_selected_device_index = self.available_cameras[selected_list_idx_for_combobox][1]
            self.selected_camera_index.set(actual_selected_device_index)
            self.selected_camera_display_name.set(self.available_cameras[selected_list_idx_for_combobox][0])
        
        self.camera_combobox.current(selected_list_idx_for_combobox)
        self.camera_combobox.config(state="readonly")
        self.toggle_button.config(state="normal")
        self.update_resolution_list()

    def on_camera_selected(self, event=None):
        selected_display_name = self.selected_camera_display_name.get()
        selected_cam_device_index = next((idx for name, idx in self.available_cameras if name == selected_display_name), -1)
        
        if selected_cam_device_index != -1:
            self.selected_camera_index.set(selected_cam_device_index)
            self.update_resolution_list()
        else:
            self.resolution_combobox['state'] = 'disabled'; self.resolution_combobox['values'] = ["Error"]; self.resolution_combobox.current(0)
            self.toggle_button['state'] = 'disabled'; self._draw_placeholder()

    def on_resolution_selected(self, event=None):
        self._update_current_dimensions()
        if not self.is_tracking: self._draw_placeholder()

    def update_resolution_list(self):
        current_cam_device_idx = self.selected_camera_index.get()
        if current_cam_device_idx == -1:
             self.resolution_combobox['state'] = 'disabled'; self.resolution_combobox['values'] = ["N/A"]; self.resolution_combobox.current(0)
             self.toggle_button['state'] = 'disabled'; self._update_current_dimensions(); self._draw_placeholder()
             return

        self.supported_resolutions = get_supported_resolutions(current_cam_device_idx)
        resolution_display_strings = [res_tuple[3] for res_tuple in self.supported_resolutions] if self.supported_resolutions else [DEFAULT_RESOLUTION_STRING]
        self.resolution_combobox['values'] = resolution_display_strings
        self.resolution_combobox['state'] = 'readonly'
        
        loaded_res_str_from_config = self.selected_resolution_str.get()
        selected_idx_for_combobox = next((i for i, res_str_val in enumerate(resolution_display_strings) if res_str_val == loaded_res_str_from_config), 0)
        
        if resolution_display_strings:
            self.selected_resolution_str.set(resolution_display_strings[selected_idx_for_combobox])

        self.resolution_combobox.current(selected_idx_for_combobox)
        self.toggle_button.config(state="normal")
        self._update_current_dimensions()
        if not self.is_tracking: self._draw_placeholder()

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
            else: 
                self.update_gui_frame_scheduled = False
                if self.is_tracking: self.toggle_tracking()
                return
            
            if self.root.winfo_exists():
                self.root.after(15, self.update_gui_frame)
            else: self.update_gui_frame_scheduled = False
        except queue.Empty:
            if self.root.winfo_exists():
                self.root.after(30, self.update_gui_frame)
            else: self.update_gui_frame_scheduled = False
        except tk.TclError as e_tk_update: 
            print(f"[PY GUI ERROR] Tkinter TclError: {e_tk_update}")
            self.update_gui_frame_scheduled = False; 
            if self.is_tracking: self.toggle_tracking()
        except RuntimeError as e_rt_update: 
            print(f"[PY GUI ERROR] RuntimeError: {e_rt_update}"); traceback.print_exc()
            self.update_gui_frame_scheduled = False
        except Exception as e_gui_update:
            print(f"[PY GUI ERROR] Unexpected error: {e_gui_update}"); traceback.print_exc()

    def on_closing(self):
        self.update_gui_frame_scheduled = False
        if self.is_tracking:
            stop_event.set()
            if video_thread and video_thread.is_alive(): video_thread.join(timeout=0.5)
        self.is_tracking = False
        self._save_config()
        
        global sock
        if sock: 
            try:
                temp_sock_close, sock = sock, None
                temp_sock_close.close()
            except Exception as e_sock_close:
                print(f"[PY GUI ERROR] Closing UDP socket: {e_sock_close}")
        
        try:
            if self.root and self.root.winfo_exists(): self.root.destroy()
        except tk.TclError: pass

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Hand Tracking Application v10.6 (HNF Sliders QTE)...") # Updated version
    splash_screen, main_tk_root, app_instance = None, None, None
    try:
        main_tk_root = tk.Tk()
        main_tk_root.withdraw()
        main_tk_root.title("Hand Tracking Control v10.6 (HNF Sliders)") # Updated title
        app_instance = HandTrackingApp(main_tk_root)
        
        if main_tk_root:
            main_tk_root.deiconify(); main_tk_root.lift(); main_tk_root.focus_force()
            main_tk_root.mainloop() 
            
    except Exception as e_startup:
        print(f"\n--- FATAL ERROR ---\n{traceback.format_exc()}Err: {e_startup}\n---")
        try:
             error_dialog_root = tk.Tk(); error_dialog_root.withdraw()
             messagebox.showerror("Startup Error", f"Critical error:\n\n{e_startup}\n\nSee console.")
             error_dialog_root.destroy()
        except Exception as e_msg_dialog: print(f"(Could not display error dialog: {e_msg_dialog})")
        finally: 
             if main_tk_root and main_tk_root.winfo_exists(): main_tk_root.destroy()
    except KeyboardInterrupt:
         print("\nCtrl+C. Closing.")
         if app_instance and hasattr(app_instance, 'on_closing') and callable(app_instance.on_closing):
             app_instance.on_closing()
         elif main_tk_root and main_tk_root.winfo_exists(): main_tk_root.destroy()
    finally:
        stop_event.set()
        if video_thread and video_thread.is_alive(): video_thread.join(timeout=0.5)
        if sock:
            try: sock.close(); sock = None
            except Exception as e_final_sock: print(f"Main: Error closing socket: {e_final_sock}")
        print("Application finished.")