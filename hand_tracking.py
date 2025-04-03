# hand_tracking_integrated_fixed_v3.py (Python) - UDP with Integrated Settings & Video GUI
# Fixes applied: FINALLY removed bg="" error in update_gui_frame success path.

import cv2
import mediapipe as mp
import socket
import tkinter as tk
from tkinter import ttk
import threading
import queue # For thread-safe communication
from PIL import Image, ImageTk
import sys
import time # For sleep
import traceback # For detailed error printing

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# --- Common Resolutions (Width, Height, Aspect Ratio Label) ---
COMMON_RESOLUTIONS = [
    (1920, 1080, "16:9"), (1280, 720, "16:9"), (1024, 768, "4:3"),
    (800, 600, "4:3"), (640, 480, "4:3"), (320, 240, "4:3")
]
DEFAULT_WIDTH, DEFAULT_HEIGHT = 640, 480

# --- Global Variables / State Management ---
video_thread = None
stop_event = threading.Event()
frame_queue = queue.Queue(maxsize=2) # Queue to pass frames from video thread to GUI
sock = None # Keep socket reference accessible

# --- Camera and Resolution Utilities ---

def find_available_cameras(max_cameras_to_check=5):
    """Tries to open cameras by index and returns a list of available ones."""
    available_cameras = []
    print("Searching for cameras...")
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # Use CAP_DSHOW on Windows
        if cap is not None and cap.isOpened():
            available_cameras.append((f"Camera {i}", i))
            cap.release()
        else:
            if cap is not None:
                cap.release()
            # break # Optional: Stop if non-sequential cameras aren't expected
    print(f"Found cameras: {available_cameras}")
    return available_cameras

def get_supported_resolutions(camera_index):
    """Checks which common resolutions are supported by the given camera index."""
    supported = []
    print(f"Probing resolutions for camera {camera_index}...")
    cap_test = None # Ensure cap_test is defined for finally block
    try:
        # Use CAP_DSHOW backend for potentially better compatibility on Windows
        cap_test = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap_test or not cap_test.isOpened():
            print(f"Error: Could not open camera {camera_index} for probing.")
            return supported # Return empty list

        # Try setting and getting resolutions
        for width, height, aspect in COMMON_RESOLUTIONS:
            cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Allow some time or a read for settings to apply if needed
            # time.sleep(0.05)
            # success_read, _ = cap_test.read() # Optional: Check if read works

            actual_width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Check if the camera accepted the resolution (allow small difference)
            if abs(actual_width - width) < 10 and abs(actual_height - height) < 10:
                 res_tuple = (width, height)
                 # Avoid adding duplicates if probing returns the same value multiple times
                 if res_tuple not in [(w, h) for w, h, a, f in supported]:
                     formatted = f"{width}x{height} ({aspect})"
                     supported.append((width, height, aspect, formatted))
                     # print(f"  -> Supported: {formatted}") # Debug

    except Exception as e:
        print(f"Exception during resolution probing for camera {camera_index}: {e}")
    finally:
        if cap_test is not None and cap_test.isOpened():
            cap_test.release()

    # Sort by width descending (highest resolution first)
    supported.sort(key=lambda x: x[0], reverse=True)
    print(f"Supported resolutions found for camera {camera_index}: {[s[3] for s in supported]}")
    return supported


# --- Video Processing Thread ---

def video_processing_loop(settings, frame_q, stop_flag):
    """Captures video, processes hands, sends UDP, and puts frames in queue."""
    global sock
    print("Video thread started.")

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = None
    cap = None

    try:
        print(f"Initializing MediaPipe Hands...")
        hands = mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1, # 0 for faster, 1 for more accurate
            static_image_mode=False,
            max_num_hands=2
        )

        print(f"Opening Camera {settings['camera_index']} with CAP_DSHOW backend...")
        cap = cv2.VideoCapture(settings['camera_index'], cv2.CAP_DSHOW)
        if not cap or not cap.isOpened():
            raise IOError(f"Cannot open camera {settings['camera_index']}")

        print(f"Attempting to set resolution to {settings['width']}x{settings['height']}...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])

        # Verify resolution after setting
        time.sleep(0.1) # Give camera time to apply settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Actual resolution reported by camera: {actual_width}x{actual_height}")
        if actual_width != settings['width'] or actual_height != settings['height']:
             print(f"Warning: Camera reported resolution {actual_width}x{actual_height}, "
                   f"which differs from requested {settings['width']}x{settings['height']}.")
             # You might want to update settings['width'] and settings['height'] here
             # if the GUI needs to know the *actual* resolution being used.
             # For now, we proceed with the actual dimensions for processing if needed.

        print(f"Setting up UDP socket to {UDP_IP}:{UDP_PORT}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print("UDP socket created.")

        print("Starting video capture and processing loop...")
        while not stop_flag.is_set():
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to grab frame from camera.")
                time.sleep(0.1) # Avoid busy-looping on read error
                continue

            # 1. Flip the frame horizontally for a mirror-view effect.
            frame = cv2.flip(frame, 1)

            # 2. Convert the BGR frame to RGB for Mediapipe processing.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. Process the frame with Mediapipe Hands.
            rgb_frame.flags.writeable = False # Optimization: pass by reference
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True # Make writable again if needed later (usually not)

            # 4. Prepare frame for display (Mediapipe draws on the input format)
            # We need BGR for drawing if we want to use cv2 drawing utils later,
            # but for Tkinter display, we need RGB eventually.
            # Let's draw on the original 'frame' (BGR) before converting final to RGB for queue.

            # 5. Draw hand landmarks and send UDP data if hands are detected.
            if results.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get handedness (Left/Right) - useful for multi-hand apps
                    # handedness = results.multi_handedness[hand_index].classification[0].label

                    # Extract wrist coordinates (normalized 0.0-1.0)
                    wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_x = wrist_landmark.x
                    wrist_y = wrist_landmark.y
                    wrist_z = wrist_landmark.z # Depth estimate relative to wrist

                    # Format data string (consider adding handedness or hand index if needed)
                    # Example: f"{hand_index},{handedness},{wrist_x:.4f},{wrist_y:.4f},{wrist_z:.4f}"
                    data_string = f"{wrist_x:.4f},{wrist_y:.4f},{wrist_z:.4f}"

                    # Send data over UDP
                    try:
                        if sock:
                            sock.sendto(data_string.encode('utf-8'), (UDP_IP, UDP_PORT))
                    except socket.error as e:
                        print(f"Error sending UDP data: {e}")
                        # Consider breaking, retrying, or closing socket on repeated errors
                        # break # Example: stop sending on error
                    except Exception as e:
                        print(f"Unexpected error sending UDP data: {e}")


                    # Draw landmarks on the BGR 'frame'
                    mp_drawing.draw_landmarks(
                        frame, # Draw on the BGR frame
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS) # Draw connecting lines

            # 6. Convert the BGR frame (with drawings) to RGB for Pillow/Tkinter display
            frame_rgb_for_tk = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 7. Put the processed RGB frame into the queue for the GUI
            try:
                # Clear the queue before putting the new frame to ensure freshness
                while not frame_q.empty():
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        break # Exit clearing loop if queue becomes empty
                # Put the latest frame
                frame_q.put(frame_rgb_for_tk, block=False) # Non-blocking put
            except queue.Full:
                # This shouldn't happen often if we clear the queue first,
                # but handle it just in case.
                # print("Warning: Frame queue still full after clearing attempt?")
                pass
            except Exception as e:
                print(f"Error putting frame in queue: {e}")

            # Optional small sleep to prevent CPU hogging if capture is very fast
            # time.sleep(0.001)

    except IOError as e:
        print(f"I/O Error in video thread: {e}")
        # Optionally put an error message/frame in the queue for the GUI
    except Exception as e:
        print(f"Error in video processing loop: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        print("Cleaning up video thread resources...")
        stop_flag.set() # Ensure stop flag is set if loop exits unexpectedly
        if cap is not None and cap.isOpened():
            cap.release()
            print("Camera released.")
        if hands:
            hands.close()
            print("Mediapipe Hands closed.")
        if sock:
            try:
                sock.close()
                print("UDP Socket closed.")
            except Exception as e:
                 print(f"Error closing socket: {e}")
            sock = None # Clear global reference

        # Clear the queue on exit
        while not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                break
        print("Frame queue cleared.")
        print("Video thread finished.")


# --- Tkinter GUI Application ---

class HandTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracking Control Panel")
        # Consider setting a minimum size
        # self.root.minsize(660, 520)

        # --- State Variables ---
        self.selected_camera_index = tk.IntVar(value=-1) # Use -1 to indicate no selection initially
        self.selected_resolution_str = tk.StringVar()
        self.available_cameras = []
        self.supported_resolutions = []
        self.is_tracking = False
        self.current_width = DEFAULT_WIDTH
        self.current_height = DEFAULT_HEIGHT

        # --- GUI Elements ---
        # Top frame for controls
        self.controls_frame = ttk.Frame(root, padding="10")
        self.controls_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Camera Selection
        ttk.Label(self.controls_frame, text="Camera:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tk.W)
        self.camera_combobox = ttk.Combobox(self.controls_frame, textvariable=self.selected_camera_index, state="readonly", width=15)
        self.camera_combobox.grid(row=0, column=1, padx=(0, 10), pady=5, sticky=tk.W)
        self.camera_combobox.bind("<<ComboboxSelected>>", self.on_camera_selected)

        # Resolution Selection
        ttk.Label(self.controls_frame, text="Resolution:").grid(row=0, column=2, padx=(10, 5), pady=5, sticky=tk.W)
        self.resolution_combobox = ttk.Combobox(self.controls_frame, textvariable=self.selected_resolution_str, state="disabled", width=20)
        self.resolution_combobox.grid(row=0, column=3, padx=(0, 10), pady=5, sticky=tk.W)

        # Start/Stop Button - Use weight to make it expand if window resizes
        self.controls_frame.columnconfigure(4, weight=1) # Allow button column to expand
        self.toggle_button = ttk.Button(self.controls_frame, text="Start Tracking", command=self.toggle_tracking, state="disabled")
        self.toggle_button.grid(row=0, column=4, padx=10, pady=5, sticky=tk.E)

        # Video Display Label - Make it expand
        self.video_label = tk.Label(root, bg="black", text="Camera Feed Area", fg="white") # Start with placeholder
        self.video_label.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # --- Initialization ---
        self.populate_camera_list()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close button

        # Start the GUI update loop only *after* controls are populated
        if self.available_cameras: # Only start if cameras were found
             self.update_gui_frame()
        else:
             self.video_label.config(text="No cameras found. Cannot start tracking.", fg="red")


    def populate_camera_list(self):
        self.available_cameras = find_available_cameras()
        if not self.available_cameras:
            self.camera_combobox['values'] = ["No cameras found"]
            self.camera_combobox.current(0)
            self.camera_combobox['state'] = 'disabled'
            self.toggle_button['state'] = 'disabled'
            print("Error: No cameras detected.")
            # Keep resolution disabled too
            self.resolution_combobox['state'] = 'disabled'
            self.resolution_combobox['values'] = ["N/A"]
            self.resolution_combobox.current(0)
            return

        camera_display_names = [name for name, index in self.available_cameras]
        self.camera_combobox['values'] = camera_display_names

        # Try to select camera 0 by default, otherwise the first one
        default_cam_list_index = 0
        selected_cam_device_index = -1
        for i, (name, index) in enumerate(self.available_cameras):
            if index == 0:
                default_cam_list_index = i
                selected_cam_device_index = 0
                break
        if selected_cam_device_index == -1 and self.available_cameras:
             # If camera 0 not found, select the first available camera
             default_cam_list_index = 0
             selected_cam_device_index = self.available_cameras[0][1]

        self.camera_combobox.current(default_cam_list_index)
        self.selected_camera_index.set(selected_cam_device_index) # Store the actual device index
        self.camera_combobox.config(state="readonly") # Enable combobox

        # Now update resolutions for the selected camera
        self.update_resolution_list()


    def on_camera_selected(self, event=None):
        # Get selected display name (e.g., "Camera 0")
        selected_display_name = self.camera_combobox.get()
        # Find the corresponding device index
        selected_cam_device_index = -1
        for name, index in self.available_cameras:
            if name == selected_display_name:
                selected_cam_device_index = index
                break

        if selected_cam_device_index != -1:
            self.selected_camera_index.set(selected_cam_device_index)
            print(f"Camera selection changed to: {selected_display_name} (Index: {selected_cam_device_index})")
            # Update resolution list for the new camera
            self.update_resolution_list()
        else:
            print(f"Error: Could not find index for camera '{selected_display_name}'")
            # Disable resolution and start button if index is invalid
            self.resolution_combobox['state'] = 'disabled'
            self.resolution_combobox['values'] = ["Error"]
            self.resolution_combobox.current(0)
            self.toggle_button['state'] = 'disabled'


    def update_resolution_list(self):
        cam_index = self.selected_camera_index.get()
        if cam_index == -1: # No valid camera selected
             self.resolution_combobox['state'] = 'disabled'
             self.toggle_button['state'] = 'disabled'
             return

        print(f"Updating resolutions for camera index: {cam_index}")
        # This can briefly freeze GUI if probing takes time.
        # Consider threading this if it's too slow for certain cameras.
        self.supported_resolutions = get_supported_resolutions(cam_index)

        if not self.supported_resolutions:
            # No specific resolutions found/supported, offer default
            default_res_string = f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT} (Default)"
            self.resolution_combobox['values'] = [default_res_string]
            self.resolution_combobox.current(0)
            self.resolution_combobox['state'] = 'readonly' # Allow selection of default
            self.selected_resolution_str.set(default_res_string)
            print(f"Warning: No specific supported resolutions found via probing for camera {cam_index}. Offering default.")
            self.toggle_button['state'] = 'normal' # Allow starting with default
        else:
            # Populate with found resolutions
            resolution_display_names = [res[3] for res in self.supported_resolutions]
            self.resolution_combobox['values'] = resolution_display_names
            self.resolution_combobox['state'] = 'readonly'

            # Try to select the default resolution if it's in the list, otherwise select the highest (first)
            default_res_target = f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}"
            selected_idx = 0 # Default to the first item (highest res)
            for i, res_tuple in enumerate(self.supported_resolutions):
                 # Check if format string contains WxH
                 if default_res_target in res_tuple[3] and res_tuple[0] == DEFAULT_WIDTH and res_tuple[1] == DEFAULT_HEIGHT:
                      selected_idx = i
                      break

            self.resolution_combobox.current(selected_idx)
            self.selected_resolution_str.set(self.supported_resolutions[selected_idx][3])
            self.toggle_button['state'] = 'normal' # Enable start button


    def toggle_tracking(self):
        global video_thread, stop_event

        if self.is_tracking:
            # --- Stop Tracking ---
            print("Stop button clicked. Stopping tracking...")
            stop_event.set() # Signal the thread to stop

            # Disable button immediately to prevent multiple clicks
            self.toggle_button.config(text="Stopping...", state="disabled")
            self.root.update_idletasks() # Force GUI update

            # Wait for the thread to finish (use join with timeout)
            if video_thread is not None and video_thread.is_alive():
                 print("Waiting for video thread to finish...")
                 video_thread.join(timeout=2.0) # Wait up to 2 seconds
                 if video_thread.is_alive():
                      print("Warning: Video thread did not stop within timeout.")
                      # Consider more forceful termination if necessary, but usually avoided
                 else:
                      print("Video thread stopped successfully.")
                 video_thread = None # Clear reference


            self.is_tracking = False
            # Reset GUI elements to idle state
            self.toggle_button.config(text="Start Tracking", state="normal") # Re-enable button
            self.camera_combobox.config(state="readonly")
            # Re-enable resolution only if there were options
            if self.resolution_combobox['values'] and self.resolution_combobox['values'][0] not in ["N/A", "Error"]:
                 self.resolution_combobox.config(state="readonly")
            else:
                 self.resolution_combobox.config(state="disabled")

            # Clear the video label back to placeholder
            self.video_label.config(image='', bg="black", text="Tracking Stopped", fg="white")
            self.video_label.image = None # Clear reference

            print("Tracking stopped.")

        else:
            # --- Start Tracking ---
            print("Start button clicked.")
            cam_index = self.selected_camera_index.get()
            if cam_index == -1:
                 print("Error: No valid camera selected.")
                 return # Don't start

            selected_res_str = self.selected_resolution_str.get()
            width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT # Fallback defaults

            # Parse width/height from selected string
            parsed_successfully = False
            for w, h, a, f in self.supported_resolutions:
                if f == selected_res_str:
                    width, height = w, h
                    parsed_successfully = True
                    break
            # Handle the case where the "(Default)" option was selected
            if not parsed_successfully and f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT} (Default)" == selected_res_str:
                 width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
                 parsed_successfully = True

            if not parsed_successfully:
                 print(f"Error: Could not parse selected resolution '{selected_res_str}'. Using default {width}x{height}.")
                 # You might want to display an error to the user here

            self.current_width = width
            self.current_height = height
            tracking_settings = {
                "camera_index": cam_index,
                "width": width,
                "height": height,
            }

            print(f"Attempting to start tracking with: Camera={cam_index}, Resolution={width}x{height}")

            # Disable controls and update button text
            self.toggle_button.config(text="Stop Tracking", state="normal") # Keep enabled to allow stopping
            self.camera_combobox.config(state="disabled")
            self.resolution_combobox.config(state="disabled")
            self.video_label.config(bg="black", text="Starting Camera...", fg="yellow") # Update status
            self.root.update_idletasks()

            # Clear the stop event and start the video thread
            stop_event.clear()
            video_thread = threading.Thread(
                target=video_processing_loop,
                args=(tracking_settings, frame_queue, stop_event),
                daemon=True # Allows main program to exit even if thread is running
            )
            video_thread.start()
            self.is_tracking = True
            print("Tracking started.")
            # GUI update loop (update_gui_frame) will display the video feed

    def update_gui_frame(self):
        """Periodically checks the queue for new frames and updates the label."""
        # Only process if tracking is supposed to be active
        if self.is_tracking and not stop_event.is_set():
            try:
                # Get the most recent frame, discard older ones if queue built up
                frame_rgb = frame_queue.get_nowait() # Get frame (already RGB)

                # Convert the frame to a PhotoImage
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)

                # Update the label widget
                # =====================================
                # == THE ONLY CHANGE IS HERE ==
                self.video_label.config(image=img_tk, text="") # REMOVED bg=""
                # =====================================
                self.video_label.image = img_tk # Keep reference!

            except queue.Empty:
                # No new frame available yet, do nothing visually, keep checking
                pass
            except Exception as e:
                 # Log error, maybe stop tracking or show error on label
                 print(f"Error updating GUI frame: {e}")
                 # traceback.print_exc() # Uncomment for detailed GUI update errors
                 # Example: Display error on label
                 # self.video_label.config(image='', bg="black", text=f"GUI Error: {e}", fg="red")
                 # self.video_label.image = None # Clear ref if error occurs

        # Schedule the next update regardless of whether a frame was processed,
        # unless the application is closing.
        # Use try-except in case root window is destroyed during schedule call
        try:
             # Schedule next check only if the root window still exists
             if self.root.winfo_exists():
                self.root.after(20, self.update_gui_frame) # ~50 FPS GUI updates attempt
        except tk.TclError:
             # This can happen if the window is destroyed right before .after() is called
             # Or if winfo_exists() fails for some reason during shutdown
             print("GUI update loop stopped (window likely closed).")


    def on_closing(self):
        """Called when the window close button ('X') is pressed."""
        print("Window close requested.")
        if self.is_tracking:
            print("Stopping tracking before closing...")
            stop_event.set()
            if video_thread is not None and video_thread.is_alive():
                video_thread.join(timeout=1.0) # Wait briefly for thread cleanup
        print("Destroying Tkinter window.")
        # Ensure GUI update loop stops trying to schedule itself
        # (Setting stop_event should prevent new frames, but belt-and-suspenders)
        self.is_tracking = False # Prevent update_gui_frame from processing queue
        self.root.destroy()


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Hand Tracking Application...")
    root = tk.Tk()
    app = HandTrackingApp(root)
    try:
        # This starts the Tkinter event loop. It will block until the window is closed.
        root.mainloop()
    except KeyboardInterrupt:
        # Handle Ctrl+C in the terminal
        print("\nCtrl+C detected. Closing application.")
        app.on_closing() # Ensure cleanup happens

    print("Application finished.")
    # A clean exit might be needed if daemon threads hang
    sys.exit(0)