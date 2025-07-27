import time
import threading
import queue
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import csv
import os
import traceback
from datetime import datetime, timedelta
from filterpy.kalman import KalmanFilter

def setup_kalman_filter(config):
    """
    Setup Kalman filter for smoothing 3D position data (X, Y, Z). (CPU)
    State vector: [X, Y, Z, vX, vY, vZ] (position and velocity)
    Args:
        config (dict): Configuration dictionary with Kalman filter settings
    Returns:
        KalmanFilter: Configured Kalman filter instance
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)
    
    # Get Kalman filter configuration
    kalman_config = config['kalman']
    fps = kalman_config['fps'] if kalman_config['fps'] > 0 else 30.0
    dt = 1.0 / fps
    
    # State transition matrix (constant velocity model)
    kf.F = np.array([[1, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    
    # Measurement matrix (we observe position only)
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])
    
    # Measurement noise covariance
    kf.R = np.eye(3) * kalman_config['measurement_noise']
    
    # Process noise covariance
    kf.Q = np.eye(6) * kalman_config['process_noise']
    
    # Initial covariance
    kf.P = np.eye(6) * kalman_config['initial_covariance']
    
    return kf

def load_stereo_calibration_npz(npz_path):
    """
    Load stereo calibration parameters from a single .npz file (CPU).
    Args:
        npz_path (str): Path to the stereo_calibration.npz file.
    Returns:
        dict: Dictionary with all required calibration parameters.
    """
    data = np.load(npz_path)
    params = {
        "image_size": data["imageSize"],
        "left_map_x_rectify": data["left_map_x_rectify"],
        "left_map_y_rectify": data["left_map_y_rectify"],
        "right_map_x_rectify": data["right_map_x_rectify"],
        "right_map_y_rectify": data["right_map_y_rectify"],
        "Q": data["disparityToDepthMap"],
        "R": data["rotationMatrix"],
        "T": data["translationVector"],
    }
    return params

def world_to_bev(X, Z, bev_origin, bev_scale):
    """
    Convert world (X, Z) coordinates in meters to BEV pixel coordinates. (CPU)
    Args:
        X (float): X position in meters.
        Z (float): Z position in meters.
        bev_origin (tuple): Pixel coordinates for (0,0) in BEV.
        bev_scale (int): Pixels per meter.
    Returns:
        tuple: (px, pz) pixel coordinates in BEV image.
    """
    px = int(bev_origin[0] + X * bev_scale)
    pz = int(bev_origin[1] - Z * bev_scale)
    return px, pz

def draw_bev(objects, track_history, config, window_name="BEV Map"):
    """
    Draw the BEV map with drone position(s), trackline, and labels. (CPU)
    Args:
        objects (list): List of (X, Z) tuples for current frame (already filtered).
        track_history (list): History of tracked positions.
        config (dict): Configuration dictionary with BEV settings.
        window_name (str): Name for the BEV display window.
    Returns:
        None. Displays the BEV image in a window.
    CPU: All BEV drawing and display
    """
    # Get BEV configuration
    bev_config = config['bev']
    bev_size = bev_config['size']
    bev_scale = bev_config['scale']
    bev_origin = (bev_size[0] // 2, bev_size[1] - bev_scale)  # bottom center as (0,0)
    draw_trackline = bev_config['draw_trackline']
    
    grid_color = tuple(bev_config['grid_color'])
    axis_color = tuple(bev_config['axis_color'])
    track_line_color = tuple(bev_config['track_line_color'])
    track_line_thickness = bev_config['track_line_thickness']
    start_point_color = tuple(bev_config['start_point_color'])
    start_point_radius = bev_config['start_point_radius']
    current_point_color = tuple(bev_config['current_point_color'])
    current_point_radius = bev_config['current_point_radius']
    font_scale = bev_config['font_scale']
    font_thickness = bev_config['font_thickness']
    label_color = tuple(bev_config['label_color'])
    
    bev_img = np.zeros((bev_size[1], bev_size[0], 3), dtype=np.uint8)
    # Draw vertical grid lines and X numbers
    for x in range(0, bev_size[0], bev_scale):
        cv2.line(bev_img, (x, 0), (x, bev_size[1]), grid_color, 1)
        # X axis numbers (meters)
        x_meter = (x - bev_origin[0]) / bev_scale
        if 0 <= x < bev_size[0]:
            if abs(x_meter) < 1e-2:
                label = '0'
            else:
                label = f"{x_meter:.0f}"
            # Place numbers below X axis
            cv2.putText(bev_img, label, (x + 2, bev_origin[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, font_scale, axis_color, 1, cv2.LINE_AA)
    # Draw horizontal grid lines and Z numbers
    for y in range(0, bev_size[1], bev_scale):
        cv2.line(bev_img, (0, y), (bev_size[0], y), grid_color, 1)
        # Z axis numbers (meters)
        z_meter = (bev_origin[1] - y) / bev_scale
        if 0 <= y < bev_size[1]:
            if abs(z_meter) < 1e-2:
                label = '0'
            else:
                label = f"{z_meter:.0f}"
            # Place numbers to the left of Z axis
            cv2.putText(bev_img, label, (bev_origin[0] + 8, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, axis_color, 1, cv2.LINE_AA)
    # Draw X axis (horizontal, through origin)
    cv2.line(bev_img, (0, bev_origin[1]), (bev_size[0], bev_origin[1]), axis_color, 2)
    # Draw Z axis (vertical, through origin)
    cv2.line(bev_img, (bev_origin[0], 0), (bev_origin[0], bev_size[1]), axis_color, 2)
    # Draw axis labels
    cv2.putText(bev_img, 'X', (bev_size[0] - 30, bev_origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, axis_color, 2, cv2.LINE_AA)
    cv2.putText(bev_img, 'Z', (bev_origin[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, axis_color, 2, cv2.LINE_AA)

    # Update track history (assume one drone, one object per frame)
    if len(objects) > 0:
        # Only track the first object (drone)
        X, Z = objects[0][0], objects[0][1]
        track_history.append((X, Z))
    
    # Draw trackline if enabled
    # NOTE: Track history is intentionally unlimited to visualize the full drone path for the entire video.
    if draw_trackline and len(track_history) > 1:
        pts = [world_to_bev(x, z, bev_origin, bev_scale) for x, z in track_history]
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(bev_img, [pts], isClosed=False, color=track_line_color, thickness=track_line_thickness)
        # Draw start point
        start_pt = pts[0]
        cv2.circle(bev_img, tuple(start_pt), start_point_radius, start_point_color, -1)
    
    # Draw current objects (filtered positions)
    for obj in objects:
        X, Z = obj[0], obj[1]
        px, pz = world_to_bev(X, Z, bev_origin, bev_scale)
        if 0 <= px < bev_size[0] and 0 <= pz < bev_size[1]:
            cv2.circle(bev_img, (px, pz), current_point_radius, current_point_color, -1)
            label = f"drone ({X:.2f}, {Z:.2f})"
            cv2.putText(bev_img, label, (px + 10, pz - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness, cv2.LINE_AA)
    
    cv2.imshow(window_name, bev_img)

def save_config_to_tracking_folder(config, tracking_folder):
    """
    Save a copy of the config.yaml to the tracking folder.
    Args:
        config (dict): Configuration dictionary.
        tracking_folder (str): Path to the tracking folder.
    """
    if tracking_folder is not None:
        config_path = os.path.join(tracking_folder, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Config saved to: {config_path}")

def setup_csv_logging(config):
    """
    Setup CSV logging for tracking data if enabled in config.
    Creates a timestamped folder and returns the folder path for later config saving.
    Returns: (csv_writer, csv_file, csv_path, tracking_folder) or (None, None, None, None) if disabled
    """
    if not config['logging']['save_tracking_data']:
        return None, None, None, None
    
    # Create output directory if it doesn't exist
    output_dir = config['logging']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracking_folder = os.path.join(output_dir, f"tracking_{timestamp}")
    os.makedirs(tracking_folder, exist_ok=True)
    
    # Create CSV file with fixed name 'track.csv'
    csv_path = os.path.join(tracking_folder, "track.csv")
    
    # Open CSV file and create writer
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # Write header
    # Match ground truth CSV: time (ms), datetime, track_id, X, Y, Z, disparity
    csv_writer.writerow(['time(millisecond)', 'datetime', 'track_id', 'X', 'Y', 'Z', 'disparity'])
    
    return csv_writer, csv_file, csv_path, tracking_folder

def setup_rectification_maps_from_npz(calib_params):
    """
    Setup rectification maps from NPZ calibration data and pre-allocated GpuMat objects (CPU+GPU).
    Args:
        calib_params (dict): Calibration parameters from NPZ file.
    Returns:
        (mapx1_gpu, mapy1_gpu, mapx2_gpu, mapy2_gpu, frame_gpu_l, frame_gpu_r, rect_gpu_l, rect_gpu_r)
    """
    # Get rectification maps from NPZ
    
    left_combined = calib_params["left_map_x_rectify"]  # Shape: (H, W, 2)
    right_combined = calib_params["right_map_x_rectify"]  # Shape: (H, W, 2)
    
    # Extract separate X and Y maps from the combined format
    mapx1 = left_combined[:, :, 0].astype(np.float32)  # X coordinates
    mapy1 = left_combined[:, :, 1].astype(np.float32)  # Y coordinates  
    mapx2 = right_combined[:, :, 0].astype(np.float32)  # X coordinates
    mapy2 = right_combined[:, :, 1].astype(np.float32)  # Y coordinates
    
    # Upload to GPU
    mapx1_gpu = cv2.cuda_GpuMat()
    mapy1_gpu = cv2.cuda_GpuMat()
    mapx2_gpu = cv2.cuda_GpuMat()
    mapy2_gpu = cv2.cuda_GpuMat()
    mapx1_gpu.upload(mapx1)
    mapy1_gpu.upload(mapy1)
    mapx2_gpu.upload(mapx2)
    mapy2_gpu.upload(mapy2)
    
    # Pre-allocate GpuMat objects
    frame_gpu_l = cv2.cuda_GpuMat()
    frame_gpu_r = cv2.cuda_GpuMat()
    rect_gpu_l = cv2.cuda_GpuMat()
    rect_gpu_r = cv2.cuda_GpuMat()
    
    return mapx1_gpu, mapy1_gpu, mapx2_gpu, mapy2_gpu, frame_gpu_l, frame_gpu_r, rect_gpu_l, rect_gpu_r

def setup_yolo_model(model_path):
    """
    Load YOLO model (CPU for loading, GPU for inference).
    Returns: model instance
    """
    print("Loading YOLO model...")
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def setup_display_windows():
    """
    Create and configure display windows for left and right rectified & tracked frames.
    """
    cv2.namedWindow("Left Rectified & Tracked", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Rectified & Tracked", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left Rectified & Tracked", *DISPLAY_SIZE)
    cv2.resizeWindow("Right Rectified & Tracked", *DISPLAY_SIZE)
    cv2.moveWindow("Left Rectified & Tracked", *LEFT_WINDOW_POS)
    cv2.moveWindow("Right Rectified & Tracked", *RIGHT_WINDOW_POS)

def rectify(frame, mapx_gpu, mapy_gpu, rect_gpu):
    """
    Rectify a frame using pre-allocated GPU Mats and download to CPU. (GPU->CPU)
    Args:
        frame (cv2.cuda_GpuMat): Input frame (already on GPU).
        mapx_gpu, mapy_gpu: Rectification maps (GPU).
        rect_gpu (cv2.cuda_GpuMat): Pre-allocated GPU Mat for remap.
    Returns:
        np.ndarray: Rectified frame (CPU).
    """
    rect_gpu = cv2.cuda.remap(frame, mapx_gpu, mapy_gpu, interpolation=cv2.INTER_LINEAR, dst=rect_gpu)
    rect = rect_gpu.download() # Currently YOLO does not support GpuMat directly, so we need to download it
    return rect, rect_gpu  # Return rect_gpu to keep reusing the same object

def compute_stereo_3d(left_centers, right_centers, Q, max_y_diff=10.0):
    """
    Compute disparity and 3D (X, Y, Z) for matched objects in a single frame. (CPU)
    Uses Y difference constraint for better stereo matching.
    Args:
        left_centers (list): List of (cx, cy) for left frame.
        right_centers (list): List of (cx, cy) for right frame.
        Q (np.ndarray): 4x4 reprojection matrix from stereo calibration.
        max_y_diff (float): Maximum allowed Y difference between left and right detections.
    Returns:
        list: results, a list of dicts with all values for printing and BEV drawing.
    CPU: Disparity and 3D calculation
    """
    results = []
    
    if len(left_centers) == 0 or len(right_centers) == 0:
        return results
    
    # Find best matches based on Y difference constraint
    used_right = set()
    
    for i, (xl, yl) in enumerate(left_centers):
        best_match = None
        min_y_diff = float('inf')
        
        # Find the right detection with minimum Y difference
        for j, (xr, yr) in enumerate(right_centers):
            if j in used_right:
                continue
                
            y_diff = abs(yl - yr)
            if y_diff < max_y_diff and y_diff < min_y_diff:
                min_y_diff = y_diff
                best_match = j
        
        if best_match is not None:
            used_right.add(best_match)
            xr, yr = right_centers[best_match]
            disparity = xl - xr
            
            if disparity == 0:
                results.append({
                    'i': i,
                    'disparity': 0,
                    'xl': xl, 'yl': yl, 'xr': xr, 'yr': yr,
                    'X': None, 'Y': None, 'Z': None
                })
            else:
                X = (xl * Q[0, 0] + yl * Q[0, 1] + disparity * Q[0, 2] + Q[0, 3])
                Y = (xl * Q[1, 0] + yl * Q[1, 1] + disparity * Q[1, 2] + Q[1, 3])
                Z = (xl * Q[2, 0] + yl * Q[2, 1] + disparity * Q[2, 2] + Q[2, 3])
                W = (xl * Q[3, 0] + yl * Q[3, 1] + disparity * Q[3, 2] + Q[3, 3])
                if W != 0:
                    X /= W
                    Y /= W
                    Z /= W
                results.append({
                    'i': i,
                    'disparity': disparity,
                    'xl': xl, 'yl': yl, 'xr': xr, 'yr': yr,
                    'X': X, 'Y': Y, 'Z': Z
                })
    
    return results

def extract_bbox_centers(result):
    """
    Extracts bounding box centers from a YOLO result object. (CPU)
    Args:
        result: YOLO result object for a single frame.
    Returns:
        list: List of (cx, cy) tuples for each detected object.
    CPU: Bounding box extraction and center calculation
    """
    centers = []
    if hasattr(result, 'boxes') and result.boxes is not None:
        bboxes = result.boxes.xyxy
        if bboxes is not None and len(bboxes) > 0:
            for box in bboxes:
                x1, y1, x2, y2 = box.tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centers.append((cx, cy))
    return centers

def stereo_frame_producer(left_video, right_video, frame_queue, close_event, startup_event, config,
                         mapx1_gpu=None, mapy1_gpu=None, mapx2_gpu=None, mapy2_gpu=None,
                         rect_gpu_l=None, rect_gpu_r=None):
    """
    Producer thread that reads synchronized frame pairs from left and right videos. 
    Puts frame pairs into a queue for consumer threads to process.
    Args:
        left_video (str): Path to left video file.
        right_video (str): Path to right video file.
        frame_queue (queue.Queue): Thread-safe queue to store frame pairs.
        close_event (threading.Event): Event to signal threads to close.
        startup_event (threading.Event): Event to signal consumer is ready.
        config (dict): Configuration dictionary.
        mapx1_gpu, mapy1_gpu, mapx2_gpu, mapy2_gpu: Rectification maps (GPU).
        rect_gpu_l, rect_gpu_r: Pre-allocated GPU matrices.
    Returns:
        None. Puts (frame_l, frame_r, frame_idx) tuples into frame_queue.
    GPU: Video capture and frame reading
    """
    # Use CUDA VideoReader for GPU decoding (VIDEO FILES ONLY)
    left_reader = cv2.cudacodec.createVideoReader(left_video)
    right_reader = cv2.cudacodec.createVideoReader(right_video)
    
    # try:
    cap_tmp = cv2.VideoCapture(left_video)
    video_fps = cap_tmp.get(cv2.CAP_PROP_FPS)
    cap_tmp.release()

    frame_time = 1.0 / video_fps

    print(f"Starting frame production (GPU decode), pacing to {video_fps:.2f} FPS...")

    frame_idx = 0
    last_frame_time = time.time()
    while not close_event.is_set():
        # Pacing Logic to prevent overwhelming the consumer
        time_since_last_frame = time.time() - last_frame_time
        if time_since_last_frame < frame_time:
            time.sleep(frame_time - time_since_last_frame)
        last_frame_time = time.time()

        # Read frames from both readers
        ret_l, gpu_frame_l = left_reader.nextFrame()
        ret_r, gpu_frame_r = right_reader.nextFrame()

        if not ret_l or not ret_r:
            break

        # Convert BGRA to BGR (GPU) 
        frame_l = cv2.cuda.cvtColor(gpu_frame_l, cv2.COLOR_BGRA2BGR)
        frame_r = cv2.cuda.cvtColor(gpu_frame_r, cv2.COLOR_BGRA2BGR)
        
        # Apply rectification
        rect_l, _ = rectify(frame_l, mapx1_gpu, mapy1_gpu, rect_gpu_l)
        rect_r, _ = rectify(frame_r, mapx2_gpu, mapy2_gpu, rect_gpu_r)
        # Use rectified frames (already on CPU from rectify function)
        final_frame_l = rect_l
        final_frame_r = rect_r

        try:
            frame_queue.put((final_frame_l, final_frame_r, frame_idx), timeout=0.1)
            frame_idx += 1
        except queue.Full:
            if close_event.is_set():
                break
            if startup_event.is_set():
                print("Frame queue is full, dropping frame")
            continue
        except Exception as e:
            print("Error in producer:")
            traceback.print_exc()
            break

    # Send sentinel to signal end of frames
    try:
        frame_queue.put(None, timeout=0.5)
    except queue.Full:
        pass

def stereo_frame_consumer(model, Q, frame_queue, close_event, startup_event, config, csv_writer=None):
    """
    Consumer thread that processes synchronized frame pairs from the queue.
    Handles rectification, detection, tracking, 3D computation, and BEV drawing. (CPU+GPU)
    Args:
        model: YOLO model instance for object detection and tracking (GPU).
        Q (np.ndarray): 4x4 reprojection matrix from stereo calibration.
        frame_queue (queue.Queue): Thread-safe queue containing frame pairs.
        close_event (threading.Event): Event to signal threads to close.
        startup_event (threading.Event): Event to signal when consumer is ready.
    Returns:
        None. Processes frames and displays results.
    CPU: 3D computation, BEV drawing, display
    GPU: Rectification, YOLO inference
    """

    # Create display windows
    setup_display_windows()
    
    # Initialize batch processing variables
    first_frame_processed = False
    batch_left = []
    batch_right = []
    batch_indices = []

    # Track history for BEV
    track_history = []
    
    # Initialize Kalman filter for position smoothing (CPU)
    kalman_filter = setup_kalman_filter(config)
    kalman_initialized = False

    # For logging every 3 frames and time tracking
    log_interval_frames = 3
    frame_time_ms = 1000.0 / 30.0  # 30 FPS assumed
    start_datetime = datetime.now()
    try:
        while not close_event.is_set():
            try:
                # Fill batch
                got_sentinel = False
                while len(batch_left) < BATCH_SIZE and not close_event.is_set():
                    item = frame_queue.get(timeout=1.0)
                    if item is None:
                        # If sentinel, process what we have and exit
                        got_sentinel = True
                        break
                    frame_l, frame_r, frame_idx = item
                    # Frames are already processed (rectified if enabled) by producer
                    batch_left.append(frame_l)
                    batch_right.append(frame_r)
                    batch_indices.append(frame_idx)
                if not batch_left and got_sentinel:
                    # No more frames and got sentinel, exit immediately
                    break
                if not batch_left:
                    continue
                
                # Batch FPS timing
                batch_start_time = time.time()
                # YOLO batch inference
                results_l = model.track(batch_left, persist=True, tracker=TRACKER_CONFIG, verbose=False, imgsz=IMAGE_SIZE)
                results_r = model.track(batch_right, persist=True, tracker=TRACKER_CONFIG, verbose=False, imgsz=IMAGE_SIZE)
                batch_end_time = time.time()
                batch_fps = len(batch_left) / (batch_end_time - batch_start_time)
                print(f"FPS: {batch_fps:.2f}")
                # Signal that consumer is ready after first batch processed (includes TensorRT loading)
                if not first_frame_processed:
                    print("TensorRT engine loaded, system ready.")
                    startup_event.set()
                    first_frame_processed = True
                # Process each pair in the batch
                for i in range(len(batch_left)):
                    frame_idx = batch_indices[i]
                    left_centers = extract_bbox_centers(results_l[i])
                    right_centers = extract_bbox_centers(results_r[i])
                    n = min(len(left_centers), len(right_centers))
                    if n > 0:
                        results_3d = compute_stereo_3d(left_centers, right_centers, Q, config['stereo']['max_y_diff'])
                        bev_objects = []
                        for res in results_3d:
                            j = res['i']
                            if res['disparity'] == 0:
                                print(f"  Object {j+1}: disparity=0, cannot compute Z.")
                                continue

                            # Apply Kalman filtering to the 3D coordinates (CPU) if enabled
                            raw_X, raw_Y, raw_Z = res['X'], res['Y'], res['Z']

                            if config['kalman']['enabled']:
                                if kalman_initialized:
                                    # Predict and update Kalman filter
                                    kalman_filter.predict()
                                    measurement = np.array([raw_X, raw_Y, raw_Z])
                                    kalman_filter.update(measurement)

                                    # Get filtered coordinates
                                    filtered_X = kalman_filter.x[0]
                                    filtered_Y = kalman_filter.x[1]
                                    filtered_Z = kalman_filter.x[2]
                                else:
                                    # Initialize Kalman filter with first valid measurement
                                    kalman_filter.x = np.array([raw_X, raw_Y, raw_Z, 0, 0, 0])  # [X, Y, Z, vX, vY, vZ]
                                    kalman_initialized = True
                                    filtered_X, filtered_Y, filtered_Z = raw_X, raw_Y, raw_Z
                            else:
                                # Use raw coordinates if Kalman filtering is disabled
                                filtered_X, filtered_Y, filtered_Z = raw_X, raw_Y, raw_Z

                            # Log filtered coordinates to CSV if enabled, every 3 frames only
                            if csv_writer is not None and (frame_idx % log_interval_frames == 0):
                                # Calculate time in ms since start
                                time_ms = int(round(frame_idx * frame_time_ms))
                                # Calculate datetime (start + time_ms)
                                dt = start_datetime + timedelta(milliseconds=time_ms)
                                dt_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                csv_writer.writerow([time_ms, dt_str, j+1, filtered_X, filtered_Y, filtered_Z, res['disparity']])

                            bev_objects.append((filtered_X, filtered_Z))
                        draw_bev(bev_objects, track_history, config)
                    else:
                        draw_bev([], track_history, config)
                    # Display annotated frames (CPU)
                    annotated_l = results_l[i].plot()
                    annotated_r = results_r[i].plot()
                    cv2.imshow("Left Rectified & Tracked", annotated_l)
                    cv2.imshow("Right Rectified & Tracked", annotated_r)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        close_event.set()
                        break
                # Clear batch
                batch_left.clear()
                batch_right.clear()
                batch_indices.clear()
                if got_sentinel:
                    # After processing last batch, exit
                    break
            except queue.Empty:
                continue
            except Exception as e:
                # Catch error instead of fallback
                print("Error in consumer:")
                traceback.print_exc()
                close_event.set()
                break
    finally:
        # Always close all OpenCV windows when done
        cv2.destroyAllWindows()

# === MAIN SCRIPT (CPU unless otherwise noted) ===
# --- Load Configuration from YAML File ---
with open("config/stereo_vision.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Paths
CALIB_PATH = config['paths']['calibration']
MODEL_PATH = config['paths']['model']
TRACKER_CONFIG = config['paths']['tracker']
LEFT_VIDEO = config['paths']['video_left']
RIGHT_VIDEO = config['paths']['video_right']

# BEV PARAMETERS
MAXSIZE = config['queue']['maxsize']

# MODEL PARAMETERS
BATCH_SIZE = config['model']['batch_size']
IMAGE_SIZE = config['model']['image_size']

# Display settings
LEFT_WINDOW_POS = config['display']['left_window_pos']
RIGHT_WINDOW_POS = config['display']['right_window_pos']
DISPLAY_SIZE = config['display']['size']
KALMAN_CONFIG = config['kalman']

# Get frame size from left video (CPU)
cap_left = cv2.VideoCapture(LEFT_VIDEO)  
ret_l, frame_l = cap_left.read() 
if not ret_l:
    raise RuntimeError("Could not read frame from left video.")
h, w = frame_l.shape[:2]
cap_left.release()

# Load calibration params from NPZ (CPU)
calib_params = load_stereo_calibration_npz(CALIB_PATH)

# Extract Q matrix for 3D calculations
Q = calib_params["Q"]  

# Setup rectification maps for both cameras (CPU+GPU)
print("Setting up rectification maps from NPZ...")
(mapx1_gpu, mapy1_gpu, mapx2_gpu, mapy2_gpu,
frame_gpu_l, frame_gpu_r, rect_gpu_l, rect_gpu_r) = setup_rectification_maps_from_npz(calib_params)

# Shared event to signal start and close
startup_event = threading.Event()
close_event = threading.Event()

# Create frame queue for producer-consumer pattern
frame_queue = queue.Queue(maxsize=MAXSIZE)

# Load YOLO model (CPU for loading, GPU for inference)
model = setup_yolo_model(MODEL_PATH)

# Setup CSV logging if enabled
csv_writer, csv_file, csv_path, tracking_folder = setup_csv_logging(config)

# Start producer thread (reads synchronized frame pairs)
producer_thread = threading.Thread(
    target=stereo_frame_producer, 
    args=(LEFT_VIDEO, RIGHT_VIDEO, frame_queue, close_event, startup_event, config,
          mapx1_gpu, mapy1_gpu, mapx2_gpu, mapy2_gpu, rect_gpu_l, rect_gpu_r)
)

# Start consumer thread (processes frame pairs)
consumer_thread = threading.Thread(
    target=stereo_frame_consumer,
    args=(model, Q, frame_queue, close_event, startup_event, config, csv_writer)
)

# Start both threads
producer_thread.start()
consumer_thread.start()

try:
    producer_thread.join()
    consumer_thread.join()
except KeyboardInterrupt:
    close_event.set()
finally:
    print("Shutting down...")
    close_event.set()
    # Clear the queue to unblock producer
    try:
        while not frame_queue.empty():
            frame_queue.get_nowait()
    except queue.Empty:
        pass
    # Give threads time to finish gracefully
    producer_thread.join(timeout=2)
    consumer_thread.join(timeout=2)
    # Close CSV file and save config if logging was enabled
    if csv_file is not None:
        csv_file.close()
        print(f"Tracking data saved in: {csv_path}")
        # Save config to tracking folder
        save_config_to_tracking_folder(config, tracking_folder)
