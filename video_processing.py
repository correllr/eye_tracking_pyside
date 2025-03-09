#!/usr/bin/env python3
import cv2
import os
from datetime import datetime
import pandas as pd
import numpy as np

MIN_PUPIL_RADIUS = 20
MAX_PUPIL_RADIUS = 150  # Fallback maximum allowed pupil radius if no initial measurement is provided.
MAX_RADIUS_DELTA = 2    # Maximum change in pupil radius allowed per frame.

# Exponential smoothing factor for the detected position.
SMOOTHING_FACTOR = 0.85

# For running averages, we now maintain buffers over a fixed number of frames.
MAX_RECENT_FRAMES = 30

def apply_rotation(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame

def track_pupil_with_video_output(video_capture, roi, threshold_value, rotation_angle,
                                  fps, range_slider_value, current_session_dir, contrast=1.0, exposure=0,
                                  progress_callback=None, initial_cumulative_center=None, initial_pupil_radius=None):
    """
    Process video frames between the given range and track the pupil.
    Returns (output_video_path, pupil_positions), where pupil_positions is a list of (timestamp, x, y).

    Processing (detection, ROI normalization, thresholding, etc.) is performed on a contrast/exposure-adjusted frame.
    The output video is produced using the original frame (after applying rotation) so that the output retains
    the original contrast and exposure. If a pupil is detected, the detection overlay is drawn on the original frame.
    """
    try:
        x, y, w, h = roi
        pupil_positions = []
        tracking_video_filename = f"pupil_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        output_path = os.path.join(current_session_dir, tracking_video_filename)

        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if rotation_angle in (90, 270):
            frame_width, frame_height = frame_height, frame_width

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        start_frame, end_frame = range_slider_value
        total_frames_to_process = end_frame - start_frame if end_frame > start_frame else 1
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Baseline mean intensity for the ROI (from the first frame processed)
        baseline_mean = None

        # Initialize exponential smoothing for position.
        exp_smoothed_center = initial_cumulative_center  # None if not provided.

        # For running average of radius.
        recent_radii = []
        if initial_pupil_radius is not None:
            recent_radii.append(initial_pupil_radius)

        # For running average of pupil center.
        recent_positions = []
        if initial_cumulative_center is not None:
            recent_positions.append(initial_cumulative_center)

        # For running average of "blackness".
        recent_blackness = []
        initial_blackness = None

        last_valid_pupil_center = None

        while True:
            current_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            if current_frame > end_frame:
                break
            ret, frame = video_capture.read()
            if not ret:
                break

            # Save the original frame (before contrast/exposure adjustment)
            original_frame = frame.copy()

            # Processing: adjust contrast/exposure, then rotate.
            processed_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=exposure)
            processed_frame = apply_rotation(processed_frame, rotation_angle)

            # Extract ROI from the processed frame and convert to grayscale.
            roi_frame = processed_frame[y:y+h, x:x+w]
            if len(roi_frame.shape) == 3:
                roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            
            # Normalize ROI to baseline.
            if baseline_mean is None:
                baseline_mean = np.mean(roi_frame)
            else:
                current_mean = np.mean(roi_frame)
                if current_mean > 0:
                    scale_factor = baseline_mean / current_mean
                    roi_frame = cv2.convertScaleAbs(roi_frame, alpha=scale_factor, beta=0)

            # Threshold the normalized ROI.
            _, thresholded = cv2.threshold(roi_frame, threshold_value, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            pupil_center = None
            used_radius = 0

            if contours:
                candidate_circles = []
                for cnt in contours:
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    if radius >= MIN_PUPIL_RADIUS:
                        candidate_circles.append(((cx + x, cy + y), radius))
                if candidate_circles:
                    if len(candidate_circles) > 1:
                        gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                        candidate_data = []
                        for candidate in candidate_circles:
                            center_candidate, radius_candidate = candidate
                            if initial_pupil_radius is not None:
                                if not (0.75 * initial_pupil_radius <= radius_candidate <= 1.25 * initial_pupil_radius):
                                    continue
                            mask_candidate = np.zeros(gray_frame.shape, dtype=np.uint8)
                            cv2.circle(mask_candidate, (int(center_candidate[0]), int(center_candidate[1])), int(radius_candidate), 255, -1)
                            candidate_intensity = np.mean(gray_frame[mask_candidate == 255])
                            if initial_blackness is None:
                                initial_blackness = candidate_intensity
                            else:
                                if not (0.75 * initial_blackness <= candidate_intensity <= 1.25 * initial_blackness):
                                    continue
                            candidate_data.append((candidate, candidate_intensity))
                        if candidate_data:
                            if len(candidate_data) > 1:
                                costs = []
                                for cand, inten in candidate_data:
                                    center_candidate, radius_candidate = cand
                                    if initial_cumulative_center is not None:
                                        pos_cost = np.linalg.norm(np.array(center_candidate) - np.array(initial_cumulative_center))
                                    else:
                                        pos_cost = 0
                                    radius_cost = abs(radius_candidate - initial_pupil_radius) if initial_pupil_radius is not None else 0
                                    cost = 0.7 * pos_cost + 0.3 * radius_cost
                                    costs.append((cand, cost))
                                best_candidate, best_cost = min(costs, key=lambda item: item[1])
                                pupil_center = (int(best_candidate[0][0]), int(best_candidate[0][1]))
                                used_radius = best_candidate[1]
                                print(f"Debug: Best candidate selected with cost {best_cost:.2f}")
                            else:
                                candidate, inten = candidate_data[0]
                                pupil_center = (int(candidate[0][0]), int(candidate[0][1]))
                                used_radius = candidate[1]
                                print("Debug: Only one candidate accepted based on criteria.")
                        else:
                            if initial_cumulative_center is not None:
                                distances = [np.linalg.norm(np.array(c[0]) - np.array(initial_cumulative_center)) for c in candidate_circles]
                                idx = int(np.argmin(distances))
                                candidate = candidate_circles[idx]
                                pupil_center = (int(candidate[0][0]), int(candidate[0][1]))
                                used_radius = candidate[1]
                                print("Debug: Fallback to candidate nearest original center.")
                            else:
                                largest = max(candidate_circles, key=lambda item: item[1])
                                pupil_center = (int(largest[0][0]), int(largest[0][1]))
                                used_radius = largest[1]
                                print("Debug: Fallback to largest candidate (no initial center available).")
                    else:
                        candidate = candidate_circles[0]
                        pupil_center = (int(candidate[0][0]), int(candidate[0][1]))
                        used_radius = candidate[1]
                        print("Debug: Only one candidate detected.")

                    if initial_pupil_radius is not None:
                        max_allowed_radius = 1.25 * initial_pupil_radius
                    else:
                        max_allowed_radius = MAX_PUPIL_RADIUS
                    if used_radius > max_allowed_radius:
                        print(f"Debug: Candidate radius {used_radius:.2f} exceeds max allowed {max_allowed_radius:.2f}, capping it.")
                        used_radius = max_allowed_radius

                    if exp_smoothed_center is None:
                        exp_smoothed_center = pupil_center
                    else:
                        exp_smoothed_center = (
                            int(SMOOTHING_FACTOR * pupil_center[0] + (1 - SMOOTHING_FACTOR) * exp_smoothed_center[0]),
                            int(SMOOTHING_FACTOR * pupil_center[1] + (1 - SMOOTHING_FACTOR) * exp_smoothed_center[1])
                        )
                    pupil_center = (int(exp_smoothed_center[0]), int(exp_smoothed_center[1]))
                    recent_positions.append(pupil_center)
                    if len(recent_positions) > MAX_RECENT_FRAMES:
                        recent_positions.pop(0)
                    
                    recent_radii.append(used_radius)
                    if len(recent_radii) > MAX_RECENT_FRAMES:
                        recent_radii.pop(0)
                    running_radius = np.mean(recent_radii)
                    
                    if abs(used_radius - running_radius) > MAX_RADIUS_DELTA:
                        if used_radius > running_radius:
                            running_radius = running_radius + MAX_RADIUS_DELTA
                        else:
                            running_radius = running_radius - MAX_RADIUS_DELTA
                    used_radius = running_radius

                    print(f"Debug: Current pupil radius: {used_radius:.2f}")

                    last_valid_pupil_center = pupil_center

                    # Draw detection on the processed frame (optional).
                    cv2.circle(processed_frame, pupil_center, int(used_radius), (0, 0, 255), 2)
                    cv2.putText(processed_frame, f"X: {pupil_center[0]} Y: {pupil_center[1]}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    pupil_positions.append((video_capture.get(cv2.CAP_PROP_POS_MSEC),
                                             pupil_center[0],
                                             pupil_center[1]))
                    
                    # Create an overlay frame: use the original frame (with original contrast/exposure),
                    # apply the same rotation, and overlay the detected pupil.
                    overlay_frame = apply_rotation(original_frame, rotation_angle)
                    cv2.circle(overlay_frame, pupil_center, int(used_radius), (0, 0, 255), 2)
                    cv2.putText(overlay_frame, f"X: {pupil_center[0]} Y: {pupil_center[1]}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    out.write(overlay_frame)
                else:
                    out.write(apply_rotation(original_frame, rotation_angle))
            else:
                out.write(apply_rotation(original_frame, rotation_angle))
            
            if progress_callback:
                progress = int(100 * (current_frame - start_frame) / total_frames_to_process)
                progress_callback(progress)
        
        out.release()
        return output_path, pupil_positions
    except Exception as e:
        print("Error in track_pupil_with_video_output:", e)
        return None, []

def save_tracking_data(positions, current_session_dir):
    df = pd.DataFrame(positions, columns=['Time (ms)', 'X Position', 'Y Position'])
    if not df.empty:
        t0 = df['Time (ms)'].iloc[0]
        df['Time (ms)'] = df['Time (ms)'] - t0
        x_start = df['X Position'].iloc[0]
        y_start = df['Y Position'].iloc[0]
        df['X Deviation'] = x_start - df['X Position']
        df['Y Deviation'] = y_start - df['Y Position']
        csv_output_path = os.path.join(current_session_dir, "eye_tracking_output.csv")
        df.to_csv(csv_output_path, index=False)
        return csv_output_path
    return None

def plot_normalized_tracking_data(df, current_session_dir):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(df['Time (ms)'], df['X Deviation'], 'b-')
    plt.ylabel("X Deviation")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(df['Time (ms)'], df['Y Deviation'], 'g-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Y Deviation")
    plt.grid(True)
    ax1 = plt.subplot(2,1,1)
    cur_ylim1 = ax1.get_ylim()
    ax1.set_ylim(min(cur_ylim1[0], -40), max(cur_ylim1[1], 40))
    ax2 = plt.subplot(2,1,2)
    cur_ylim2 = ax2.get_ylim()
    ax2.set_ylim(min(cur_ylim2[0], -20), max(cur_ylim2[1], 20))
    plt.tight_layout()
    graph_output_path = os.path.join(current_session_dir, f"eye_tracking_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(graph_output_path)
    print(f"Tracking graph saved to: {graph_output_path}")
    plt.close()