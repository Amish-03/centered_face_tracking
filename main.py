import cv2
import time
from detector import FaceDetector
from tracker import FaceTracker
from zoom import ZoomController

def main():
    # Initialize components
    # 0 is usually the default camera.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Please ensure your webcam is connected and not used by another application.")
        return

    # Initialize modules
    # min_detection_confidence: 0.6 ensures we don't pick up garbage
    detector = FaceDetector(min_detection_confidence=0.6)
    tracker = FaceTracker()
    
    # smoothing_factor: 0.1 (very smooth, more lag) to 0.9 (less smooth, fast response)
    # margin_factor: 0.5 means 50% extra margin around the face
    zoom_controller = ZoomController(smoothing_factor=0.15, margin_factor=0.6)

    print("Starting Face Tracking & Zoom System (Optimized)...")
    print("Press 'q' to quit.")

    prev_time = time.time()
    
    # Processing width (smaller = faster)
    PROCESS_WIDTH = 640

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Create a copy for processing (detection/tracking)
        # Resize to fixed width, maintaining aspect ratio
        h_orig, w_orig = frame.shape[:2]
        scale = PROCESS_WIDTH / w_orig
        h_proc = int(h_orig * scale)
        
        frame_proc = cv2.resize(frame, (PROCESS_WIDTH, h_proc))

        # Check tracking status on logic frame
        tracking_success, bbox_proc = tracker.update(frame_proc)

        final_bbox = None

        if not tracking_success:
            # If tracking failed or not active, try detection on logic frame
            detected_face_proc = detector.detect(frame_proc)
            
            if detected_face_proc:
                # Initialize tracker with detected face
                tracker.init(frame_proc, detected_face_proc)
                bbox_proc = detected_face_proc
                tracking_success = True
            else:
                bbox_proc = None

        # If we have a bounding box from processing frame, scale it back to original
        if bbox_proc:
            px, py, pw, ph = bbox_proc
            final_bbox = (
                int(px / scale),
                int(py / scale),
                int(pw / scale),
                int(ph / scale)
            )

        # Process zoom on the ORIGINAL high-quality frame using the scaled-up bbox
        output_frame = zoom_controller.process(frame, final_bbox)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Display FPS on output
        cv2.putText(output_frame, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show output
        cv2.imshow("Smart Zoom", output_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
