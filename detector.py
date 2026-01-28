import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Initialize MediaPipe Face Detection.
        :param min_detection_confidence: Minimum confidence value ([0.0, 1.0]).
        :param model_selection: 0 for short-range (<=2m), 1 for full-range (<=5m).
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )

    def detect(self, frame):
        """
        Detects the largest face in the frame.
        :param frame: BGR image (numpy array) from OpenCV.
        :return: Tuple (x, y, w, h) of the largest face bounding box, or None if no face found.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)

        if not results.detections:
            return None

        img_h, img_w, _ = frame.shape
        largest_face = None
        max_area = 0

        for detection in results.detections:
            # mp_face_detection returns relative bounding box
            bboxC = detection.location_data.relative_bounding_box
            
            # Convert to absolute integer coordinates
            x = int(bboxC.xmin * img_w)
            y = int(bboxC.ymin * img_h)
            w = int(bboxC.width * img_w)
            h = int(bboxC.height * img_h)

            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = (x, y, w, h)

        return largest_face
