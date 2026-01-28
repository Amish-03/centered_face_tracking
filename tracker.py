import cv2

class FaceTracker:
    def __init__(self):
        self.tracker = None
        self.tracking_active = False

    def init(self, frame, bbox):
        """
        Initialize the KCF tracker with the initial face bounding box.
        :param frame: The current video frame.
        :param bbox: The bounding box (x, y, w, h) of the face to track.
        """
        # Create a new tracker instance for each initialization
        # KCF is much faster than CSRT, though slightly less accurate. Good for real-time.
        try:
            self.tracker = cv2.TrackerKCF_create()
        except AttributeError:
             # Fallback for older/newer OpenCV versions or if KCF is in contrib
            print("Warning: KCF tracker not found, trying CSRT.")
            self.tracker = cv2.TrackerCSRT_create()
            
        self.tracker.init(frame, bbox)
        self.tracking_active = True

    def update(self, frame):
        """
        Update the tracker with the current frame.
        :param frame: The current video frame.
        :return: (success, bbox)
                 success: Boolean indicating if tracking was successful.
                 bbox: The updated bounding box (x, y, w, h) or None if failed.
        """
        if not self.tracking_active or self.tracker is None:
            return False, None

        success, bbox = self.tracker.update(frame)
        if success:
            # bbox is returned as a tuple of floats by OpenCV, convert to int
            bbox = tuple(map(int, bbox))
            return True, bbox
        else:
            self.tracking_active = False
            return False, None
