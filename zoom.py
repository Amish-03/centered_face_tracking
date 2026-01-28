import cv2
import numpy as np

class ZoomController:
    def __init__(self, smoothing_factor=0.2, margin_factor=0.5):
        """
        Initialize the ZoomController.
        :param smoothing_factor: Factor for exponential moving average (0 < alpha <= 1).
        :param margin_factor: Margin to add around the face as a fraction of the max face dimension.
        """
        self.smoothing_factor = smoothing_factor
        self.margin_factor = margin_factor
        
        # Current smoothed state [x, y, w, h]
        self.current_box = None

    def reset(self):
        """Reset the internal state, e.g. when tracking is lost."""
        self.current_box = None

    def process(self, frame, target_bbox):
        """
        Update the smooth bounding box based on target_bbox and return the zoomed/cropped frame.
        If target_bbox is None, returns the original frame (or maintains last position? User said 're-run detection').
        
        For this implementation: 
        If target_bbox is valid, we smooth and crop.
        If target_bbox is None (tracking lost/initializing), we just return the full frame 
        OR we could return the last valid crop. Ideally, the main loop provides a box from detector if tracker fails.
        """
        if target_bbox is None:
            # If we have no target, we can't really zoom meaningfully on a user.
            # We could return the full frame.
            return frame

        x, y, w, h = target_bbox
        target_box = np.array([x, y, w, h], dtype=np.float32)

        if self.current_box is None:
            self.current_box = target_box
        else:
            # Exponential Moving Average
            # new_val = alpha * target + (1 - alpha) * old_val
            self.current_box = (self.smoothing_factor * target_box + 
                                (1 - self.smoothing_factor) * self.current_box)

        # 1. Calculate Crop Region based on smoothed box
        sx, sy, sw, sh = self.current_box
        
        # Center of the face
        cx, cy = sx + sw / 2, sy + sh / 2
        
        # Determine crop size (height-based)
        # The crop should be tall enough to cover the face + margin
        crop_dim = max(sw, sh) * (1 + self.margin_factor)
        
        img_h, img_w = frame.shape[:2]
        
        # Handle case where crop_dim is larger than image (zooming out beyond image bounds)
        # Limit crop_dim to fit within the smallest dimension relative to aspect ratio?
        # Simpler: just clamp calculations later.
        
        # Aspect Ratio of the Output (Original Frame)
        aspect_ratio = img_w / img_h
        
        crop_h = crop_dim
        crop_w = crop_h * aspect_ratio
        
        # 2. Calculate coordinates centered at (cx, cy)
        crop_x = cx - crop_w / 2
        crop_y = cy - crop_h / 2
        
        # 3. Clamp to image boundaries (Shift method)
        # Shift rights if < 0
        if crop_x < 0: crop_x = 0
        # Shift down if < 0
        if crop_y < 0: crop_y = 0
        
        # Shift left if exceeds width
        if crop_x + crop_w > img_w: crop_x = img_w - crop_w
        # Shift up if exceeds height
        if crop_y + crop_h > img_h: crop_y = img_h - crop_h
        
        # Final validity check after shifting (double check for small images or huge margins)
        if crop_x < 0: crop_x = 0
        if crop_y < 0: crop_y = 0
        
        # 4. Crop
        x1, y1 = int(crop_x), int(crop_y)
        x2, y2 = int(crop_x + crop_w), int(crop_y + crop_h)
        
        # Ensure we don't crash with invalid indices
        x2 = min(x2, img_w)
        y2 = min(y2, img_h)
        
        if x2 <= x1 or y2 <= y1:
            return frame # Fallback
            
        cropped = frame[y1:y2, x1:x2]
        
        # 5. Resize to original resolution
        try:
            zoomed = cv2.resize(cropped, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            return zoomed
        except Exception as e:
            print(f"Zoom error: {e}")
            return frame
