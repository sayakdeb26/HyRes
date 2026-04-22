import cv2
import mediapipe as mp

class Visualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def draw_roi(self, image, bbox, label="ROI"):
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image, label, (x_min, int(max(0, y_min - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def draw_skeleton(self, image, keypoints):
        if keypoints:
            connections = self.mp_hands.HAND_CONNECTIONS
            
            # Draw lines
            for connection in connections:
                idx1, idx2 = connection
                if idx1 < len(keypoints) and idx2 < len(keypoints):
                    p1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                    p2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                    cv2.line(image, p1, p2, (0, 255, 0), 2)
                    
            # Draw points
            for kp in keypoints:
                cv2.circle(image, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

    def draw_prediction(self, image, prediction, hand_label):
        h, w, _ = image.shape
        x = 50 if hand_label == 'left' else w - 300
        y = 80
        
        text = f"{hand_label.upper()}: {prediction}"
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def draw_fps(self, image, fps):
        cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
