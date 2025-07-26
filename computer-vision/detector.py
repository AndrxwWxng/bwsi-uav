import cv2
import numpy as np


class SphereDetector:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def detect_yellow_balls(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yellow_balls = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                if radius > 8:
                    circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
                    if circularity > 0.5:
                        yellow_balls.append({
                            'center': center,
                            'radius': radius,
                            'area': area,
                            'type': 'yellow_ball'
                        })

        return yellow_balls
    
    def process_frame(self, frame):
        yellow_balls = self.detect_yellow_balls(frame)

        result_frame = frame.copy()

        for ball in yellow_balls:
            cv2.circle(result_frame, ball['center'], ball['radius'], (0, 255, 255), 2)
            cv2.circle(result_frame, ball['center'], 2, (0, 255, 255), -1)
            cv2.putText(result_frame, 'ball',
                       (ball['center'][0] - 15, ball['center'][1] - ball['radius'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return result_frame, yellow_balls

def main():
    detector = SphereDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, yellow_balls = detector.process_frame(frame)

        cv2.imshow('detector', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
