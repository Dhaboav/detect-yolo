import cv2 as cv
import numpy as np
import torch
from vision.depth_location import DepthLocation as Depth

class Detect:
    def __init__(self, yolo_path, model_path, index, width, height):
        # Yolo env
        self.yolo_path = yolo_path
        self.model_path = model_path

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(self.yolo_path, 'custom', path=self.model_path, source="local").to(self.device)

        # Camera setting
        self.camera = cv.VideoCapture(index)
        self.camera_size = (width, height)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.camera_size[0])
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, self.camera_size[1])

        # Other
        self.depth = Depth()

    def apply_roi(self, frame, x, y, radius):
        circle_mask = np.zeros_like(frame)
        cv.circle(img=circle_mask, center=(x, y), radius=radius, color=(255, 255, 255), thickness=-1)
        circle_roi = cv.bitwise_and(frame, circle_mask)
        return circle_roi

    def run(self):
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                break

            # Apply ROI
            radius = 230
            x_center = frame.shape[1] // 2
            y_center = frame.shape[0] // 2

            circle_roi = self.apply_roi(frame=frame, x=x_center, y=y_center-30, radius=radius)
            output = circle_roi.copy()

            # Kartesian
            self.depth.draw_kartesian(output=output, x=x_center, y=y_center, radius=radius, value=13)

            # Model detection
            detections = self.model(circle_roi[..., ::-1])
            detect = detections.pandas().xyxy[0].to_dict(orient="records")

            color_mapping = {
                "ROBOT": (0, 255, 0),
                "BOLA": (0, 140, 255),
                "PENGHALANG": (0, 0, 255),
                "GAWANG": (255, 255, 255)
            }

            for info in detect:
                id_class, x1, y1, x2, y2 = info["class"], int(info['xmin']), int(info['ymin']), int(info['xmax']), int(info['ymax'])
                class_name = self.model.names[id_class]

                # centeroid
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                color = color_mapping.get(class_name, (0, 0, 0))  # Default to black if class_name is not found

                cv.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cv.circle(output, (mid_x, mid_y), 2, color, -1)

                a = self.depth.quadrant(mid_x, mid_y, x_center, y_center)
                print(a) if class_name in color_mapping else None
                
            cv.imshow("Depth Estimation", output)
            if cv.waitKey(1) & 0xFF == ord("x"):
                break

        self.camera.release()
        cv.destroyAllWindows()



# Given paths
yolo_path = r'D:\repository\yolov5'
model_path = r'robot5s.pt'

app = Detect(yolo_path, model_path, 1, 640, 480)
app.run()