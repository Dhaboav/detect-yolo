import cv2 as cv
import torch

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load(r'D:\repository\yolov5', 'yolov5s', source='local', pretrained=True).to(device)

# Distance
real_distance = 30
object_real_width = 5.5

# Formula of focal and depth
def focal_length(distance, real_width, pixel_width):
    focal = (pixel_width * distance) / real_width
    return focal

def depth_estimation(focal_length, real_width, pixel_width):
    depth = (real_width * focal_length) / pixel_width
    return depth

camera = cv.VideoCapture(1)
while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    output = frame.copy()
    detections = model(frame[..., ::-1])
    detect = detections.pandas().xyxy[0].to_dict(orient="records")
    for info in detect:
        id_class, x1, y1, x2, y2 = info["class"], int(info['xmin']), int(info['ymin']), int(info['xmax']), int(info['ymax'])
        class_name = model.names[id_class]
        
        
        if class_name == 'cell phone':
            cv.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            width_pixels = int((x2 - x1) * frame.shape[1]) if 0 <= x1 <= 1 and 0 <= x2 <= 1 else int(x2 - x1)
            d = depth_estimation(540, object_real_width, width_pixels)
            cv.putText(output, f'Jarak: {d:.0f} cm', (x1, y1-10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

    cv.imshow("Depth Estimation", output)    
    if cv.waitKey(1) & 0xFF == ord("x"):
        break

camera.release()
cv.destroyAllWindows()