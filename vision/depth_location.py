import cv2 as cv

class DepthLocation:
    def __init__(self) -> None:
        pass

    def focal_length(self, distance, real_width, pixel_width):
        return (pixel_width * distance) / real_width

    def depth_estimation(self, focal_length, real_width, pixel_width):
        return (real_width * focal_length) / pixel_width
    
    def draw_kartesian(self, output, x, y, radius, value):
        cv.circle(img=output, center=(x - value, y - value), radius=5, color=(0, 255, 0), thickness=-1)
        cv.line(img=output, pt1=(x - radius, y - value), pt2=(x + radius, y - value), color=(0, 255, 0), thickness=2)
        cv.line(img=output, pt1=(x - value, y - radius), pt2=(x - value, y + radius), color=(0, 255, 0), thickness=2)

    def quadrant(self, x_object, y_object, x_center, y_center):
        if (x_object > x_center) and (y_object < y_center):
            return 1
        elif (x_object < x_center) and (y_object < y_center):
            return 2
        elif (x_object < x_center) and (y_object > y_center):
            return 3
        elif (x_object > x_center) and (y_object > y_center):
            return 4