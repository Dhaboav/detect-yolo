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
        
    def process_object(self, frame, x_center, y_center, x1, y1, x2, y2, focal_length, mid_x, mid_y, width_attribute, color, output):
        # Calculate width in pixels
        width_pixels = int((x2 - x1) * frame.shape[1]) if 0 <= x1 <= 1 and 0 <= x2 <= 1 else int(x2 - x1)
        
        # Store or use the width for the class
        distance_est = self.depth_estimation(focal_length, width_attribute, width_pixels)
        quadrant = self.quadrant(mid_x, mid_y, x_center, y_center)

        # Draw rectangle and circle
        cv.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv.circle(output, (mid_x, mid_y), 2, color, -1)

        # Draw text
        cv.putText(output, f'{int(distance_est)} cm', (x2, y1 - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        cv.putText(output, f'{quadrant}', (x1, y1 - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, color, 1)