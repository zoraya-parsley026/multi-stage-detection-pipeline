import cv2
import numpy as np


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """Draws one bounding box on image (in-place)."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        
        # Guard against label pushing out of the top of the image
        if c1[1] - t_size[1] - 3 < 0:
            c2_bg = c1[0] + t_size[0], c1[1] + t_size[1] + 3
            cv2.rectangle(img, c1, c2_bg, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        else:
            c2_bg = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2_bg, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def draw_detections(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draws Contract 2 detections on the image."""
    img_copy = img.copy()
    
    # Predefined color palette for distinct classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
        (0, 255, 255), (255, 0, 255), (192, 192, 192), (128, 128, 128), 
        (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128), 
        (0, 128, 128), (0, 0, 128)
    ]
    
    for det in detections:
        xyxy = det["localization"]
        conf = det["confidence"]
        cls = det["class_id"]
        
        color = colors[cls % len(colors)]
        label = f"cls:{cls} {conf:.2f}"
        
        plot_one_box(xyxy, img_copy, color=color, label=label)
        
    return img_copy

def show_image(img, window_name="Detections"):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img, path):
    cv2.imwrite(str(path), img)
