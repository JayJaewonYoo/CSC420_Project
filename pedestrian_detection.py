import cv2
import numpy as np
import os

video_src = 'pedestrians.avi'
output_dir = "output/"
cap = cv2.VideoCapture(video_src)

pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')

os.makedirs(output_dir, exist_ok=True)

frame_number = 0

while True:
    ret, img = cap.read()

    if not ret:
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bboxes = pedestrian_cascade.detectMultiScale(img_gray, 1.3, 2)

    detections = np.array(bboxes)
    mask = np.zeros_like(img_gray)

    display = img.copy()
    for (y, x, h, w) in bboxes:
        cv2.rectangle(display, (y, x), (y + h, x + w), (0, 255, 0), 4)
        mask[y:y+h,x:x+w] = 255

    np.save(os.path.join(output_dir, "frame%06d_detections.npy" % frame_number), detections)
    np.save(os.path.join(output_dir, "frame%06d_mask.npy" % frame_number), mask)
    np.save(os.path.join(output_dir, "frame%06d_image.npy" % frame_number), img)
    cv2.imwrite(os.path.join(output_dir, "frame%06d_display.png" % frame_number), display)
    print("Processed frame: %6d" % frame_number)

    frame_number += 1
