import cv2
import numpy as np
import os

class PedestrianDetector():
    def __init__(self, cascade_classifier='pedestrian.xml'):
        self.pedestrian_cascade = cv2.CascadeClassifier(cascade_classifier)

    def get_frame_masks(self, video_src='pedestrians.avi', output_dir='output/', display_video='display.avi'):
        # Video Capture object
        cap = cv2.VideoCapture(video_src)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Output video
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter(display_video,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)

        masks = []
        frames = []

        frame_number = 0

        while True:
            ret, img = cap.read()

            if not ret:
                break

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Get bounding boxes
            bboxes = self.pedestrian_cascade.detectMultiScale(img_gray, 1.3, 2)

            detections = np.array(bboxes)
            mask = np.zeros_like(img_gray)

            display = img.copy()
            for idx, (x, y, w, h) in enumerate(bboxes, 1):
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 4)
                mask[y:y + h, x:x + w] = idx

            frames.append(img)
            masks.append(mask)
            result.write(display)

            frame_number += 1

        cap.release()
        result.release()

        frames = np.stack(frames, axis=0)
        masks = np.stack(masks, axis=0)

        return frames, masks


def main():
    # The input video
    video_src = 'pedestrians.avi'
    # The output directory
    output_dir = "output/"
    # Video Capture object
    cap = cv2.VideoCapture(video_src)

    # Load the pedestrian Haar Cascade from xml file
    pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Output video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('display.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)

    masks = []
    frames = []

    frame_number = 0

    while True:
        ret, img = cap.read()

        if not ret:
            break

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Get bounding boxes
        bboxes = pedestrian_cascade.detectMultiScale(img_gray, 1.3, 2)

        detections = np.array(bboxes)
        mask = np.zeros_like(img_gray)

        display = img.copy()
        for idx, (x, y, w, h) in enumerate(bboxes, 1):
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 4)
            mask[y:y+h,x:x+w] = idx

        frames.append(img)
        masks.append(mask)

        result.write(display)

        # np.save(os.path.join(output_dir, "frame%06d_detections.npy" % frame_number), detections)
        # np.save(os.path.join(output_dir, "frame%06d_mask.npy" % frame_number), mask)
        # np.save(os.path.join(output_dir, "frame%06d_image.npy" % frame_number), img)
        cv2.imwrite(os.path.join(output_dir, "frame%06d_mask.png" % frame_number), mask)
        cv2.imwrite(os.path.join(output_dir, "frame%06d_frame.png" % frame_number), img)
        cv2.imwrite(os.path.join(output_dir, "frame%06d_display.png" % frame_number), display)
        print("Processed frame: %6d" % frame_number)

        frame_number += 1

    frames = np.stack(frames, axis=0)
    masks = np.stack(masks, axis=0)

    print("Frames shape", frames.shape)
    print("Masks shape", masks.shape)

    np.save(os.path.join(output_dir, "frames.npy"), frames)
    np.save(os.path.join(output_dir, "frames.npy"), masks)

    cap.release()
    result.release()


if __name__ == '__main__':
    detector = PedestrianDetector('pedestrian.xml')
    frames, masks = detector.get_frame_masks()
    print(frames.shape)
    print(masks.shape)
