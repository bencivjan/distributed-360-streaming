import cv2
from ultralytics import YOLO
import json
import os
import pandas as pd
import numpy as np
import torch

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1, box2: Lists or tuples with 4 elements each [x, y, width, height].

    Returns:
    - IoU: Intersection over Union (float).
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)

    inter_area = inter_width * inter_height
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def upperleft_xywh2xyxy(xywh):
    x, y, w, h = xywh
    x2 = x + w
    y2 = y + h
    return x, y, x2, y2

def middle_xywh2xyxy(xywh):
    x, y, w, h = xywh
    x2 = x + w/2
    y2 = y + h/2
    x = x - w/2
    y = y - h/2
    return x, y, x2, y2


def plot_bounding_boxes(image, bounding_boxes, output_path=None):
    if image is None:
        print(f"No image found")
        return

    # Plot each bounding box
    for bbox in bounding_boxes:
        x, y, x2, y2, c = bbox
        x = round(x)
        y = round(y)
        x2 = round(x2)
        y2 = round(y2)

        if c == 'r':
            color_code = (0, 0, 255)
        elif c == 'g':
            color_code = (0, 255, 0)
        elif c == 'b':
            color_code = (255, 0, 0)
        else:
            print('Unknown Color')
            color_code = (0, 0, 255)

        # Draw the rectangle
        cv2.rectangle(image, (x, y), (x2, y2), color_code, 2)
    
    return image

def crosswalk_main():
    VIDEOS = ['crosswalk_250k_30.mp4', 'crosswalk_500k_30.mp4', 'crosswalk_750K_30.mp4', 'crosswalk_1m_30.mp4',
              'crosswalk_250k_15.mp4', 'crosswalk_500k_15.mp4', 'crosswalk_750k_15.mp4', 'crosswalk_1m_15.mp4',
              'crosswalk_250k_10.mp4', 'crosswalk_500k_10.mp4', 'crosswalk_750k_10.mp4', 'crosswalk_1m_10.mp4']

    for video in VIDEOS:
        vid_name = video.split('.')[0]

        yolov8n_model = YOLO('yolov8l.pt')
        cap = cv2.VideoCapture(os.path.join('videos', video))
        labels = pd.read_csv('crosswalk.csv')

        ret = True
        iou = {}
        idx = 0

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fps = int(video.split('.')[0].split('_')[-1])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(width, height)
        os.mkdir('out_videos', exist_ok=True)
        vid_writer = cv2.VideoWriter(f'out_videos/boxed_{vid_name}.mp4', fourcc, fps, (width, height))

        while True:
            ret, image = cap.read()
            
            if ret is False:
                print('Finished reading video frames...')
                break

            bounding_boxes = []

            ground_truth = upperleft_xywh2xyxy(labels.iloc[idx])
            bounding_boxes.append((*ground_truth, 'g'))

            results = yolov8n_model.predict(image, verbose=False)
            for box in results[0].boxes:
                if box.cls.numpy()[0] == 0: # Is a person
                    bb = middle_xywh2xyxy(box.xywh.numpy()[0])
                    bounding_boxes.append((*bb, 'r'))

            print(bounding_boxes)
            image_boxed = plot_bounding_boxes(image, bounding_boxes)

            # os.makedirs('imgs', exist_ok=True)
            # cv2.imwrite(f'imgs/frame{idx}.jpg', image_boxed)
            vid_writer.write(image_boxed)

            if len(bounding_boxes) > 1:
                iou[f'frame{idx}'] = calculate_iou(bounding_boxes[0][:4], bounding_boxes[1][:4])
            else:
                iou[f'frame{idx}'] = 0

            idx += (30 // fps)

        cap.release()
        vid_writer.release()

        s = 0
        for val in iou.values():
            s += val

        iou['Average Iou'] = s / len(iou.values())

        with open(f'{vid_name}_iou.json', 'a') as f:
            json.dump(iou, f, indent=4)

def driving_main():
    VIDEOS = ['nydriving_1m_10.mp4', 'nydriving_1m_15.mp4', 'nydriving_1m_30.mp4',
              'nydriving_3m_10.mp4', 'nydriving_3m_15.mp4', 'nydriving_3m_30.mp4',
              'nydriving_5m_10.mp4', 'nydriving_5m_15.mp4', 'nydriving_5m_30.mp4']

    for video in VIDEOS:
        vid_name = video.split('.')[0]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        truth_model = YOLO('yolov8x.pt').to(device)
        pred_model = YOLO('yolov8n.pt').to(device)

        ret = True
        iou = {}
        idx = 0

        gt = cv2.VideoCapture(os.path.join('videos', 'nydriving_5m_30.mp4')) # Use highest quality video for ground truth predictions
        gt_fps = int(gt.get(cv2.CAP_PROP_FPS))

        cap = cv2.VideoCapture(os.path.join('videos', video))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(width, height)
        os.makedirs('out_videos', exist_ok=True)
        vid_writer = cv2.VideoWriter(f'out_videos/boxed_{vid_name}.mp4', fourcc, gt_fps, (width, height))

        while True:
            gt_ret, gt_image = gt.read()
            # Only read new image from test video if ground truth frame index is divisible by fps ratio
            # E.g. gt_fps = 30, fps = 10. We want to read a new test video frame every 3 ground truth frames
            #       to avoid running out of frames early. We will reuse the same predictions for these intermediate frames.
            if idx % (gt_fps // fps) == 0:
                ret, image = cap.read()
                print(idx)
            
            if ret is False:
                print('Finished reading video frames...')
                break
            elif gt_ret is False:
                print('Ground truth video ran out of frames... double check video files')
                break

            bounding_boxes = []

            if idx % (gt_fps // fps) == 0: # Same logic as above
                results = pred_model.predict(image, verbose=False)

            for box in results[0].boxes:
                bb = middle_xywh2xyxy(box.xywh.numpy()[0])
                bounding_boxes.append((*bb, 'r'))

            ground_truth = truth_model.predict(gt_image, verbose=False)
            gt_to_pred_iou = np.zeros((len(ground_truth[0].boxes), len(results[0].boxes)))

            for i, truth in enumerate(ground_truth[0].boxes):
                truth_bb = middle_xywh2xyxy(truth.xywh.numpy()[0])
                bounding_boxes.append((*bb, 'g'))
                
                for j, pred in enumerate(results[0].boxes):
                    pred_bb = middle_xywh2xyxy(pred.xywh.numpy()[0])
                    if pred.cls[0] == truth.cls[0]:
                        gt_to_pred_iou[i][j] = calculate_iou(truth_bb, pred_bb)


            gt_to_pred_iou = gt_to_pred_iou.max(axis=1)
            iou[f'frame{idx}'] = gt_to_pred_iou.mean()

            print(gt_to_pred_iou)
            print(gt_to_pred_iou.mean())

            image_boxed = plot_bounding_boxes(gt_image, bounding_boxes)

            vid_writer.write(image_boxed)

            idx += 1

        cap.release()
        vid_writer.release()

        s = 0
        for val in iou.values():
            s += val

        if len(iou.values()) > 0:
            iou['Average Iou'] = s / len(iou.values())

        with open(f'{vid_name}_iou.json', 'a') as f:
            json.dump(iou, f, indent=4)

if __name__ == '__main__':
    driving_main()