import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import matplotlib.path as mplPath

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


Color = [(255,0,0), (0,255,0), (0,0,255)]

## Dimensiones del video
# (480, 848, 3)

'''############################################ ZONES ############################################'''
zoneA=np.array([
    [62, 155],
    [121, 178],
    [87, 218],
    [10, 218]
])

zoneB=np.array([
    [514, 425],
    [606, 384],
    [691, 445],
    [613, 475]
])

zoneC=np.array([
    [712, 113],
    [781, 130],
    [777, 187],
    [667, 162]
])

zoneD = np.array([
    [802, 249],
    [741, 308],
    [825, 336],
    [825, 249]
])

'''############################################ OPTICAL FLOW ############################################'''
# Params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                    qualityLevel = 0.3,
                    minDistance = 7,
                    blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def inside_boxes(xdet, ydet, boxes, valid_bx):
    for i, box in enumerate(boxes):
        x, y, xmax, ymax = box
        if xdet>=x and xdet<=xmax and ydet>=y and ydet<=ymax and (valid_bx[i] is False):
            valid_bx[i]=True
            return True
    return False

# def direction_object(current_xdet, current_ydet, prev_xdet, prev_ydet):


'''################################## FUNCTIONS FOR COUNTING CLASSES ###################################'''
def middle_point(coords):
    return (int(coords[0])+int(coords[2]))//2,(int(coords[1])+int(coords[3]))//2
    
def inside_area(xcenter, ycenter):
    if mplPath.Path(zoneA).contains_point((xcenter,ycenter)):
        return 'A'
    elif mplPath.Path(zoneB).contains_point((xcenter,ycenter)):
        return 'B'
    elif mplPath.Path(zoneC).contains_point((xcenter,ycenter)):
        return 'C'
    elif mplPath.Path(zoneD).contains_point((xcenter,ycenter)):
        return 'D'
    else:
        return None



def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Filtering by people by default
    if opt.classes is None:
        opt.classes = [0]

    # variables to work with the previous frame
    prev_frame = None
    prev_det = None
    prev_gray = None

    # variables to work with the previous frame for apply Optical Flow
    next_points = [] # Verify points detect
    prev_points = None
    mask = None

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    prev_colors = [[abs(random.randint(0, 255)-random.randint(0, 255)) for _ in range(3)] for _ in names]


    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        frame_gray = cv2.cvtColor(im0s, cv2.COLOR_BGR2GRAY)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()
        
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()


        # # Process detections
        elementsA, elementsB, elementsC, elementsD = 0, 0, 0, 0
        
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    cv2.putText(im0s, f"{names[int(c)]}: {n}", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, Color[0], 2)

                # Write results
                for *xyxy, conf, cls in det:
                    if save_img or view_img:  # Add bbox to image
                        xm,ym = middle_point(xyxy)
                        ch = inside_area(xm,ym)

                        # Verifies whether a detected object is within any zone
                        if ch is not None:
                            if ch == 'A': elementsA = elementsA + 1
                            elif ch == 'B': elementsB = elementsB + 1
                            elif ch == 'C': elementsC = elementsC + 1
                            else: elementsD = elementsD + 1
                        
                        label = f'{names[int(cls)]}{conf:.2f}'
                        plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)
                        # Display the midpoint of each box in the current frame
                        # cv2.circle(img = im0s, center =(xm,ym), radius=5, color=(0,255,0), thickness=1) #REF

                        # Verify exist previous points
                        if prev_points is not None: 
                            next_points.append(xyxy)
        
        #display elements in areas
        cv2.putText(im0s, f"Zona A: {elementsA}", (0,60),   cv2.FONT_HERSHEY_SIMPLEX, 1, (240,255,51),  2)
        cv2.putText(im0s, f"Zona B: {elementsB}", (0,90),   cv2.FONT_HERSHEY_SIMPLEX, 1, (51,255,94),   2)
        cv2.putText(im0s, f"Zona C: {elementsC}", (180,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (51,156,255),  2)
        cv2.putText(im0s, f"Zona D: {elementsD}", (180,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,50,250),  2)

        # Storing the previously detected elements                
        if prev_frame is not None:
            for *xyxy, conf, cls in prev_det:
                    if save_img or view_img:
                        label = f'pp{conf:.2f}'
                        plot_one_box(xyxy, im0s, label=label, color=prev_colors[int(cls)], line_thickness=1)
            prev_det=[]

        # Valid for optical flow application
        if prev_points is not None:

            match_boxes = [False for _ in range(len(next_points))]
            current_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_points, None, **lk_params)

            # Select good points of 'prev' and 'new' frame
            if current_points is not None:
                good_new = current_points[st==1]
                good_prev = prev_points[st==1]

            # Save points detection boxes
            auxiliar_points = np.empty((0,2))
            
            # Draw the tracks
            for i, (new, prev) in enumerate(zip(good_new, good_prev)):
                a, b = new.ravel()
                c, d = prev.ravel()
                if(inside_boxes(a,b,next_points, match_boxes)):
                    auxiliar_points = np.concatenate((auxiliar_points,[[a,b]]), axis=0)
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (159,51,255), 2)
                    im0s = cv2.circle(im0s, (int(a), int(b)), 5, (51,51,255), -1)
            im0s = cv2.add(im0s, mask)
            next_points = []
            print(auxiliar_points.shape)
        # Save prev frames
        if prev_points is None:
            mask = np.zeros_like(im0s)
            prev_points = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
            # for pp in prev_points:
                # a, b = pp.ravel()
                # if (inside_boxes(a,b,next_points)):
        else:
            prev_points = good_new.reshape(-1,1,2)

        
        prev_frame = im0s
        prev_det = det
        prev_gray = frame_gray.copy()

        # Drawing zones in frame
        cv2.polylines(img=im0s, pts=[zoneA], isClosed = True, color = (240,255,51),  thickness=3)
        cv2.polylines(img=im0s, pts=[zoneB], isClosed = True, color = (51,255,94),   thickness=3)
        cv2.polylines(img=im0s, pts=[zoneC], isClosed = True, color = (51,156,255),  thickness=3)
        cv2.polylines(img=im0s, pts=[zoneD], isClosed = True, color = (255,50,250),  thickness=3)
        
        #Show Video
        cv2.imshow("YOLO-V7",im0s)
        cv2.waitKey(20)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    # print(opt)

    with torch.no_grad():
        detect()
