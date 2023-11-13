import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter as tk
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

# //===-----------------------------------------------------------------------===------------------//
'''############################################ ZONES ############################################'''
# //===-----------------------------------------------------------------------===------------------//
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

# //===-----------------------------------------------------------------------===-------------------------//
'''############################################ OPTICAL FLOW ############################################'''
# //===-----------------------------------------------------------------------===-------------------------//
# Mask
mask = None

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

def direction_object(current_xdet, current_ydet, prev_xdet, prev_ydet):
    xdif = prev_xdet - current_xdet
    ydif = prev_ydet - current_ydet
    if abs(ydif) > abs(xdif):
        if ydif < 0: return 'DOWN'
        else: return 'UP'
    else:
        if xdif > 0: return 'LEFT'
        else : return 'RIGHT' 

# //===-----------------------------------------------------------------------===------------------------//
'''################################## FUNCTIONS FOR COUNTING CLASSES ###################################'''
# //===-----------------------------------------------------------------------===------------------------//
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

# //===-----------------------------------------------------------------------===-----------//
'''################################## FUNCTIONS FOR GUI ###################################'''
# //===-----------------------------------------------------------------------===-----------//
def countZones(countA, countB, countC, countD):
    lblZoneA.config(text = f"N° Zone A: {countA}")
    lblZoneB.config(text = f"N° Zone B: {countB}")
    lblZoneC.config(text = f"N° Zone C: {countC}")
    lblZoneD.config(text = f"N° Zone D: {countD}")

def countPerDir(nPerson, nLeft, nRight, nUp, nDown):
    lblPerson.config(text =f"N° Person: {nPerson}")
    lblLeft.config(text = f"N° Left: {nLeft}")
    lblRight.config(text = f"N° Right: {nRight}")
    lblUp.config(text = f"N° Up: {nUp}")
    lblDown.config(text = f"N° Down: {nDown}")

# //===-----------------------------------------------------------------------===---------//
''' ################################## VARIABLE VIDEO ################################## '''
# //===-----------------------------------------------------------------------===---------//
dataset = None
model = None
names = None
colors = None
device = None
half = None
old_img_w = old_img_h = old_img_b = None
# variables to work with the previous frame
prev_frame = None
prev_det = None
prev_gray = None
prev_colors = None
# variables to work with the previous frame for apply Optical Flow
next_points = [] # Verify points detect
prev_points = None
mask = None

# //===-----------------------------------------------------------------------===-----//
''' ################################## LOAD MODEL ################################## '''
# //===-----------------------------------------------------------------------===-----//
def detect(save_img=False):
    global dataset, model, names, colors, prev_colors, old_img_w, old_img_h, old_img_b, device, half
    source, weights, imgsz, trace = opt.source, opt.weights, opt.img_size, not opt.no_trace

    # Filtering by people by default
    if opt.classes is None:
        opt.classes = [0]

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

# //===-----------------------------------------------------------------------===-----//
''' ################################## PLAY VIDEO ################################## '''
# //===-----------------------------------------------------------------------===-----//
def load_video():
    global dataset, model, prev_frame, prev_det, prev_gray, next_points, prev_points, mask, names, colors, prev_colors
    global old_img_w, old_img_h, old_img_b, device, half, mask

    # Adjustment btn
    btnVideo.grid(column=0, row=2, padx=10, pady=5)
    btnMask.grid(column=5, row=0, padx=10, pady=5)
    btnHist.grid(column=5, row=1, padx=10, pady=5)

    # Adjustment lbl
    lblPerson.grid(column = 0, row = 0, columnspan=1)
    lblZoneA.grid(column=1, row = 0, columnspan=1)
    lblZoneB.grid(column=2, row = 0, columnspan=1)
    lblZoneC.grid(column=3, row = 0, columnspan=1)
    lblZoneD.grid(column=4, row = 0, columnspan=1)
    lblLeft.grid(column=0, row = 1, columnspan=1)
    lblRight.grid(column=1, row = 1, columnspan=1)
    lblUp.grid(column=2, row = 1, columnspan=1)
    lblDown.grid(column=3, row = 1, columnspan=1)

    _,img,im0s,_ = next(iter(dataset))

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
    left, right, up, down = 0, 0, 0, 0
    n = 0
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            # Print results
            ix = 0
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # cv2.putText(im0s, f"{names[int(c)]}: {n}", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, Color[0], 2)


            # Write results
            for *xyxy, conf, cls in det:
                # Add bbox to image
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
    countZones(elementsA, elementsB, elementsC, elementsD)

    # Storing the previously detected elements                
    if prev_frame is not None:
        for *xyxy, conf, cls in prev_det:
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
                # Counting object direction
                direction = direction_object(a,b,c,d)
                if direction == 'UP': up += 1
                elif direction == 'DOWN': down += 1
                elif direction == 'LEFT': left += 1
                else: right += 1

                auxiliar_points = np.concatenate((auxiliar_points,[[a,b]]), axis=0)
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (159,51,255), 2)
                im0s = cv2.circle(im0s, (int(a), int(b)), 5, (51,51,255), -1)
        im0s = cv2.add(im0s, mask)
        next_points = []

    # Show count of object direction
    countPerDir(n,left,right,up,down)

    # Save prev frames
    if prev_points is None:
        mask = np.zeros_like(im0s)
        prev_points = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    else:
        prev_points = auxiliar_points.reshape(-1,1,2)
        save_new_points = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
        prev_points = np.concatenate((prev_points,save_new_points), axis = 0)
        prev_points = np.array(prev_points, dtype= np.float32)
        
    prev_frame = im0s
    prev_det = det
    prev_gray = frame_gray.copy()

    # Drawing zones in frame
    cv2.polylines(img=im0s, pts=[zoneA], isClosed = True, color = (0,0,255),  thickness=3)
    cv2.polylines(img=im0s, pts=[zoneB], isClosed = True, color = (0,255,66),   thickness=3)
    cv2.polylines(img=im0s, pts=[zoneC], isClosed = True, color = (255,243,0),  thickness=3)
    cv2.polylines(img=im0s, pts=[zoneD], isClosed = True, color = (255,0,0),  thickness=3)

    # Play Mask
    # maks1 = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    # maks1 = Image.fromarray(maks1)
    # maks1 = ImageTk.PhotoImage(image=maks1)
    # lblMask.configure(image=maks1)
    # lblMask.image = maks1

    # Play Video
    im0s = cv2.cvtColor(im0s, cv2.COLOR_RGB2BGR)
    imagen = Image.fromarray(im0s)
    img1 = ImageTk.PhotoImage(image=imagen)
    lblVideo.configure(image=img1)
    lblVideo.image = img1
    lblVideo.after(10, load_video)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    with torch.no_grad():
        detect()

# ''' ############################# LOAD TKINTER #############################'''

root = Tk()
root.title = "Image Proccessing - TF"

# Play Video
btnVideo = Button(root, text="Play", command=load_video)
btnVideo.grid(column=0, row=0, padx=10, pady=5)

# Button Option
btnMask = Button(root, text="Mask")
btnMask.grid(column=1, row=0, padx=10, pady=5)

btnHist = Button(root, text="Hist")
btnHist.grid(column=1, row=1, padx=10, pady=5)

# Text Container Total Classes
lblPerson = Label(root)
lblPerson.grid(column = 2, row = 0, columnspan = 1)
lblPerson.config(fg="purple", font=("Arial", 14))

# Text Container Zones
lblZoneA = Label(root)
lblZoneA.grid(column=2, row = 0, columnspan=1)
lblZoneA.config(fg="red",font=("Arial", 14))

lblZoneB = Label(root)
lblZoneB.grid(column=3, row = 0, columnspan=1)
lblZoneB.config(fg='lawn green',font=("Arial", 14))

lblZoneC = Label(root)
lblZoneC.grid(column=4, row = 0, columnspan=1)
lblZoneC.config(fg='cyan',font=("Arial", 14))

lblZoneD = Label(root)
lblZoneD.grid(column=5, row = 0, columnspan=1)
lblZoneD.config(fg='blue',font=("Arial", 14))

# Text Container Directions
lblLeft = Label(root)
lblLeft.grid(column=0, row = 1, columnspan=1)
lblLeft.config(fg="snow4",font=("Arial", 14))

lblRight = Label(root)
lblRight.grid(column=2, row = 1, columnspan=1)
lblRight.config(fg='deep sky blue',font=("Arial", 14))

lblUp = Label(root)
lblUp.grid(column= 2, row = 1, columnspan=1)
lblUp.config(fg='orange',font=("Arial", 14))

lblDown = Label(root)
lblDown.grid(column= 3, row = 1, columnspan=1)
lblDown.config(fg='gold',font=("Arial", 14))


# Video Container
lblVideo = Label(root)
lblVideo.grid(column = 0, row = 2, columnspan = 6)

# Mask Container
# lblMask = Label(root)
# lblMask.grid(column = 5, row = 1, columnspan = 5)

root.mainloop()