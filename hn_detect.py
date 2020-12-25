import numpy as np 
import pandas as pd
import os
import cv2
from sklearn.metrics import average_precision_score
from skimage.feature import hog
import pickle
from tqdm import tqdm

def sliding_window(im, step, winSize=(16,32)):
    """
    Perform sliding window on image
    @INPUT:
        - im: Image
        - step: Step in pixels between window
        - winSize: window size, tupple (width, height)
    @OUTPUT:
        - window: window of size winSize
    """
    # slide a window across the image
    for y in range(0, im.shape[0], step):
        for x in range(0, im.shape[1], step):
            # yield the current window
            yield (x, y, im[y:y + winSize[1], x:x + winSize[0]])

def pre_process(path, name, factor=1.3):
    """
    Preprocess image: RGB image --> Gray image --> Blur image (reduce noise) --> 
    High Pass Filter (sharpen) --> Constrast Limited Adaptive Histogram Equalization --> Exposure
    @INPUT:
        - path: image path
        - name: image name
        - factor: exposure factor
    @OUTPUT:
        - imgAHE: preprocessed image
    """
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], np.float32)
    
    img = cv2.imread(os.path.join(path, name))

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray image

    imgBlur = cv2.GaussianBlur(imgGray, (3,3), 0)   # reduce noise

    imgH = cv2.filter2D(imgBlur, -1, kernel)        # sharpen

    AHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))# Contrast Limited Adaptive Histogram Equalization 
    imgAHE = AHE.apply(imgH)
    imgAHE = np.array(imgAHE)

    img_processed = imgAHE*factor   # exposure

    return img_processed

def pyramid(im, scale=1.5, minSize=(64,128)):
    """
    Perform pyramid scaling
    @INPUT:
        - im: image need to perform pyramid scaling
        - scale: scaling factor
        - minSize: minimum size. minSize = (minW, minH)
    @OUTPUT:
        - im: scaled image
    """
    # First, return original image
    yield im
    
    # Perform pyramid scaling
    while True:
        # Calculate destination size
        w = np.int(im.shape[1]/scale)
        h = np.int(im.shape[0]/scale)
        
        # Resize image
        im = cv2.resize(im, (w, h))
        
        # Check if minimum size is sastified
        if (w < minSize[0]) or (h < minSize[1]):
            break
        
        yield im

def iou(a, b):
    """
    Calculate intersect area between two bounding boxes
    @INPUT:
        - a: tupple, list or array contain (xmin, xmax, ymin, ymax)
        - b: tupple, list or array contain (xmin, xmax, ymin, ymax)
    @OUTPUT:
        - p: Percentage of overlap area between two bounding boxes
    """
    # Get bounding box position
    axmin, axmax, aymin, aymax = a[0], a[1], a[2], a[3]
    bxmin, bxmax, bymin, bymax = b[0], b[1], b[2], b[3]
    
    # Calculate area of each bounding box
    a_area = (axmax - axmin)*(aymax - aymin)
    b_area = (bxmax - bxmin)*(bymax - bymin)
    
    # Calculate overlap area
    dx = np.min((axmax, bxmax)) - np.max((axmin, bxmin))
    dy = np.min((aymax, bymax)) - np.max((aymin, bymin))
    
    if (dx <= 0) or (dy <= 0):
        return 0
    else:
        return (dx*dy)/(a_area + b_area - dx*dy)

def soft_nms(boxes, score, threshold=0.5):
    """
    Perform Non-Maximal Suppression to remove overlap predicted bounding box
    @INPUT:
        - boxes: predicted bouding box
        - score: score of each box
        - iou_threshold: threshold to decide if two boxes are one
    @OUTPUT:
        - D: suppressed bounding box
        - S: score for each box
    """
    # Initialize output
    D = np.zeros(boxes.shape)
    S = np.zeros(score.shape)
    
    # Get number of boxes
    N = boxes.shape[0]
    
    # Soft-NMS
    for i in range(N):
        # Finding boxes with largest score
        index = np.argmax(score)
        
        # Add that box and score to output
        D[i,:] = boxes[index,:]
        S[i] = score[index]
        
        # Remove that box from boxes and score
        boxes = np.delete(boxes, index, axis=0)
        score = np.delete(score, index)
        
        # Re-calculate box score base on iou
        for j in range(boxes.shape[0]):
            iou_score = iou(D[i,:], boxes[j,:])
            score[j] *= 1-iou_score
    
    # Remove all box with score lower than threshold
    index = np.where(S < threshold)[0]
    D = np.delete(D, index, axis=0)
    S = np.delete(S, index)
    
    return D,S


# def hard_nms(boxes, overlapThresh=0.8):
#     	# if there are no boxes, return an empty list
# 	if len(boxes) == 0:
# 		return []
# 	# initialize the list of picked indexes
# 	pick = []
# 	# grab the coordinates of the bounding boxes
# 	x1 = boxes[:,0]
# 	x2 = boxes[:,1]
# 	y1 = boxes[:,2]
# 	y2 = boxes[:,3]
# 	# compute the area of the bounding boxes and sort the bounding
# 	# boxes by the bottom-right y-coordinate of the bounding box
# 	area = (x2 - x1 + 1) * (y2 - y1 + 1)
# 	idxs = np.argsort(y2)
#     # keep looping while some indexes still remain in the indexes
# 	# list
# 	while len(idxs) > 0:
# 		# grab the last index in the indexes list, add the index
# 		# value to the list of picked indexes, then initialize
# 		# the suppression list (i.e. indexes that will be deleted)
# 		# using the last index
# 		last = len(idxs) - 1
# 		i = idxs[last]
# 		pick.append(i)
# 		suppress = [last]
#         		# loop over all indexes in the indexes list
# 		for pos in range(0, last):
# 			# grab the current index
# 			j = idxs[pos]
# 			# find the largest (x, y) coordinates for the start of
# 			# the bounding box and the smallest (x, y) coordinates
# 			# for the end of the bounding box
# 			xx1 = max(x1[i], x1[j])
# 			yy1 = max(y1[i], y1[j])
# 			xx2 = min(x2[i], x2[j])
# 			yy2 = min(y2[i], y2[j])
# 			# compute the width and height of the bounding box
# 			w = max(0, xx2 - xx1 + 1)
# 			h = max(0, yy2 - yy1 + 1)
# 			# compute the ratio of overlap between the computed
# 			# bounding box and the bounding box in the area list
# 			overlap = float(w * h) / area[j]
# 			# if there is sufficient overlap, suppress the
# 			# current bounding box
# 			if overlap > overlapThresh:
# 				suppress.append(pos)
# 		# delete all indexes from the index list that are in the
# 		# suppression list
# 		idxs = np.delete(idxs, suppress)
# 	# return only the bounding boxes that were picked
#     print(boxes.shape)
# 	return boxes[pick]




def get_fp(im, clf, bbox, scale=1.5, winSize=(16,32), step=8, orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2), threshold=0.5):
    """
    Perform pyramid scaling and sliding window to find all false positive samples in an image.
    @INPUT:
        - im: image
        - clf: Classifier
        - bbox: ground truth bouding box: [xmin, xmax, ymin, ymax]
        - scale: scaling factor
        - winSize: window size: (winH, winW)
        - step: number of pixels between each window
        - orientations: number of histogram bin
        - pixels_per_cell: number of pixels in a cell
        - cells_per_block: number of cells in a block
    @OUTPUT:
        - fp: HOG feature of false positive window
        - proba: score of that false positive window
    """
    # Pyramid search
    ndigits = bbox.shape[0]
    k = -1
    for resized in pyramid(im, scale, winSize):
        k += 1
        for (x, y, window) in sliding_window(resized, step, winSize):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winSize[1] or window.shape[1] != winSize[0]:
                continue
                
            # Calculate HOG feature of window
            fd = hog(window, orientations=orientations, 
                     pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
            
            # Predict if window contains number or background
            fd = fd.reshape(1,-1)
            pred = clf.predict(fd)
            prob = clf.predict_proba(fd)
            
            # Check if this is a false positive
            if pred:
                # By default it's a false positive
                flag = True
                # Get window position
                left=np.int(x*scale**k)
                right = np.int((x+winSize[0]-1)*scale**k)
                top = np.int(y*scale**k)
                bottom = np.int((y+winSize[1]-1)*scale**k)
                
                pbox = np.array([left, right, top, bottom])
                pbox = pbox.astype(np.int)

                # Calculate intersect area between window and bounding box
                area_max = 0
                for digit in range(ndigits):
                    gbox = bbox[digit,:]
                    area = iou(gbox, pbox)
                    if area >= threshold:
                        flag = False
                        break
                if flag:
                    yield (fd, prob[0,1], window)

def get_predicted_bbx(path, name, clf, w=16, h=32, scale=2):
    """
    Get predicted bounding boxes for an image in order: [x_min, x_max, y_min, y_max]

    @INPUT:
    - path, name: identify the image
    - clf: Classifier to predict bounding boxes
    - w, h: width and height of the slicing window
    - scale: scale in pyramid algorithm
    @OUTPUT:
    - box: an array contains the bounding boxes coordinations, with every row as the coordinations of a box
    - score: confidence of the model related to predict bounding boxes
    """
    bbx, pred = [], []
    im = pre_process(path, name)

    for i, im_rsz in enumerate(pyramid(im, scale=scale, minSize=(16,32))):
        for (x, y, im_window) in sliding_window(im_rsz, step=8):
            if (im_window.shape[0]<h or im_window.shape[1]<w):
                continue
            feature = hog(im_window, orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2))
            feature = feature.reshape(1,-1)
            pred_proba = clf.predict_proba(feature)

            if pred_proba[0,1] > 0.8:
                x_min = np.int(x*(scale**i))
                x_max = np.int((x+w-1)*(scale**i))
                y_min = np.int(y*(scale**i))
                y_max = np.int((y+h-1)*(scale**i))
                coor = np.array([x_min, x_max, y_min, y_max])
                pred_proba = pred_proba[0,1]
                bbx.append(coor)
                pred.append(pred_proba)

    bbx = np.array(bbx)
    pred = np.array(pred)

    box, score = soft_nms(bbx, pred, threshold=0.5)
    
    return box, score

def get_ground_truth(path, name):

    """
    Get ground truth of an image
    
    @INPUT:
    - path: directory containing the image
    - name: the image's name 
    @OUTPUT:
    - bounding box (ground truth of the image)
    """

    labels = pd.read_csv('./label_no_png.csv')
    i_th = np.int(name.split('.')[0])
    
    x_min = labels[labels['name']==i_th]['left']
    x_max = x_min + labels[labels['name']==i_th]['width']
    y_min = labels[labels['name']==i_th]['top']
    y_max = y_min + labels[labels['name']==i_th]['height']

    bbx = np.vstack([x_min, x_max, y_min, y_max]).T

    return bbx

def scoring(clf, path, threshold):
    """
    Scoring the model
    @INPUT:
    - clf: classifier
    - path: the folder containing images to score
    - threshold: if iou<threshold: this bounding box is false positive
    @OUTPUT:
    - AP: average precision
    """
    names = os.listdir(path)
    y_test = np.array([])   # y_true
    SC = np.array([])       # score

    for name in names:
        GT = get_ground_truth(path, name)
        pred, score = get_predicted_bbx(path, name, clf)

        y_true = np.zeros(len(score))
        for i in range(GT.shape[0]):
            if pred.shape[0]:
                for j in range(pred.shape[0]):
                    if iou(GT[i,:], pred[j,:])>threshold:
                        y_true[j] = 1

        y_test = np.concatenate((y_test, y_true))
        SC = np.concatenate((SC, score))

    AP = average_precision_score(y_test, SC)

    return AP

def inference(path, new_path, clf):
    if not(os.path.exists(new_path)):
        os.mkdir(new_path)

    names = os.listdir(path)

    for name in names:
        box, score = get_predicted_bbx(path, name, clf, w=16, h=32, scale=2)
        box = box.astype(int)
        img = cv2.imread(os.path.join(path, name))
        for i in range(score.shape[0]):
            cv2.rectangle(img, (box[i,0], box[i,2]), (box[i,1], box[i,3]), (20, 200, 200), 1)
        cv2.imwrite(os.path.join(new_path, name), img)

    return None
