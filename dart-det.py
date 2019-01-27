import numpy as np
import cv2

cascade = cv2.CascadeClassifier('./dartcascade/cascade.xml')

img_names = ['dart0.jpg', 'dart1.jpg', 'dart2.jpg', 'dart3.jpg', 'dart4.jpg', 'dart5.jpg', 'dart6.jpg', 'dart7.jpg',
 'dart8.jpg', 'dart9.jpg', 'dart10.jpg', 'dart11.jpg', 'dart12.jpg', 'dart13.jpg', 'dart14.jpg', 'dart15.jpg']

ground_truth0 = [{'x1': 426, 'y1': 1, 'x2': 621, 'y2': 223}]
ground_truth1 = [{'x1': 167, 'y1': 105, 'x2': 417, 'y2': 354}]
ground_truth2 = [{'x1': 90, 'y1': 85, 'x2': 203, 'y2': 197}]
ground_truth3 = [{'x1': 312, 'y1': 138, 'x2': 398, 'y2': 229}]
ground_truth4 = [{'x1': 156, 'y1': 66, 'x2': 413, 'y2': 334}]
ground_truth5 = [{'x1': 414, 'y1': 125, 'x2': 557, 'y2': 261}]
ground_truth6 = [{'x1': 216, 'y1': 108, 'x2': 282, 'y2': 188}]
ground_truth7 = [{'x1': 235, 'y1': 151, 'x2': 412, 'y2': 336}]
ground_truth8 = [{'x1': 831, 'y1': 203, 'x2': 974, 'y2': 354}, {'x1': 65, 'y1': 241, 'x2': 134, 'y2': 355}]
ground_truth9 = [{'x1': 168, 'y1': 14, 'x2': 467, 'y2': 316}]
ground_truth10 = [{'x1': 76, 'y1': 88, 'x2': 200, 'y2': 229}, {'x1': 578, 'y1': 120, 'x2': 645, 'y2': 225}, {'x1': 913, 'y1': 143, 'x2': 954, 'y2': 222}]
ground_truth11 = [{'x1': 167, 'y1': 93, 'x2': 241, 'y2': 193}]
ground_truth12 = [{'x1': 152, 'y1': 59, 'x2': 223, 'y2': 235}]
ground_truth13 = [{'x1': 256, 'y1': 102, 'x2': 419, 'y2': 270}]
ground_truth14 = [{'x1': 102, 'y1': 85, 'x2': 265, 'y2': 247}, {'x1': 969, 'y1': 77, 'x2': 1130, 'y2': 242}]
ground_truth15 = [{'x1': 130, 'y1': 36, 'x2': 302, 'y2': 212}]



## create an array with all the ground truths
ground_truths = [ground_truth0, ground_truth1, ground_truth2, ground_truth3,
 ground_truth4, ground_truth5, ground_truth6, ground_truth7, ground_truth8,
  ground_truth9, ground_truth10, ground_truth11, ground_truth12, ground_truth13,
   ground_truth14, ground_truth15]

## from viewing the images, no false positives were detected
false_negatives = 0


def detectDartsAndGetScores(img_name, ground_truth, overlap_threshold):
        img = cv2.imread(img_name)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(grey, grey)

        ## scale factor used to create scale pyramid, reducing step size of 1%, incresing chance of matching model for detection
        ## but expensive. minNeighbours is set to 1, higher values means less detections but better quality, so with value of 1, 
        ## expect lots of matches of low quality
        darts = cascade.detectMultiScale( grey, 1.1, 1, flags=0 or cv2.CASCADE_SCALE_IMAGE, minSize=(50, 50), maxSize=(500,500) )
        # print('size of detected darts', len(darts))
        # overlapSum = 0

        true_pos_count = 0
        false_neg_count = 0
        skip_indices = []
        for j in range(len(ground_truth)):
                val = 0
                max_val = 0
                max_val_index = None
                for i in range(len(darts)):
                        if(i in skip_indices):
                                continue
                        face = darts[i]
                        x = face[0]
                        y = face[1]
                        w = face[2]
                        h = face[3]
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                        dartDict = {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h}
                        val = boundingBoxOverlap(dartDict, ground_truth[j])

                        ## if value exceeds threshold increment true positive count
                        if(val > max_val):
                                max_val+= val
                                max_val_index = i
                if (max_val > overlap_threshold and max_val_index != None):
                        ## if we find a suitable candidate for a dart detection, we increment tp and add the index so it can be
                        ## skipped in future iterations

                        true_pos_count+=1
                        skip_indices.append(max_val_index)
                if(not(max_val > overlap_threshold)):
                        false_neg_count+=1


        precision = true_pos_count / len(darts)
        if(precision == 0):
                recall  = 0
                f1 = 0
        else:
                recall = true_pos_count / (true_pos_count + false_neg_count)
                f1 = 2 * (precision * recall) / (precision + recall)

        
        return precision, recall, f1, img


def boundingBoxOverlap(bb1, bb2):
        ## bounding boxes are dict eg. bb1['x1'] = top left x, bb1['x2'] = bottom right x etc

        bb1Area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2Area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        ## check if bb1 contains bb2
        if(bb1['x1'] < bb2['x1'] < bb2['x2'] < bb1['x2'] and bb1['y1'] < bb2['y1'] < bb2['y2'] < bb1['y2']):
                return bb2Area / bb1Area


        ## check if bb2 contains bb1
        if((bb2['x1'] < bb1['x1'] < bb1['x2'] < bb1['x2']) and (bb2['y1'] < bb1['y1'] < bb1['y2'] < bb2['y2'])):
                return bb1Area / bb2Area

        ## get coords for intersection rectangle
        xLeft = max(bb1['x1'], bb2['x1'])
        xRight = min(bb1['x2'], bb2['x2'])
        yTop = max(bb1['y1'], bb2['y1'])
        yBottom = min(bb1['y2'], bb2['y2'])

        ## if there is no intersection we just return 0
        if xRight < xLeft or yBottom < yTop:
                return 0.0

        ## get the area of intersection
        intersectionArea = (xRight - xLeft) * (yBottom - yTop)

        ## percentage overlap for bounding box is calculated by intersection over union
        return intersectionArea / float(bb1Area + bb2Area - intersectionArea)

def  detectDarts(img_name): 
        img = cv2.imread(img_name)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(grey, grey)

        ## scale factor used to create scale pyramid, reducing step size of 1%, incresing chance of matching model for detection
        ## but expensive. minNeighbours is set to 1, higher values means less detections but better quality, so with value of 1, 
        ## expect lots of matches of low quality
        darts = cascade.detectMultiScale( grey, 1.1, 1, flags=0 or cv2.CASCADE_SCALE_IMAGE, minSize=(50, 50), maxSize=(500,500) )
        print('image ' + img_name + ' has ' + str(len(darts)) + 'darts')
        for (x,y,w,h) in darts:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        return img

scores = []
totalf1 = 0
for i in range(len(ground_truths)):
        precision, recall, f1, img = detectDartsAndGetScores(img_names[i], ground_truths[i], 0.3)
        totalf1+=f1
        scores.append({'image': img_names[i], 'precison': precision, 'recall': recall, 'f1': f1 })
        # cv2.imshow('img',img)
        # cv2.waitKey(0)


print(scores)
print('average f1 score', totalf1 / len(img_names))

cv2.destroyAllWindows()