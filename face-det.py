import numpy as np
import cv2

cascade = cv2.CascadeClassifier('frontal_face_cascade.xml')

img_names = ['dart4.jpg', 'dart5.jpg', 'dart13.jpg', 'dart14.jpg', 'dart15.jpg']

groundTruth4 = [{'x1': 356, 'y1': 111, 'x2': 471, 'y2': 270}]

groundTruth5 = [{'x1': 66, 'y1': 140, 'x2': 123, 'y2': 206},
                {'x1': 55, 'y1': 248, 'x2': 113, 'y2': 319},
                {'x1': 199, 'y1': 215, 'x2': 252, 'y2': 284},
                {'x1': 255, 'y1': 171, 'x2': 306, 'y2': 232},
                {'x1': 296, 'y1': 241, 'x2': 347, 'y2': 310},
                {'x1': 382, 'y1': 189, 'x2': 430, 'y2': 249},
                {'x1': 435, 'y1': 236, 'x2': 484, 'y2': 303},
                {'x1': 521, 'y1': 181, 'x2': 577, 'y2': 245},
                {'x1': 569, 'y1': 248, 'x2': 613, 'y2': 314},
                {'x1': 653, 'y1': 185, 'x2': 696, 'y2': 246},
                {'x1': 687, 'y1': 252, 'x2': 729, 'y2': 308}]

groundTruth13 = [{'x1': 429, 'y1': 126, 'x2': 552, 'y2': 260}]

groundTruth14 = [{'x1': 474, 'y1': 209, 'x2': 552, 'y2': 324},
                 {'x1': 734, 'y1': 188, 'x2': 824, 'y2': 296}]

groundTruth15 = [{'x1': 29, 'y1': 129, 'x2': 121, 'y2': 214},
                 {'x1': 375, 'y1': 110, 'x2': 453, 'y2': 195},
                 {'x1': 538, 'y1': 130, 'x2': 639, 'y2': 216}]

ground_truths = [groundTruth4, groundTruth5, groundTruth13, groundTruth14, groundTruth15]

false_negatives = [0,1,0,0,2]

def detectFacesAndPrintPercOverlap(img_name, ground_truth, overlap_threshold):
        img = cv2.imread(img_name)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(grey, grey)

        ## scale factor used to create scale pyramid, reducing step size of 1%, incresing chance of matching model for detection
        ## but expensive. minNeighbours is set to 1, higher values means less detections but better quality, so with value of 1, 
        ## expect lots of matches of low quality
        faces = cascade.detectMultiScale( grey, 1.1, 1, flags=0 or cv2.CASCADE_SCALE_IMAGE, minSize=(50, 50), maxSize=(500,500) )

        # print('size of detected faces', len(faces))
        true_pos_count = 0
        false_neg_count = 0
        skip_indices = []
        for j in range(len(ground_truth)):
                val = 0
                max_val = 0
                max_val_index = None
                for i in range(len(faces)):
                        if(i in skip_indices):
                                continue
                        face = faces[i]
                        x = face[0]
                        y = face[1]
                        w = face[2]
                        h = face[3]
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
                        faceDict = {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h}
                        val = boundingBoxOverlap(faceDict, ground_truth[j])

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


        precision = true_pos_count / len(faces)
        # print('precision', precision)
        recall = true_pos_count / (true_pos_count + false_neg_count)
        # print('recall', recall)
        f1 = 2 * (precision * recall) / (precision + recall)
        # print('f1', f1)
        return precision, recall, f1, img


def boundingBoxOverlap(bb1, bb2):
        ## bounding boxes are dict eg. bb1['x1'] = top left x, bb1['x2'] = bottom right x etc

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

        ## get area of both bounding boxes
        bb1Area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2Area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        ## percentage overlap for bounding box is calculated by intersection over union
        return intersectionArea / float(bb1Area + bb2Area - intersectionArea)

scores = []
f1sum = 0
for i in range(len(ground_truths)):
        precision, recall, f1, img = detectFacesAndPrintPercOverlap(img_names[i], ground_truths[i], 0.4)
        f1sum+= f1
        scores.append({'image': img_names[i], 'precision': precision, 'recall': recall, 'f1': f1})
        cv2.imshow('img', img)
        cv2.waitKey(0)

print(scores)
print('average f1', f1sum/len(img_names))
cv2.destroyAllWindows()