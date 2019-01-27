import numpy as np
import cv2


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

def detectDarts(original_img, grey, matches, match_threshold):
        cv2.equalizeHist(grey, grey)

        ## scale factor used to create scale pyramid, reducing step size of 1%, incresing chance of matching model for detection
        ## but expensive. minNeighbours is set to 1, higher values means less detections but better quality, so with value of 1, 
        ## expect lots of matches of low quality
        darts = cascade.detectMultiScale( grey, 1.1, 2, flags=0 or cv2.CASCADE_SCALE_IMAGE, minSize=(50, 50), maxSize=(500,500) )

        for (x,y,w,h) in darts:
                if(checkForPointMatches(matches, x, w, y, h, match_threshold)):
                        cv2.rectangle(original_img,(x,y),(x+w,y+h),(255,0,255),2)
        return original_img

def checkForPointMatches(matches, x, w, y, h, match_threshold):
    matchCount = 0
    for x2, y2 in matches:
        if((x < x2 < x + w) and (y < y2 < y + h)):
            matchCount += 1
    if(matchCount >= match_threshold): 
        return True
    return False




def detectDartsAndGetScores(original_img, grey, matches, match_threshold, ground_truth, overlap_threshold):
        cv2.equalizeHist(grey, grey)

        ## scale factor used to create scale pyramid, reducing step size of 1%, incresing chance of matching model for detection
        ## but expensive. minNeighbours is set to 1, higher values means less detections but better quality, so with value of 1, 
        ## expect lots of matches of low quality
        darts = cascade.detectMultiScale( grey, 1.1, 1, flags=0 or cv2.CASCADE_SCALE_IMAGE, minSize=(50, 50), maxSize=(500,500) )


        rectangle_count = 0
        true_pos_count = 0
        false_neg_count = 0
        skip_indices = []
        for j in range(len(ground_truth)):
                val = 0
                max_val = 0
                max_val_index = None
                bb = ground_truth[j]
                gx1 = bb['x1']
                gx2 = bb['x2']
                gy1 = bb['y1']
                gy2 = bb['y2']
                # cv2.rectangle(original_img,(gx1,gy1),(gx2,gy2),(255,0, 255),2)
                for i in range(len(darts)):
                        if(i in skip_indices):
                                continue
                        dart = darts[i]
                        x = dart[0]
                        y = dart[1]
                        w = dart[2]
                        h = dart[3]
                        if(checkForPointMatches(matches, x, w, y, h, match_threshold)):
                                rectangle_count+=1
                                cv2.rectangle(original_img,(x,y),(x+w,y+h),(255,0,0),2)
                                

                                dartDict = {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h}
                                val = boundingBoxOverlap(dartDict, ground_truth[j])

                                ## if value exceeds threshold increment true positive count
                                if(val > max_val):
                                        max_val+= val
                                        max_val_index = i
                # cv2.imshow('img',original_img)
                # cv2.waitKey(0)
                if (max_val > overlap_threshold and max_val_index != None):
                        ## if we find a suitable candidate for a dart detection, we increment tp and add the index so it can be
                        ## skipped in future iterations

                        true_pos_count+=1
                        skip_indices.append(max_val_index)
                if(not(max_val > overlap_threshold)):
                        false_neg_count+=1


        precision = true_pos_count / rectangle_count
        recall = true_pos_count / (true_pos_count + false_neg_count)
        f1 = 2 * (precision * recall) / (precision + recall)
        return original_img, precision, recall, f1

cascade = cv2.CascadeClassifier('./dartcascade/cascade.xml')
img_name = 'dart15.jpg'
original_img = cv2.imread(img_name)
img1 = cv2.imread("dart.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
 
# Brute Force Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

## sort matches by distance
matches = sorted(matches, key = lambda x:x.distance)
good = matches
positions = []

# we get the x and y coord for each match in the train image (the image we are detecting darts in)
for mat in good:

    # Get the matching keypoints for each of the images
    img2_idx = mat.trainIdx

    x2 = int(kp2[img2_idx].pt[0])
    y2 = int(kp2[img2_idx].pt[1])
    positions.append((x2, y2))

matching_result = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)


# img = detectDarts(original_img, img2, positions, 30)
img, precision, recall, f1 = detectDartsAndGetScores(original_img, img2, positions, 130, ground_truth15, 0.2)

print('precision', precision)
print('recall', recall)
print('f1', f1)
 
cv2.imshow("Img1", img1)
cv2.imshow("Img2", img2)
cv2.imshow("Matching result", matching_result)
cv2.imshow('detectedDartboards', img)
cv2.waitKey(0)
cv2.destroyAllWindows()