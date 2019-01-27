import numpy as np
import cv2
import matplotlib.pyplot as plt

cascade = cv2.CascadeClassifier('./dartcascade/cascade.xml')

def sobel(image, sobelx, sobely, threshold):
        gradient_x = np.zeros(image.shape)
        gradient_y = np.zeros(image.shape)
        gradient_mag = np.zeros(image.shape)
        gradient_dir = np.zeros(image.shape)
        for row in range(1, image.shape[0] -1):
                for column in range(1, image.shape[1]-1):
                        vecinos = [[image[row-1][column-1],image[row-1][column],image[row-1][column+1]],[image[row][column-1],image[row][column], image[row][column+1]], [image[row+1][column-1], image[row+1][column], image[row+1][column+1]]]
                        gradient_x[row][column] = np.sum(np.multiply(sobel_kernel_x, vecinos))
                        gradient_y[row][column] = np.sum(np.multiply(sobel_kernel_y, vecinos))
                        gradient_mag[row][column] = (gradient_x[row][column]**2 + gradient_y[row][column]**2)**0.5
                        gradient_dir[row][column] = np.arctan2(gradient_y[row][column], gradient_x[row][column])
                        if(gradient_mag[row][column] >= threshold):
                                gradient_mag[row][column] = 255
        ## replace nan to zeros
        gradient_dir = np.nan_to_num(gradient_dir)
        

        return gradient_mag, gradient_dir

def houghCircle(grad_mag, grad_dir, rmin, rmax, hough_thresh):
        circles = []
        rows = grad_mag.shape[0]
        columns = grad_mag.shape[1]

        ## create accumulator
        A = np.zeros((rows, columns, rmax-rmin))

        for row in range(rows):
                for column in range(columns):
                        ## when we find an edge we draw circles for each radius in the range
                        if(grad_mag[row][column] == 255):
                                for r in range(rmax-rmin):
                                        ## we do this for the negative values as well as positive
                                        for sign in [1, -1]:
                                                a = int(row + sign * (r+rmin) * np.sin(grad_dir[row][column]))
                                                b = int(column + sign * (r+rmin) * np.cos(grad_dir[row][column]))

                                                ## make sure it is within bounds
                                                if ((a > 0 and a < rows) and (b > 0 and b < columns)):
                                                        A[a][b][r]+= 1

        ## now that we have the accumulator, we then find the positions of the circles given that there are sufficient votes
        for row in range(rows):
                for column in range(columns):
                        for r in range(rmax-rmin):
                                if(A[row][column][r] >= hough_thresh):
                                        circles.append([row, column, r+rmin, A[row][column][r]])


        return A, circles

def detectDarts(original_img, grey, circles):
        cv2.equalizeHist(grey, grey)

        ## scale factor used to create scale pyramid, reducing step size of 1%, incresing chance of matching model for detection
        ## but expensive. minNeighbours is set to 1, higher values means less detections but better quality, so with value of 1, 
        ## expect lots of matches of low quality
        darts = cascade.detectMultiScale( grey, 1.1, 1, flags=0 or cv2.CASCADE_SCALE_IMAGE, minSize=(50, 50), maxSize=(500,500) )

        for (x,y,w,h) in darts:
                if(checkForCircles(circles, x, w, y, h)):
                        cv2.rectangle(original_img,(x,y),(x+w,y+h),(255,0,255),2)

        return original_img

def checkForCircles(circles, x, w, y, h):
        for c in circles:
                if((c[0] >= y and c[0] <= y + h) and (c[1] >= x and c[1] <= x + w)):
                        return True
        return False




def detectDartsAndGetScores(original_img, grey, circles, ground_truth, overlap_threshold):
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

                for i in range(len(darts)):
                        if(i in skip_indices):
                                continue
                        dart = darts[i]
                        x = dart[0]
                        y = dart[1]
                        w = dart[2]
                        h = dart[3]

                        if(checkForCircles(circles, x, w, y, h)):
                                rectangle_count+=1
                                cv2.rectangle(original_img,(x,y),(x+w,y+h),(255,0,0),2)
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


        precision = true_pos_count / rectangle_count
        # print('precision', precision)
        recall = true_pos_count / (true_pos_count + false_neg_count)
        # print('recall', recall)
        f1 = 2 * (precision * recall) / (precision + recall)
        # print('f1', f1)
        return original_img, precision, recall, f1




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



sobel_kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

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


original_img = cv2.imread('dart9.jpg')
grey = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
img_bin = cv2.threshold(grey, 55, 255, cv2.THRESH_BINARY)[1]
grad_mag, grad_dir = sobel(img_bin, sobel_kernel_x, sobel_kernel_y, 500)

hough_acc, circles = houghCircle(grad_mag, grad_dir, 10, 100, 6)

# for circle in circles:
#         cv2.circle(original_img,(circle[1], circle[0]), circle[2], (0,0,255), 1)

# img = detectDarts(original_img, grey, circles)
img, precision, recall, f1 = detectDartsAndGetScores(original_img, grey, circles, ground_truth4, 0.2)

print('precision', precision)
print('recall', recall)
print('f1', f1)

## show magnitude image as well as result image
cv2.imshow('grad mag', grad_mag)
cv2.imshow('img', img)
cv2.waitKey(0)

## show centres with matplotlib because it looks nice and pretty
centres = np.sum(hough_acc, axis=2)
fig3,ax3 = plt.subplots()
ax3.set_xlim((0, grey.shape[1]))
ax3.set_ylim((0, grey.shape[0]))
ax3.imshow(centres)
plt.gca().invert_yaxis()

plt.show()