# -*- coding:utf-8 -*-
# original author: DuanshengLiu
import cv2
import numpy as np


def locate_and_correct(img_src, img_mask):
    """
    This function performs edge detection on img_mask through cv2, and obtains the edge coordinates of the license plate area (stored in contours) and the four endpoint coordinates of the smallest circumscribed rectangle.
     From the edge coordinates of the license plate, the points nearest to the four endpoints of the smallest circumscribed rectangle are calculated as the four endpoints of the parallelogram license plate, so as to realize the positioning and correction of the license plate
     :param img_src: original image
     :param img_mask: The binary image obtained by image segmentation through u_net, the license plate area is white, and the background area is black
     :return: The located and corrected license plate
    """
    # cv2.imshow('img_mask',img_mask)
    # cv2.waitKey(0)
    # ret,thresh = cv2.threshold(img_mask[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Binarization
    # cv2.imshow('thresh',thresh)
    # cv2.waitKey(0)
    try:
        contours, hierarchy = cv2.findContours(img_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:  # Prevent inconsistent opencv versions from reporting errors
        ret, contours, hierarchy = cv2.findContours(img_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):  # The length of contours1 is 0, indicating that no license plate is detected
        # print("license plate not detected")
        return [], []
    else:
        Lic_img = []
        img_src_copy = img_src.copy()  # img_src_copy is used to draw the outline of the license plate
        for ii, cont in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cont)  # Get the smallest bounding rectangle
            img_cut_mask = img_mask[y:y + h, x:x + w]  # Cut out the license plate area of the label
            # cv2.imshow('img_cut_mask',img_cut_mask)
            # cv2.waitKey(0)
            # print('w,h,mean, aspect ratio',w,h,np.mean(img_cut_mask),w/h)
            # In addition to the license plate area in the contours, there may be small noises with a width or height of 1 or 2,
            # The average value of the license plate area to be selected should be high, and the width and height should not be very small, so filter by the following conditions
            if np.mean(img_cut_mask) >= 75 and w > 15 and h > 15:
                rect = cv2.minAreaRect(cont)  # Obtain the minimum circumscribed rectangle with orientation angle, center point coordinates, width and height, and rotation angle for the coordinate point
                box = cv2.boxPoints(rect).astype(np.int32)  # Get the coordinates of the four vertices of the smallest circumscribed rectangle
                # cv2.drawContours(img_mask, contours, -1, (0, 0, 255), 2)
                # cv2.drawContours(img_mask, [box], 0, (0, 255, 0), 2)
                # cv2.imshow('img_mask',img_mask)
                # cv2.waitKey(0)
                cont = cont.reshape(-1, 2).tolist()
                # Since the two sets of coordinate positions of the transformation matrix need to be in one-to-one correspondence, the coordinates of the smallest circumscribed rectangle need to be sorted, and the final sorting is [upper left, lower left, upper right, lower right]
                box = sorted(box, key=lambda xy: xy[0])  # First sort according to left and right, divided into coordinates on the left and coordinates on the right
                box_left, box_right = box[:2], box[2:]  # At this time, the first two of the box are the coordinates on the left, and the last two are the coordinates on the right.
                box_left = sorted(box_left, key=lambda x: x[1])  # Then sort according to the top and bottom, that is, y. At this time, box_left is the upper left and lower left endpoint coordinates
                box_right = sorted(box_right, key=lambda x: x[1])  # At this time, the coordinates of the upper right and lower right endpoints in box_right
                box = np.array(box_left + box_right)  # [upper left, lower left, upper right, lower right]
                # print(box)
                x0, y0 = box[0][0], box[0][1]  # The four coordinates here are the four coordinates of the smallest circumscribed rectangle, and then the coordinates of the parallel (or irregular) quadrilateral need to be obtained
                x1, y1 = box[1][0], box[1][1]
                x2, y2 = box[2][0], box[2][1]
                x3, y3 = box[3][0], box[3][1]

                def point_to_line_distance(X, Y):
                    if x2 - x0:
                        k_up = (y2 - y0) / (x2 - x0)  # slope is not infinite
                        d_up = abs(k_up * X - Y + y2 - k_up * x2) / (k_up ** 2 + 1) ** 0.5
                    else:  # infinite slope
                        d_up = abs(X - x2)
                    if x1 - x3:
                        k_down = (y1 - y3) / (x1 - x3)  # slope is not infinite
                        d_down = abs(k_down * X - Y + y1 - k_down * x1) / (k_down ** 2 + 1) ** 0.5
                    else:  # infinite slope
                        d_down = abs(X - x1)
                    return d_up, d_down

                d0, d1, d2, d3 = np.inf, np.inf, np.inf, np.inf
                l0, l1, l2, l3 = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
                for each in cont:  # Calculate the distance between the coordinates in cont and the four coordinates of the rectangle and the distance to the upper and lower two straight lines, add the weight to the distance sum, and successfully calculate the coordinates of the four vertices of the selected quadrilateral
                    x, y = each[0], each[1]
                    dis0 = (x - x0) ** 2 + (y - y0) ** 2
                    dis1 = (x - x1) ** 2 + (y - y1) ** 2
                    dis2 = (x - x2) ** 2 + (y - y2) ** 2
                    dis3 = (x - x3) ** 2 + (y - y3) ** 2
                    d_up, d_down = point_to_line_distance(x, y)
                    weight = 0.975
                    if weight * d_up + (1 - weight) * dis0 < d0:  # update if less than
                        d0 = weight * d_up + (1 - weight) * dis0
                        l0 = (x, y)
                    if weight * d_down + (1 - weight) * dis1 < d1:
                        d1 = weight * d_down + (1 - weight) * dis1
                        l1 = (x, y)
                    if weight * d_up + (1 - weight) * dis2 < d2:
                        d2 = weight * d_up + (1 - weight) * dis2
                        l2 = (x, y)
                    if weight * d_down + (1 - weight) * dis3 < d3:
                        d3 = weight * d_down + (1 - weight) * dis3
                        l3 = (x, y)

                # print([l0,l1,l2,l3])
                # for l in [l0, l1, l2, l3]:
                #     cv2.circle(img=img_mask, color=(0, 255, 255), center=tuple(l), thickness=2, radius=2)
                # cv2.imshow('img_mask',img_mask)
                # cv2.waitKey(0)
                p0 = np.float32([l0, l1, l2, l3])  # The upper left corner, the lower left corner, the upper right corner, the lower right corner, the coordinates in p0 and p1 correspond in order to form the transformation matrix
                p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])  # The rectangle we need
                transform_mat = cv2.getPerspectiveTransform(p0, p1)  # Constitute the transformation matrix
                lic = cv2.warpPerspective(img_src, transform_mat, (240, 80))  # Carry out license plate correction
                # cv2.imshow('lic',lic)
                # cv2.waitKey(0)
                Lic_img.append(lic)
                cv2.drawContours(img_src_copy, [np.array([l0, l1, l3, l2])], -1, (0, 255, 0), 2)  # Draw the outline of the license plate on img_src_copy, (0, 255, 0) means that the drawn line is green
    return img_src_copy, Lic_img
