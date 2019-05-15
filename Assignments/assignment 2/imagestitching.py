from __future__ import print_function  #
import cv2
import argparse
import os
import numpy
import numpy as np
import random

def up_to_step_1(imgs):
    """Complete pipeline up to step 3: Detecting features and descriptors"""
    
    imgs1 = []
    for i in range(len(imgs)):
        
        surf = cv2.xfeatures2d.SURF_create(3000)
        kp, des = surf.detectAndCompute(imgs[i], None)
        img = cv2.drawKeypoints(imgs[i], kp, imgs[i], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imgs1.append(img)
    
    return imgs1


def save_step_1(imgs, output_path='./output/step1'):
    """Save the intermediate result from Step 1"""
    

    for n, img in enumerate(imgs, start=1):
        m = '0' + str(n) if n < 10 else str(n)
        filename = 'img' + m + '.jpg'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, filename), img)
    
    pass

''' this is the helper function for 2nd part'''
def draw_matches(window_name, kp_pairs, img1, img2):

    mkp1, mkp2 = zip(*kp_pairs)

    p1 = numpy.float32([kp.pt for kp in mkp1])
    p2 = numpy.float32([kp.pt for kp in mkp2])

    if len(kp_pairs) >= 4:
        H, status = None, None
        
    else:
        H, status = None, None

    if len(p1):
        vis = explore_match(window_name, img1, img2, kp_pairs, status, H)
    return vis


''' this is the helper function for 2nd part'''
def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = numpy.zeros((max(h1, h2), w1 + w2), numpy.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = numpy.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = numpy.ones(len(kp_pairs), numpy.bool_)
    p1 = numpy.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = numpy.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)
    
    return vis
    



def up_to_step_2(imgs):
    """Complete pipeline up to step 2: Calculate matching feature points"""
    
    imagename=[]
    index=[]
    imgs2=[]
    for n in range(len(imgs)):
        
        for m in range(len(imgs)):
            if n==m or [n,m] in index or [m,n] in index:
                continue
            surf = cv2.xfeatures2d.SURF_create(3000)
            kp1, des1 = surf.detectAndCompute(imgs[n],None)
            kp2, des2 = surf.detectAndCompute(imgs[m],None)
            bestj = []
            best_ones=[]
            first_match=[]
            best_matches=[]
            keysorted=[]
            for k in range(0, len(des1)):
                a = des1[k]
                maxdist = 10000
                jvalue = 0
                for l in range(0, len(des2)):
                    b = des2[l]
                    dist = numpy.linalg.norm(a-b)
                    first_match.append([k,l,dist])
                keysorted=list(sorted(first_match,key = lambda x:x[2]))
                bestj.append([kp1[keysorted[0][0]],kp2[keysorted[0][1]]])
                best_ones.append([cv2.DMatch(keysorted[0][0],keysorted[0][1],keysorted[0][2]),cv2.DMatch(keysorted[1][0],keysorted[1][1],keysorted[1][2])])
                first_match=[]
                keysorted=[]
            for o,p in best_ones:
                if o.distance < 0.80*p.distance:
                    best_matches.append([o])
            if len(best_matches)<10:
                print("Anything is not in common")
            matched_img = cv2.drawMatchesKnn(imgs[n],kp1,imgs[m],kp2,best_matches,None,flags=2)
            imgs2.append(matched_img)
            imagename.append([n,m,len(kp1),len(kp2),len(best_matches)])
            index.append([n,m])
    return imgs2, imagename


def save_step_2(imgs, match_list, output_path="./output/step2"):
    """Save the intermediate result from Step 2"""
    # ... your code here ...

    filename=[]
    for i in os.listdir(args.input):
        filename.append(i)
    for n,i in enumerate(imgs):
        filength=output_path+'/'+str(filename[match_list[n][0]])+'_'+str(match_list[n][2])+'_'+str(filename[match_list[n][1]])+'_'+str(match_list[n][3])+'_'+str(match_list[n][4])+'.jpg'
        cv2.imwrite(filength,i)
    pass

def up_to_step_3(imgs):
    """Complete pipeline up to step 3: estimating homographies and warpings"""
    # ... your code here ...

    
    homographyList = []
    filenum=[]
    comman=[]
    imgs2=[]
    image_warping=[]
    for n,i in enumerate(imgs):
        surf = cv2.xfeatures2d.SURF_create(3000)
        kp1, des1 = surf.detectAndCompute(i,None)
        for m,j in enumerate(imgs):
            if n==m or [n,m] in comman or [m,n] in comman:
                continue
            
            kp2, des2 = surf.detectAndCompute(j,None)
            best_ones=[]
            first_match=[]
            best_matches=[]
            match_sorted=[]
            for k in range(0, len(des1)):
                a = des1[k]
                maxdist = 10000
                jvalue = 0
                for l in range(0, len(des2)):
                    b = des2[l]
                    dist = numpy.linalg.norm(a-b)
                    first_match.append([k,l,dist])
                match_sorted=list(sorted(first_match,key = lambda x:x[2]))
                best_ones.append([cv2.DMatch(match_sorted[0][0],match_sorted[0][1],match_sorted[0][2]),cv2.DMatch(match_sorted[1][0],match_sorted[1][1],match_sorted[1][2])])
                first_match=[]
                match_sorted=[]
            for o,p in best_ones:
                if o.distance < 0.80*p.distance:
                    best_matches.append([o])
            if len(best_matches)<10:
                print("Anything is not in common")
            for first_match in best_matches:
                (x1, y1) = kp1[first_match[0].queryIdx].pt
                (x2, y2) = kp2[first_match[0].trainIdx].pt
                homographyList.append([x1, y1, x2, y2])
            homo = np.matrix(homographyList)
            finalH, inliers = ransac(homo, 0.60)
            #print(finalH)
            filenum.append([n,m,len(kp1),len(kp2),len(best_matches)])
            comman.append([n,m])
            source,destination=warpAffinePadded(i,j,finalH[:2])
            image_warping.append([source,destination])
    return image_warping, comman

''' helper function for part 3'''
def ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(1000):
        
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))

        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))

        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers


''' helper function for part 3'''
def calculateHomography(correspondences):
    
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    
    u, s, v = np.linalg.svd(matrixA)

    
    h = np.reshape(v[8], (3, 3))

    
    h = (1/h.item(8)) * h
    return h

''' helper function for part 3'''
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


''' helper function for part 3'''
def warpAffinePadded(
        source, destination, M,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0):

    assert M.shape == (2, 3), \
        'Affine transformation shape should be (2, 3).\n'

    if flags in (cv2.WARP_INVERSE_MAP,
                 cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                 cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP):
        M = cv2.invertAffineTransform(M)
        flags -= cv2.WARP_INVERSE_MAP

    
    src_h, src_w = source.shape[:2]
    lin_pts = np.array([
        [0, src_w, src_w, 0],
        [0, 0, src_h, src_h]])

    
    transf_lin_pts = M[:, :2].dot(lin_pts) + M[:, 2].reshape(2, 1)

    
    min_x = np.floor(np.min(transf_lin_pts[0])).astype(int)
    min_y = np.floor(np.min(transf_lin_pts[1])).astype(int)
    max_x = np.ceil(np.max(transf_lin_pts[0])).astype(int)
    max_y = np.ceil(np.max(transf_lin_pts[1])).astype(int)
    
    anchor_x, anchor_y = 0, 0
    if min_x < 0:
        anchor_x = -min_x
    if min_y < 0:
        anchor_y = -min_y
    shifted_transf = M + [[0, 0, anchor_x], [0, 0, anchor_y]]

    
    dst_h, dst_w = destination.shape[:2]

    pad_widths = [anchor_y, max(max_y, dst_h) - dst_h,
                  anchor_x, max(max_x, dst_w) - dst_w]

    dst_padded = cv2.copyMakeBorder(destination, *pad_widths,
                                    borderType=borderMode, value=borderValue)

    dst_pad_h, dst_pad_w = dst_padded.shape[:2]
    src_warped = cv2.warpAffine(
        source, shifted_transf, (dst_pad_w, dst_pad_h),
        flags=flags, borderMode=borderMode, borderValue=borderValue)

    return dst_padded, src_warped



def save_step_3(img_pairs, match_list, output_path="./output/step3"):
    """Save the intermediate result from Step 3"""
    

    filename, images = [], []
    
    for k in os.listdir(args.input):
        filename.append(k)
    for n in range(len(img_pairs)):
        for m in range(len(img_pairs[n])):
            if m==0:
                filen=output_path+'/'+str(filename[match_list[n][0]])+'_'+str(filename[match_list[n][1]])+'.jpg'
            else:
                filen=output_path+'/'+str(filename[match_list[n][1]])+'_'+str(filename[match_list[n][0]])+'.jpg'
            cv2.imwrite(filen,img_pairs[n][m])
            images.append(img_pairs[n][m])
    
    pass


def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    
    return output_img


def up_to_step_4(imgs):
    """Complete the pipeline and generate a panoramic image"""
    # ... your code here ...

    res=[]
    for i in range(len(imgs)-1):
        if i == 0:
            a,b=imgs[0],imgs[1]
            sift = cv2.xfeatures2d.SURF_create()
            # Extract the keypoints and descriptors
            keypoints1, descriptors1 = sift.detectAndCompute(imgs[0], None)
            keypoints2, descriptors2 = sift.detectAndCompute(imgs[1], None)
        elif i == 1 and len(imgs) < 4:
            a,b=result,imgs[2]
            sift = cv2.xfeatures2d.SURF_create()
            # Extract the keypoints and descriptors
            keypoints1, descriptors1 = sift.detectAndCompute(result, None)
            keypoints2, descriptors2 = sift.detectAndCompute(imgs[2], None)
        elif i==1 :
            a,b=result,imgs[2]
            sift = cv2.xfeatures2d.SURF_create()
            # Extract the keypoints and descriptors
            keypoints1, descriptors1 = sift.detectAndCompute(imgs[2], None)
            keypoints2, descriptors2 = sift.detectAndCompute(imgs[3], None)
        
        if i==2:
            a,b=res[0],res[1]
            sift = cv2.xfeatures2d.SURF_create()
            # Extract the keypoints and descriptors
            keypoints1, descriptors1 = sift.detectAndCompute(res[0], None)
            keypoints2, descriptors2 = sift.detectAndCompute(res[1], None)

        # Initialize parameters for Flann based matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

    # Initialize the Flann based matcher object
        flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Compute the matches
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Store all the good matches as per Lowe's ratio test
        good_matches = []
        for m1,m2 in matches:
            if m1.distance < 0.7*m2.distance:
                good_matches.append(m1)
        if len(good_matches) > 10:
            src_pts = np.float32([ keypoints1[good_match.queryIdx].pt for good_match in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints2[good_match.trainIdx].pt for good_match in good_matches ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            result = warpImages(a, b, M)
            res.append(result)
            #cv2.imwrite('Stitched.jpg', result)
        
    
    return result


def save_step_4(imgs, output_path="./output/step4"):
    """Save the intermediate result from Step 4"""
    # ... your code here ...

    filen = output_path+'/stitched.jpg'
    cv2.imwrite(filen, imgs)
    
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        help="compute image stitching pipeline up to this step",
        type=int
    )

    parser.add_argument(
        "input",
        help="a folder to read in the input images",
        type=str
    )

    parser.add_argument(
        "output",
        help="a folder to save the outputs",
        type=str
    )

    args = parser.parse_args()

    imgs = []
    for filename in os.listdir(args.input):
        print(filename)
        img = cv2.imread(os.path.join(args.input, filename))
        imgs.append(img)

    if args.step == 1:
        print("Running step 1")
        modified_imgs = up_to_step_1(imgs)
        save_step_1(modified_imgs, args.output)
    elif args.step == 2:
        print("Running step 2")
        modified_imgs, match_list = up_to_step_2(imgs)
        save_step_2(modified_imgs, match_list, args.output)
    elif args.step == 3:
        print("Running step 3")
        img_pairs,comman = up_to_step_3(imgs)
        save_step_3(img_pairs, comman, args.output)
    elif args.step == 4:
        print("Running step 4")
        print("Please give at least 4 images :)")
        panoramic_img = up_to_step_4(imgs)
        save_step_4(panoramic_img, args.output)
