from __future__ import division
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob
import cv2
import math
from numpy import array
from sklearn.cluster import KMeans
from io import StringIO
from PIL import Image
import math
import matplotlib.pyplot as plt
import time
from utils import visualization_utils as vis_util
from utils import label_map_util
from sklearn.feature_extraction import image
from sklearn.cluster import DBSCAN
from multiprocessing.dummy import Pool as ThreadPool
from matplotlib import pyplot as plt
MAX_NUMBER_OF_BOXES = 10
MINIMUM_CONFIDENCE =0.9

PATH_TO_LABELS = 'detection3/label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'test_images'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'output5'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
def hull(ymin,xmin,ymax,xmax,image):
    img=image[ymin:ymax,xmin:xmax]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blur = cv2.blur(gray, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    temp=[]
    hull=[]
    for i in range(len(contours)):
    # creating convex hull object for each contour
        hull1 = cv2.convexHull(contours[i])
        temp.append(len(cv2.convexHull(contours[i], False).tolist()))
        #cv2.drawContours(img, [hull1], -1, (0, 0, 255), 1)
    for j in temp:
        hull.append(j)
    #print(hull)
    print(hull)
    #cv2.imwrite("pool_trial4_hull/contours"+str(count)+'-'+str(count1)+'.jpg', img)
    return float(sum(hull)/len(hull))
def remove_sea(ymin,xmin,ymax,xmax,image):
    img=image[ymin:ymax,xmin:xmax]
    b,g,r=cv2.split(img)
    sum=0
    sum1=0
    for i in g:
        for j in i:
            sum=sum+j
    for k in r:
        for l in k:
            sum1=sum1+l
    #print([sum,sum1])
    return [sum,sum1]
def get_hist(ymin,xmin,ymax,xmax,image):
    img=image[ymin:ymax,xmin:xmax]
    img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    maxl=max(hist[0])
    sum=0
    for j in hist:
        for i in j:
            sum=sum+i
    print(sum)
    #plt.figure()
    #plt.title("Grayscale Histogram")
    #plt.xlabel("Bins")
    #plt.ylabel("# of Pixels")
    #plt.plot(hist)
    #plt.xlim([0, 256])
    #plt.show()
    return float(sum/((ymax-ymin)*(xmax-xmin)))
def get_average_color(ymin,xmin,ymax,xmax,image):
    image=image[ymin:ymax,xmin:xmax]
    #image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    #color=cv2.mean(image)
    image=np.array(image)
    color=[]
    crop=np.array_split(image,4)
    for i in range(4):
        color.append(np.mean(crop[i]))
    #print(color)
    return color
def get_moment(ymin,xmin,ymax,xmax,image):
    image=image[ymin:ymax,xmin:xmax]
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(image)).flatten()
def remove_inf(ls):
    for x in range(len(ls)):
        for y in range(len(ls[x])):
            for z in range(len(ls[x][y])):
                if ls[x][y][z]==float('-inf'):
                    ls[x][y][z]=0
    return ls
def get_amp(ymin,xmin,ymax,xmax,image):
    image=image[ymin:ymax,xmin:xmax]
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum =remove_inf(20*np.log(np.abs(fshift)))
    #plt.imshow(20*np.log(np.abs(fshift))[1])
    #plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #plt.savefig('pool_trial3_fft.jpg')
    sum=0
    for x in range(len(magnitude_spectrum)):
        for y in range(len(magnitude_spectrum[x])):
            sum=sum+magnitude_spectrum[x][y]
    sum[0]=float(sum[0]/float(((ymax-ymin)*(xmax-xmin))))
    sum[1]=float(sum[1]/float(((ymax-ymin)*(xmax-xmin))))
    sum[2]=float(sum[2]/float(((ymax-ymin)*(xmax-xmin))))
    return sum
def matcher_count(ymin,xmin,ymax,xmax,image):
    orb=cv2.ORB()
    orb=orb.create()
    image=image[ymin:ymax,xmin:xmax]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    kp,des=orb.detectAndCompute(gray,None)
    sum=0
    count=0
    return len(kp)
def orb_matcher(ymin,xmin,ymax,xmax,image):
    orb = cv2.ORB()
    orb = cv2.ORB_create()
    image3=image[ymin:ymax,xmin:xmax]
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(image3,None)
    kp2, des2 = orb.detectAndCompute(image,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    sum=0
    for m in matches:
        sum=sum+m.distance
    return len(matches)
def surf(ymin,xmin,ymax,xmax,image):
    image=image[ymin:ymax,xmin:xmax]
    surf = cv2.xfeatures2d.SURF_create()
    (kps, descs) = surf.detectAndCompute(image, None)
    return len(kps)
def approx_shape(ymin,xmin,ymax,xmax,image):
    pts=[]
    image=image[ymin:ymax,xmin:xmax]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    #cv2.imwrite("pool_trial4_edged/edges-"+str(count)+'-'+str(count1)+'.jpg',edged)
    _,cnts,_=cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        pts.append(len(approx))
        #cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
    #cv2.imwrite('pool_trial4_polygon/result-'+str(count)+'-'+str(count1)+'.jpg',image)
    if len(pts)!=0:
        return float(sum(pts)/(len(pts)))
    else:
        return 0
# convert the image to grayscale, blur it, and find edges
# in the image
def findEuclideanDistance(a, b):
    euclidean_distance = a - b
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def k_means(diver,num):
    vec=[]
    vec3=[]
    for obj in diver:
        for (i,image_path,id) in obj:
            vec.append(i)
            vec3.append((i,image_path,id))
    vec=np.array(vec)
    vec=np.float64(vec)
    kmeans =KMeans (n_clusters=num,random_state=0).fit(vec)
    vec2=kmeans.labels_
    vec2=vec2.tolist()
    vec4=[]
    for i in range(len(vec2)):
        j,image_path,id=vec3[i]
        vec4.append((image_path,id,vec2[i]))
    return vec4
def corner(ymin,xmin,ymax,xmax,image):
    img=image[ymin:ymax,xmin:xmax]
    fast = cv2.FastFeatureDetector_create(threshold=25)
    kp = fast.detect(img,None)
    print(len(kp))
    return len(kp)
def detect_objects(image_path):
    image = Image.open(image_path)
    (im_width,im_height)=image.size
    image2 = cv2.imread(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
    total=[]
    extra=[]
    #count1=0
    for i in range(int(num[0])):
        if classes[0][i]!=2:
            feature=[]
            ymin=int(boxes[0][i][0]*im_height)
            ymax=int(boxes[0][i][2]*im_height)
            xmin=int(boxes[0][i][1]*im_width)
            xmax=int(boxes[0][i][3]*im_width)
            temp=hull(ymin,xmin,ymax,xmax,image2)
            #for i in temp:
            feature.append(temp)
            #for m in get_hist(ymin,xmin,ymax,xmax,image2):
                #feature.append(m)
            #feature.append(remove_sea(ymin,xmin,ymax,xmax,image2)[0])
            #feature.append(remove_sea(ymin,xmin,ymax,xmax,image2)[1])

            #feature.append(matcher_count(ymin,xmin,xmax,ymax,image2))
            #feature.append(image[ymin:ymax,xmin:xmax].mean())
            #feature.append(surf(ymin,xmin,ymax,xmax,image2))
            #feature.append(orb_matcher(ymin,xmin,ymax,xmax,image2))
            #feature.append(corner(ymin,xmin,ymax,xmax,image2))
            for k in range(len(get_average_color(ymin,xmin,ymax,xmax,image2))):
                feature.append(get_average_color(ymin,xmin,ymax,xmax,image2)[k])
            for j in get_moment(ymin,xmin,ymax,xmax,image2).tolist():
                feature.append(j)
            #feature.append(corner(ymin,xmin,ymax,xmax,image2))
            feature.append(approx_shape(ymin,xmin,ymax,xmax,image2))
            #feature.append(get_hist(ymin,xmin,ymax,xmax,image2))
            #feature.append(m)
            r=get_amp(ymin,xmin,ymax,xmax,image2)[0]
            g=get_amp(ymin,xmin,ymax,xmax,image2)[1]
            b=get_amp(ymin,xmin,ymax,xmax,image2)[2]
            feature.append(r)
            feature.append(g)
            feature.append(b)
            print(feature)
            #count1=count1+1
            total.append((feature,image_path,(ymin,xmin,ymax,xmax)))
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        min_score_thresh=MINIMUM_CONFIDENCE,
        use_normalized_coordinates=True,
        line_thickness=8)
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(image_np, aspect = 'auto')
    plt.savefig('output/{}'.format(image_path), dpi = 62)
    plt.close(fig)
    return total
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image-{}.jpg'.format(i)) for i in range(1, 4) ]
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))
diver=[]
# Load model into memory
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('detecting...')
names=['Emma','Liam','Noah','Olivia',
'William','Ava', 'James'
'Isabella','Logan','Sophia','Benjamin',
'Mia','Mason','Charlotte',
'Elijah','Amelia',
'Oliver','Evelyn',
'Jacob','Abigail']
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        #count=0
        for image_path in TEST_IMAGE_PATHS:
            diver.append(detect_objects(image_path))
            #count=count+1
        max=0
        for i in range(len(diver)):
            if len(diver[i])>max:
                max=len(diver[i])
        vec3=k_means(diver,max)
        print(vec3)
        for (image_path,(ymin,xmin,ymax,xmax),id) in vec3:
            image=cv2.imread(image_path)
            cv2.rectangle(image,(xmin,ymax),(xmax,ymin),(0,255,0),2)
            cv2.putText(image, names[id], (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            cv2.imwrite(image_path,image)
