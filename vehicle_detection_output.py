
import glob #函数功能：匹配所有的符合条件的文件，并将其以list的形式返回。
from collections import deque
from seaborn import heatmap
from scipy.ndimage.measurements import label
from skimage.feature import hog
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

car_path='../data/vehicles/vehicles/**/*.png'
noncar_path='../data/non-vehicles/non-vehicles/GTI/*.png'
car_images=glob.glob(car_path)
noncar_images=glob.glob(noncar_path)

def hog_feature(images, cspace='RGB',  hog_channel=0):
    features = []
    for image in images:
        image = mpimg.imread(image)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(hog(feature_image[:,:,channel],
                                        orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),
                                        visualize=False, feature_vector=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = hog(feature_image[:,:,hog_channel], orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),
                               visualize=False, feature_vector=True)
        features.append(hog_features)

    return features

car_features=hog_feature(car_images)
noncar_features=hog_feature(noncar_images)

X = np.vstack((car_features, noncar_features)).astype(np.float64)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)


svc = LinearSVC()
svc.fit(X_train, y_train)
print('Test Accuracy of SVC = {:.2%}'.format(svc.score(X_test, y_test)))
CM = confusion_matrix(y_test, svc.predict(X_test))
print('False positives {:.2%}'.format( CM[0][1] / len(y_test)))
print('False negatives {:.2%}'.format( CM[1][0] / len(y_test)))

def finding_vehicle(image,cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)

    image1=np.copy(feature_image)
    x_end = image1.shape[1]
    x_start=0
    y_end=image1.shape[0]
    y_start = 336#去掉不要的上半部分
    nx_step=25
    ny_step = 50
    bbox=[]

    for ny in range(y_start, y_end, ny_step):
        for nx in range(x_start, x_end, nx_step):
            startx = nx
            endx = nx + 150
            starty = ny
            endy = ny + 150
            sub_image=image1[starty:endy, startx:endx]
            sub_image = cv2.resize(sub_image, (64, 64))

            ch1 = sub_image[:, :, 0]
            ch2 = sub_image[:, :, 1]
            ch3 = sub_image[:, :, 2]

            hog_features1 = hog(ch1, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               visualize=False, feature_vector=True).ravel()
            hog_features2 = hog(ch2, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               visualize=False, feature_vector=True).ravel()
            hog_features3 = hog(ch3, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               visualize=False, feature_vector=True).ravel()

            hog_feature = np.vstack((hog_features1, hog_features2, hog_features3)).astype(np.float64)
            test_prediction = svc.predict(hog_feature)

            if np.all(test_prediction) == 1:
                #output=cv2.rectangle(image, (startx, starty), (endx, endy), (0, 255, 0), 6)
                bbox.append(((startx, starty), (endx, endy)))

    return bbox

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

def heatmap(image,bbox):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, bbox)
    threshold = 5
    heat = apply_threshold(heat, threshold)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img

def pipline(image):
    bbox=finding_vehicle(image)
    heatmap_image=heatmap(image,  bbox=bbox)
    return heatmap_image

import imageio
imageio.plugins.ffmpeg.download
from moviepy.editor import VideoFileClip
from IPython.display import HTML

history = deque(maxlen = 8)
output = '../data/test_video/test_result.mp4'
clip = VideoFileClip("../data/test_video/test_video.mp4")
video_clip = clip.fl_image(pipline)
video_clip.write_videofile(output, audio=False, fps=5)


history = deque(maxlen = 8)
output = '../data/test_video/project_result.mp4'
clip = VideoFileClip("../data/test_video/project_video.mp4")
video_clip = clip.fl_image(pipline)
video_clip.write_videofile(output, audio=False, fps=5)

