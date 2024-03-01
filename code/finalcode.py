#from server import *
import numpy as np
import itertools
import threading
import darknet
import queue
import time
import copy
import cv2
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'python/'))
sys.path.append(os.getcwd().replace('darknet-master', ''))



# 각 파일 path
BASE_DIR = Path(__file__).resolve().parent

# 영상의 크기
#WIDTH = 1920
#HEIGHT = 1080
WIDTH = 800
HEIGHT = 600


#############################################################
#test1의 변수

BASE_DIR = 'C:'
weightsFile = 'C:/openpose/models/pose/body_25/pose_iter_584000.caffemodel'
protoFile = 'C:/openpose/models/pose/body_25/pose_deploy.prototxt'

nPoints = 25
keypointsMapping = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
                    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee",
                    "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                    "LHeel",  "RBigToe", "RSmallToe", "RHeel"]

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14],
              [11, 24], [11, 22], [22, 23], [14, 21], [14, 19], [19, 20],
              [1, 0], [0, 15], [15, 17], [0, 16], [16, 18],
              [2, 17], [5, 18]]
mapIdx = [[40, 41], [48, 49], [42, 43], [44, 45], [50, 51], [52, 53],
          [26, 27], [32, 33], [28, 29], [30, 31], [34, 35], [36, 37],
          [38, 39], [76, 77], [72, 73], [74, 75], [70, 71], [66, 67],
          [68, 69], [56, 57], [58, 59], [62, 63], [60, 61], [64, 65],
          [46, 47], [54, 55]]
colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [
              0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [
              255, 0, 0], [125, 200, 125], [125, 200, 0],
          [200, 200, 200], [200, 100, 200], [200, 200, 0], [
              0, 200, 0], [200, 0, 255], [0, 250, 125],
          [0, 200, 0], [0, 120, 200]]

device = 'gpu'


####################################################################
#detector_ip의 함수

def intersection(rect_a, rect_b):  # check if rect A & B intersect
    _a, _b = rect_a, rect_b
    a = [_a[0] - _a[2] / 2, _a[1] - _a[3] / 2, _a[0] + _a[2] / 2, _a[1] + _a[3] / 2]
    b = [_b[0] - _b[2] / 2, _b[1] - _b[3] / 2, _b[0] + _b[2] / 2, _b[1] + _b[3] / 2]
    start_x = max(min(a[0], a[2]), min(b[0], b[2]))
    start_y = max(min(a[1], a[3]), min(b[1], b[3]))
    end_x = min(max(a[0], a[2]), max(b[0], b[2]))
    end_y = min(max(a[1], a[3]), max(b[1], b[3]))
    if start_x < end_x and start_y < end_y:
        return True
    else:
        return False


def union(rect_a, rect_b):  # create bounding box for rect A & B
    _a, _b = rect_a, rect_b
    a = [_a[0] - _a[2] / 2, _a[1] - _a[3] / 2, _a[0] + _a[2] / 2, _a[1] + _a[3] / 2]
    b = [_b[0] - _b[2] / 2, _b[1] - _b[3] / 2, _b[0] + _b[2] / 2, _b[1] + _b[3] / 2]
    start_x = min(a[0], b[0])
    start_y = min(a[1], b[1])
    end_x = max(a[2], b[2])
    end_y = max(a[3], b[3])
    w = end_x - start_x
    h = end_y - start_y
    x = start_x + w / 2
    y = start_y + h / 2
    return [x, y, w, h]


def combine_boxes(rect):
    while True:
        found = 0
        for ra, rb in itertools.combinations(rect, 2):
            if intersection(ra[1], rb[1]):
                if ra in rect:
                    rect.remove(ra)
                if rb in rect:
                    rect.remove(rb)
                rect.append([ra[0], union(ra[1], rb[1])])
                found = 1
                break
        if found == 0:
            break
    return rect


# buffer less VideoCapture
class IpVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.frame_counter = 0
        read_thread = threading.Thread(target=self._reader)
        read_thread.daemon = True
        read_thread.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()

            self.frame_counter += 1
            # If the last frame is reached, reset the capture and the frame_counter
            if self.frame_counter == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self.frame_counter = 0  # Or whatever as long as it is the same as next line
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return True, self.q.get()


class Detector:
    def __init__(self, init_img, camera, factor=0.5):
        ########################################YOLOV4##############################################
        path = 'C:/darknet-master/build/darknet/x64'
        cfg_path = path + '/cfg/yolov4.cfg'
        weight_path = path + '/yolov4.weights'
        meta_path = 'C:/darknet-master/datadata/gohome/coco.data'
        self.network, self.class_names, self.colors = darknet.load_network(cfg_path, meta_path, weight_path)
        self.darknet_image = darknet.make_image(darknet.network_width(self.network),
                                                darknet.network_height(self.network), 3)

        self.prev_frame = init_img
        self.blending_factor = factor
        self.prev_results = []
        self.cam = camera
        self.factor = 2
        self.finish = False
        self.result = init_img
        self.origin = init_img

    def detect_img(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(self.network), darknet.network_height(self.network)),
                                   cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        meta_results = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=0.3)
        results = []
        h, w, c = frame_resized.shape

        for i in meta_results:
            if i[0] == 'person':
                results.append([i[0], [i[2][0] / w, i[2][1] / h, i[2][2] / w, i[2][3] / h]])

        return results

    def draw_result(self, frame, results):
        _h, _w, _c = frame.shape
        for i in results:
            w = int(i[1][2] * _w)
            h = int(i[1][3] * _h)
            x = int(i[1][0] * _w - w / 2)
            y = int(i[1][1] * _h - h / 2)

            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return frame

    def mosaic(self, src, ratio=0.1):
        # small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        # return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        small = cv2.resize(src, None, fx=ratio, fy=ratio)
        return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_AREA)

    def mosaic_area(self, src, x, y, width, height, ratio=0.1):
        dst = src.copy()
        dst[y:y + height, x:x + width] = self.mosaic(dst[y:y + height, x:x + width], ratio)
        return dst

    # mosaic
    def update_prev_frame_mosaic(self, frame, results):
        _h, _w, _c = frame.shape
        for i in results:
            w = int(i[1][2] * _w * self.factor)
            h = int(i[1][3] * _h * self.factor)
            x = int(i[1][0] * _w - w / 2)
            y = int(i[1][1] * _h - h / 2)
            if x < 0:
                x = 0
            elif x + h > _w:
                w = _w - x
            if y < 0:
                y = 0
            elif y + h > _h:
                h = _h - y
            self.result = self.mosaic_area(frame, x, y, w, h)

    # humanless
    def update_prev_frame(self, frame, results):
        _h, _w, _c = frame.shape
        results = combine_boxes(copy.deepcopy(results))
        prev_frame = self.prev_frame.copy()
        for i in results:
            w = int(i[1][2] * _w * self.factor)
            h = int(i[1][3] * _h * self.factor)
            x = int(i[1][0] * _w - w / 2)
            y = int(i[1][1] * _h - h / 2)
            if x < 0:
                x = 0
            elif x + h > _w:
                w = _w - x
            if y < 0:
                y = 0
            elif y + h > _h:
                h = _h - y
            cropped = prev_frame[y:y + h, x:x + w]
            frame[y:y + h, x:x + w] = cropped

        self.prev_frame = cv2.addWeighted(frame.copy(), self.blending_factor, self.prev_frame.copy(),
                                          1 - self.blending_factor, 0)

    # mosaic
    def mosaic_main_loop(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                continue
            image = cv2.resize(frame, (WIDTH, HEIGHT))
            res = self.detect_img(image)
            for t in self.prev_results:
                for i in t:
                    res.append(i)
            meta_result = image.copy()
            self.origin = meta_result
            self.result = meta_result
            self.update_prev_frame_mosaic(image.copy(), res)
            cv2.imshow("test", self.result)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break

    # humanless
    def main_loop(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                continue
            image = cv2.resize(frame, (WIDTH, HEIGHT))
            res = self.detect_img(image)
            cur_res = copy.deepcopy(res)

            for t in self.prev_results:
                for i in t:
                    res.append(i)
            meta_result = image.copy()
            self.origin = meta_result
            self.update_prev_frame(image.copy(), res)
            wr_result = cv2.hconcat([meta_result, self.prev_frame])
            self.result = wr_result
            self.result = self.prev_frame
            # cv2.imshow("test", self.result)
            # key = cv2.waitKey(10)
            # if key == ord('q'):
            #    break

            # if len(res) != 0:
            #    detector.prev_results.append(cur_res)
            # while len(detector.prev_results) >= 60:
            #    detector.prev_results.pop(0)


##################################################################
#test1의 함수

def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []
    # find the blobs
    contours, _ = cv2.findContours(
        mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        # maxVal = 있을 확률
        # maxLoc = x,y좌표
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    return keypoints


def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10  # 벡터를 나눌 수
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if (nA != 0 and nB != 0):
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(
                        valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else:  # If no keypoints are detected
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    # print(valid_pairs)
    return valid_pairs, invalid_pairs
# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
# It finds the person and index at which the joint should be added. This can be done since we have an id for each joint


def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 26))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(
                        int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 24:
                    row = -1 * np.ones(26)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k]
                                  [i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


##################################################################
#detector_ip의 main
if __name__ == '__main__':

    time1 = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    if device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")

    #cam_url = "rtsp://admin:sky123!!@192.168.0.13"
    blending_factor = 0.5
    #cam = IpVideoCapture(cam_url)
    cam = cv2.VideoCapture('C:/Users/ghkan/Downloads/aespa1.mp4')
    #cam = cv2.VideoCapture(0)
    init_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    init_image[:, :] = (0, 0, 0)

    detector = Detector(init_image, cam, blending_factor)
    # detector.main_loop()  # 휴면리스 처리 영상 출력
    # detector.mosaic_main_loop()  # 모자이크 처리 영상 출력
    t = threading.Thread(target=detector.main_loop)
    t.daemon = True
    t.start()

    while cv2.waitKey(1) < 0:
        imgOrigin = detector.origin
        imgResult = detector.result
        time.sleep(0.05)

        hasFrame1 = imgOrigin
        frame1 = imgOrigin
        hasFrame2 = imgResult
        frame2 = imgResult

        # 영상이 커서 느리면 사이즈를 줄이자
        # frame=cv2.resize(frame,dsize=(320,240),interpolation=cv2.INTER_AREA)

        # 웹캠으로부터 영상을 가져올 수 없으면 웹캠 중지
        if not cam:
            cv2.waitKey()
            break

        frameWidth = frame1.shape[1]
        frameHeight = frame1.shape[0]

        # Fix the input Height and get the width according to the Aspect Ratio
        inHeight = 368
        inWidth = int((inHeight / frameHeight) * frameWidth)

        inpBlob = cv2.dnn.blobFromImage(
            frame1, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)
        output = net.forward()
        # print("Time Taken = {}".format(time.time() - t))

        # for i in range(78):
        #     n = i+1
        #     probMap = output[0,i, :, :]
        #     probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0
        threshold = 0.1

        for part in range(nPoints):
            probMap = output[0, part, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))
            #     plt.figure()
            #     plt.imshow(255*np.uint8(probMap>threshold))
            keypoints = getKeypoints(probMap, threshold)
            # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)
        # print(detected_keypoints)

        for i in range(nPoints):
            for j in range(len(detected_keypoints[i])):
                cv2.circle(frame2, detected_keypoints[i][j][0:2], 3, [
                    0, 0, 255], -1, cv2.LINE_AA)

        valid_pairs, invalid_pairs = getValidPairs(output)

        personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

        for i in range(24):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frame2, (B[0], A[0]), (B[1], A[1]),
                         colors[i], 3, cv2.LINE_AA)

        cv2.imshow("Output-Keypoints", frame2)


    #------------------------------------------------------------------------


    cam.release()  # 카메라 장치에서 받아온 메모리 해제
    cv2.destroyAllWindows()  # 모든 윈도우 창 닫음
