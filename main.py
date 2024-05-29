import cv2
import numpy as np

class VisualSLAM3D:
    def __init__(self, map_size=1000):
        #Oriented FAST rotated BRIEF
        self.orb = cv2.ORB_create()
        #Brute Force Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.position = np.array([map_size // 2, map_size // 2, 0], dtype=np.float64)
        self.map_size = map_size
        self.map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        self.focal_length = 718.856
        self.baseline = 0.54
        #Stereo Semi-Global Block Matching
        self.stereo = cv2.StereoSGBM_create(minDisparity=0,
                                            numDisparities=16 * 5,
                                            blockSize=5,
                                            P1=8 * 3 * 5 ** 2,
                                            P2=32 * 3 * 5 ** 2,
                                            disp12MaxDiff=1,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=32)

    def compute_depth(self, left_img, right_img):
        disparity = self.stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0.1
        depth = (self.focal_length * self.baseline) / disparity
        return depth

    def process_frame(self, left_frame, right_frame):
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        keypoints_left, descriptors_left = self.orb.detectAndCompute(gray_left, None)
        keypoints_right, descriptors_right = self.orb.detectAndCompute(gray_right, None)

        if self.prev_descriptors is not None and descriptors_left is not None and len(descriptors_left) > 0:
            matches = self.bf.match(self.prev_descriptors, descriptors_left)
            matches = sorted(matches, key=lambda x: x.distance)
            self.update_position(matches, keypoints_left, gray_left, gray_right)
            self.draw_matches(left_frame, keypoints_left, matches)
            self.update_map(keypoints_left, matches, gray_left, gray_right)

        self.prev_keypoints = keypoints_left
        self.prev_descriptors = descriptors_left

    def update_position(self, matches, keypoints, left_img, right_img):
        if len(matches) < 10:
            return

        delta_positions = []
        depth_map = self.compute_depth(left_img, right_img)

        for match in matches:
            prev_pt = self.prev_keypoints[match.queryIdx].pt
            curr_pt = keypoints[match.trainIdx].pt

            z = depth_map[int(curr_pt[1]), int(curr_pt[0])]
            if z <= 0 or z > 100:
                continue

            delta_positions.append(np.array([curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1], z]))

        if delta_positions:
            avg_delta = np.mean(delta_positions, axis=0)
            self.position += avg_delta * 0.1

    def update_map(self, keypoints, matches, left_img, right_img):
        depth_map = self.compute_depth(left_img, right_img)
        for match in matches:
            kp = keypoints[match.trainIdx].pt
            z = depth_map[int(kp[1]), int(kp[0])]
            if z <= 0 or z > 100:
                continue
            x = int(self.position[0] + (kp[0] - self.map_size // 2) * 0.5)
            y = int(self.position[1] + (kp[1] - self.map_size // 2) * 0.5)
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                self.map[y, x] = [255, 255, 255]

    def draw_matches(self, frame, keypoints, matches):
        for match in matches:
            pt1 = tuple(map(int, self.prev_keypoints[match.queryIdx].pt))
            pt2 = tuple(map(int, keypoints[match.trainIdx].pt))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(frame, pt2, 5, (0, 0, 255), -1)

        cv2.putText(frame, f"Position: {self.position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Visual SLAM', frame)
        cv2.waitKey(1)

    def visualize_depth(self, depth_map):
        depth_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow('Depth Map', depth_visual)
        cv2.waitKey(1)

    def show_map(self):
        map_with_axes = self.map.copy()
        cv2.line(map_with_axes, (self.map_size // 2, 0), (self.map_size // 2, self.map_size), (0, 255, 0), 1)
        cv2.line(map_with_axes, (0, self.map_size // 2), (self.map_size, self.map_size // 2), (0, 255, 0), 1)
        cv2.circle(map_with_axes, (int(self.position[0]), int(self.position[1])), 5, (0, 0, 255), -1)
        cv2.imshow('Map', map_with_axes)
        cv2.waitKey(1)

def main():
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)
    slam = VisualSLAM3D()

    while cap_left.isOpened() and cap_right.isOpened():
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        if not ret_left or not ret_right:
            break

        slam.process_frame(frame_left, frame_right)
        slam.show_map()

        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        depth_map = slam.compute_depth(gray_left, gray_right)
        slam.visualize_depth(depth_map)

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()