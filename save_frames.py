
new_width, new_height = 640, 360

import cv2
import numpy as np

from video_player import SilencePath, SpeechPath, VideoPath

def load_and_process_video(video_path: VideoPath, new_size):
    video_file = cv2.VideoCapture(video_path.video_path)
    processed_frames = []
    while True:
        ret, frame = video_file.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(frame_rgb)

    video_file.release()
    np.save(video_path.np_path, processed_frames)


if __name__ == '__main__':
    load_and_process_video(SpeechPath, (new_width, new_height))
    load_and_process_video(SilencePath, (new_width, new_height))