import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import math
import cv2
from pydantic import BaseModel
from enum import StrEnum, auto


class VideoType(StrEnum):
    SILENCE = auto()
    SPEECH = auto()


class VideoPath(BaseModel):
    video_path: str
    video_type: VideoType
    np_path: str


SilencePath = VideoPath(
    video_path="data/video/Idle.mp4",
    video_type=VideoType.SILENCE,
    np_path="data/np/silence_cache.npy"
)
SpeechPath = VideoPath(
    video_path="data/video/Speaking.mp4",
    video_type=VideoType.SPEECH,
    np_path="data/np/voice_cache.npy"
)

StartSpeakingPath = VideoPath(
    video_path="data/video/start_speaking.mp4",
    video_type=VideoType.SPEECH,
    np_path="data/np/start_speaking_cache.npy"
)

EndSpeakingPath = VideoPath(
    video_path="data/video/start_speaking.mp4",
    video_type=VideoType.SPEECH,
    np_path="data/np/end_speaking_cache.npy"
)


class VideoPlayer:
    def __init__(self, child_conn):
        self.conn = child_conn

        self.root = tk.Tk()
        self.root.title("Ai helper")
        self.root.attributes('-topmost', True)
        self.video_label = tk.Label(self.root)
        self.video_label.pack(fill="both", expand=True)

        self.videos = {}
        self.current_video_type = None
        self.frame_index = 0
        self.window_width = 0
        self.window_height = 0

        self._setup_ui()
        self._load_and_preprocess_videos()

    def _setup_ui(self):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        temp_frames = np.load(SilencePath.np_path)
        frame_h, frame_w, _ = temp_frames[0].shape
        aspect_ratio = frame_w / frame_h

        target_area = (screen_w * screen_h) / 16
        self.window_height = int(math.sqrt(target_area / aspect_ratio))
        self.window_width = int(self.window_height * aspect_ratio)

        x_pos = int(screen_w * 0.04)
        y_pos = int(screen_h * 0.50)

        self.root.geometry(f"{self.window_width}x{self.window_height}+{x_pos}+{y_pos}")

    def _load_and_preprocess_videos(self):
        for path_obj in [SilencePath, SpeechPath]:
            all_frames = np.load(path_obj.np_path)
            resized_frames = []
            for frame in all_frames:
                resized = cv2.resize(frame, (self.window_width, self.window_height))
                resized_frames.append(resized)

            self.videos[path_obj.video_type] = resized_frames

    def _check_for_commands(self):
        if self.conn.poll():
            command = self.conn.recv()
            if command in self.videos and command != self.current_video_type:
                self._switch_video(command)

    def _switch_video(self, video_type: VideoType):
        self.current_video_type = video_type
        self.frame_index = 0

    def _update_frame(self):
        self._check_for_commands()

        current_frames = self.videos[self.current_video_type]

        frame_rgb = current_frames[self.frame_index]

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.frame_index = (self.frame_index + 1) % len(current_frames)
        self.root.after(20, self._update_frame)

    def run(self):
        self._switch_video(VideoType.SILENCE)  # Устанавливаем начальное видео
        self._update_frame()
        self.root.mainloop()


def player_process(child_conn):
    player = VideoPlayer(child_conn)
    player.run()
