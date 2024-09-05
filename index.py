import os
import cv2
import numpy as np
import logging
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
from scipy.spatial.distance import cosine
import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QProgressBar, QTextEdit, QLineEdit, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
import sys
from yt_dlp import YoutubeDL
from googleapiclient.discovery import build

# YouTube API setup - Replace with your own API key
YOUTUBE_API_KEY = 'AIzaSyCRFtIfiEyeYmCrCZ8Bvy8Z4IPBy1v2iwo'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


class YouTubeVideoInfo:
    def __init__(self, api_key):
        self.youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=api_key)

    def get_video_info(self, video_id):
        request = self.youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        return response['items'][0] if response['items'] else None


class VideoDownloader(QThread):
    progress_update = pyqtSignal(str)
    download_complete = pyqtSignal(str, str)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            video_id = self.url.split("v=")[-1]
            yt_info = YouTubeVideoInfo(YOUTUBE_API_KEY)
            video_details = yt_info.get_video_info(video_id)

            if not video_details:
                self.progress_update.emit("Error: Video not found.")
                return

            video_title = video_details["snippet"]["title"]
            thumbnail_url = video_details["snippet"]["thumbnails"]["high"]["url"]

            self.progress_update.emit("Downloading video...")
            ydl_opts = {
                'format': 'best',
                'outtmpl': f'{video_id}_video.%(ext)s',
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])

            video_path = os.path.join(os.getcwd(), f"{video_id}_video.mp4")

            self.progress_update.emit("Downloading thumbnail...")
            thumbnail_path = os.path.join(os.getcwd(), f"{video_id}_thumbnail.jpg")
            response = requests.get(thumbnail_url)
            with open(thumbnail_path, 'wb') as f:
                f.write(response.content)

            self.download_complete.emit(video_path, thumbnail_path)

        except Exception as e:
            self.progress_update.emit(f"Error: {str(e)}")


class VideoThumbnailMatcher(QThread):
    progress_update = pyqtSignal(int)
    result_ready = pyqtSignal(dict)

    def __init__(self, video_path, thumbnail_path, initial_skip_frames=30, min_skip_frames=5,
                 similarity_threshold=0.90, save_similar_threshold=0.40, resize_factor=0.25, debug=False):
        super().__init__()
        self.video_path = video_path
        self.thumbnail_path = thumbnail_path
        self.initial_skip_frames = initial_skip_frames
        self.min_skip_frames = min_skip_frames
        self.similarity_threshold = similarity_threshold
        self.save_similar_threshold = save_similar_threshold  # Frames with similarity > 0.40 will be saved
        self.best_frame = None
        self.best_frame_number = -1
        self.best_similarity = -1
        self.fps = 0
        self.resize_factor = resize_factor
        self.frame_size = None
        self.debug = debug
        self.debug_info = {}
        self.similar_frames = []
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def preprocess_image(self, image):
        if self.frame_size:
            image_resized = cv2.resize(image, self.frame_size)
        else:
            image_resized = image
        image_resized = cv2.resize(image_resized, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                           cells_per_block=(1, 1), visualize=False)
        return gray, hog_features

    def compare_images(self, img1, img1_hog, img2, img2_hog):
        ssim_score, _ = ssim(img1, img2, full=True)
        hog_similarity = 1 - cosine(img1_hog, img2_hog)
        combined_score = 0.6 * ssim_score + 0.4 * hog_similarity
        if self.debug:
            self.debug_info = {
                'ssim_score': ssim_score,
                'hog_similarity': hog_similarity,
                'combined_score': combined_score
            }
        return combined_score

    def find_similar_frames(self):
        thumbnail = cv2.imread(self.thumbnail_path)
        if thumbnail is None:
            raise ValueError(f"Could not load thumbnail from {self.thumbnail_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read the first frame from the video.")

        self.frame_size = (first_frame.shape[1], first_frame.shape[0])
        thumbnail = cv2.resize(thumbnail, self.frame_size)
        thumbnail, thumbnail_hog = self.preprocess_image(thumbnail)

        skip_frames = self.initial_skip_frames
        frame_number = 0
        last_improvement = 0
        output_dir = os.path.dirname(self.video_path)
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]

        while frame_number < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                frame_number += skip_frames
                self.progress_update.emit(int(frame_number / total_frames * 100))
                continue

            processed_frame, processed_frame_hog = self.preprocess_image(frame)
            similarity = self.compare_images(thumbnail, thumbnail_hog, processed_frame, processed_frame_hog)

            # Save similar frames above threshold
            if similarity > self.save_similar_threshold:
                similar_frame_path = os.path.join(output_dir, f"{base_name}_frame_{frame_number}_sim_{similarity:.4f}.jpg")
                cv2.imwrite(similar_frame_path, frame)

            # Check for best frame
            if similarity > self.best_similarity:
                self.best_similarity = similarity
                self.best_frame_number = frame_number
                self.best_frame = frame
                last_improvement = frame_number

            # Stop if similarity reaches the threshold
            if similarity >= self.similarity_threshold:
                break

            frame_number += skip_frames
            self.progress_update.emit(int(frame_number / total_frames * 100))

        cap.release()

        # Save best frame
        best_frame_path = os.path.join(output_dir, f"{base_name}_best_frame.jpg")
        if self.best_frame is not None:
            cv2.imwrite(best_frame_path, self.best_frame)

        result = {
            'best_frame': (self.best_frame_number, self.best_similarity, best_frame_path),
        }
        self.result_ready.emit(result)

    def run(self):
        self.find_similar_frames()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Video Thumbnail Matcher")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()

        # YouTube URL input
        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube URL")
        url_layout.addWidget(self.url_input)
        self.fetch_button = QPushButton("Fetch Video and Thumbnail")
        self.fetch_button.clicked.connect(self.fetch_youtube_video)
        url_layout.addWidget(self.fetch_button)
        main_layout.addLayout(url_layout)

        # File selection buttons
        file_layout = QHBoxLayout()
        self.video_button = QPushButton("Select Video")
        self.video_button.clicked.connect(self.select_video)
        file_layout.addWidget(self.video_button)
        self.thumbnail_button = QPushButton("Select Thumbnail")
        self.thumbnail_button.clicked.connect(self.select_thumbnail)
        file_layout.addWidget(self.thumbnail_button)
        main_layout.addLayout(file_layout)

        # Start processing button
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        main_layout.addWidget(self.start_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Result text
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        main_layout.addWidget(self.result_text)

        # Thumbnail preview
        self.thumbnail_preview = QLabel()
        self.thumbnail_preview.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.thumbnail_preview)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.video_path = ""
        self.thumbnail_path = ""

    def fetch_youtube_video(self):
        url = self.url_input.text()
        self.downloader = VideoDownloader(url)
        self.downloader.progress_update.connect(self.update_progress_text)
        self.downloader.download_complete.connect(self.set_downloaded_files)
        self.downloader.start()

    def update_progress_text(self, text):
        self.result_text.append(text)

    def set_downloaded_files(self, video_path, thumbnail_path):
        self.video_path = video_path
        self.thumbnail_path = thumbnail_path
        self.video_button.setText(f"Video: {os.path.basename(self.video_path)}")
        self.thumbnail_button.setText(f"Thumbnail: {os.path.basename(self.thumbnail_path)}")
        self.result_text.append("Video and thumbnail downloaded successfully.")
        self.update_thumbnail_preview()

    def select_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if self.video_path:
            self.video_button.setText(f"Video: {os.path.basename(self.video_path)}")

    def select_thumbnail(self):
        self.thumbnail_path, _ = QFileDialog.getOpenFileName(self, "Select Thumbnail Image", "", "Image Files (*.jpg *.png)")
        if self.thumbnail_path:
            self.thumbnail_button.setText(f"Thumbnail: {os.path.basename(self.thumbnail_path)}")
            self.update_thumbnail_preview()

    def update_thumbnail_preview(self):
        if self.thumbnail_path:
            pixmap = QPixmap(self.thumbnail_path)
            self.thumbnail_preview.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_processing(self):
        if not self.video_path or not self.thumbnail_path:
            self.result_text.setText("Please select both video and thumbnail")
            return

        self.matcher = VideoThumbnailMatcher(self.video_path, self.thumbnail_path)
        self.matcher.progress_update.connect(self.update_progress)
        self.matcher.result_ready.connect(self.show_results)
        self.matcher.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_results(self, result):
        best_frame_number, best_similarity, best_frame_path = result['best_frame']
        result_text = f"Best frame found at frame {best_frame_number} with similarity {best_similarity:.4f}\n"
        result_text += f"Best frame saved as: {best_frame_path}\n"
        self.result_text.setText(result_text)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
