import os
import cv2
import numpy as np
import logging
import re
import requests
import sys
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
from scipy.spatial.distance import cosine
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, 
                             QProgressBar, QTextEdit, QLineEdit, QHBoxLayout, QGridLayout, QScrollArea, QSplitter, 
                             QFrame, QStyleFactory, QMessageBox, QComboBox, QDialog, QDialogButtonBox, QCheckBox,
                             QSlider, QAction, QMenuBar, QMenu)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
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
        self.video_path = ""
        self.thumbnail_path = ""
    def run(self):
        try:
            video_id_match = re.search(r"(?:v=|/v/|/vi/|v%3D|vi%3D)([^&?/\"]{11})", self.url)
            if not video_id_match:
                self.progress_update.emit("Error: Could not extract video ID from URL.")
                return
            video_id = video_id_match.group(1)
            yt_info = YouTubeVideoInfo(YOUTUBE_API_KEY)
            video_details = yt_info.get_video_info(video_id)
            if not video_details:
                self.progress_update.emit("Error: Video not found.")
                return
            video_title = video_details["snippet"]["title"]
            thumbnail_url = video_details["snippet"]["thumbnails"]["high"]["url"]
            self.progress_update.emit("Downloading video...")
            ydl_opts = {'format': 'best', 'outtmpl': f'{video_id}_video.%(ext)s'}
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            self.video_path = os.path.join(os.getcwd(), f"{video_id}_video.mp4")
            self.progress_update.emit("Downloading thumbnail...")
            self.thumbnail_path = os.path.join(os.getcwd(), f"{video_id}_thumbnail.jpg")
            response = requests.get(thumbnail_url)
            with open(self.thumbnail_path, 'wb') as f:
                f.write(response.content)
            self.download_complete.emit(self.video_path, self.thumbnail_path)
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
        self.save_similar_threshold = save_similar_threshold
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
        image_resized = cv2.resize(image, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
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
    def run(self):
        try:
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
            while frame_number < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    frame_number += skip_frames
                    continue
                processed_frame, processed_frame_hog = self.preprocess_image(frame)
                similarity = self.compare_images(thumbnail, thumbnail_hog, processed_frame, processed_frame_hog)
                if similarity > self.best_similarity:
                    self.best_similarity = similarity
                    self.best_frame_number = frame_number
                    self.best_frame = frame
                if similarity > self.save_similar_threshold:
                    self.similar_frames.append((frame_number, similarity))
                frame_number += skip_frames
                self.progress_update.emit(int(frame_number / total_frames * 100))
            cap.release()
            best_frame_path = os.path.join(os.path.dirname(self.video_path), f"best_frame_{self.best_frame_number}.jpg")
            if self.best_frame is not None:
                cv2.imwrite(best_frame_path, self.best_frame)
            self.result_ready.emit({
                'best_frame': (self.best_frame_number, self.best_similarity, best_frame_path),
                'similar_frames': self.similar_frames
            })
        except Exception as e:
            self.progress_update.emit(int(0))
            self.result_ready.emit({'error': str(e)})
class HDOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HD Download Options")
        layout = QVBoxLayout(self)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG (Lossless)", "JPEG (High Quality)", "TIFF (Uncompressed)"])
        layout.addWidget(QLabel("Format:"))
        layout.addWidget(self.format_combo)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["Original", "4K (3840x2160)", "1080p (1920x1080)", "720p (1280x720)"])
        layout.addWidget(QLabel("Resolution:"))
        layout.addWidget(self.resolution_combo)
        self.denoise_check = QCheckBox("Apply denoising")
        layout.addWidget(self.denoise_check)
        self.sharpen_check = QCheckBox("Apply sharpening")
        layout.addWidget(self.sharpen_check)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Video Thumbnail Matcher")
        self.setGeometry(100, 100, 1200, 800)
        # Menu bar
        self.create_menu_bar()
        # Dark mode
        self.dark_mode = False
        self.set_style()
        main_layout = QVBoxLayout()
        # YouTube URL input
        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube URL")
        url_layout.addWidget(self.url_input)
        self.fetch_button = QPushButton("Fetch Video and Thumbnail")
        self.fetch_button.setIcon(QIcon('icons/download.png'))
        self.fetch_button.clicked.connect(self.fetch_youtube_video)
        url_layout.addWidget(self.fetch_button)
        self.clear_button = QPushButton("Clear All")
        self.clear_button.setIcon(QIcon('icons/clear.png'))
        self.clear_button.clicked.connect(self.clear_all)
        url_layout.addWidget(self.clear_button)
        main_layout.addLayout(url_layout)
        # File selection buttons
        file_layout = QHBoxLayout()
        self.video_button = QPushButton("Select Video")
        self.video_button.setIcon(QIcon('icons/video.png'))
        self.video_button.clicked.connect(self.select_video)
        file_layout.addWidget(self.video_button)
        self.thumbnail_button = QPushButton("Select Thumbnail")
        self.thumbnail_button.setIcon(QIcon('icons/image.png'))
        self.thumbnail_button.clicked.connect(self.select_thumbnail)
        file_layout.addWidget(self.thumbnail_button)
        main_layout.addLayout(file_layout)
        # Start processing button
        self.start_button = QPushButton("Start Processing")
        self.start_button.setIcon(QIcon('icons/play.png'))
        self.start_button.clicked.connect(self.start_processing)
        main_layout.addWidget(self.start_button)
        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        # Video player
        self.video_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = QVideoWidget()
        main_layout.addWidget(self.video_widget)
        self.video_player.setVideoOutput(self.video_widget)
        # Playback controls
        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_pause_video)
        playback_layout.addWidget(self.play_button)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.sliderMoved.connect(self.set_position)
        playback_layout.addWidget(self.seek_slider)
        main_layout.addLayout(playback_layout)
        # Similarity threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Similarity Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(90)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.90")
        threshold_layout.addWidget(self.threshold_label)
        main_layout.addLayout(threshold_layout)
        # Similarity grouping dropdown
        grouping_layout = QHBoxLayout()
        grouping_layout.addWidget(QLabel("Group by similarity:"))
        self.grouping_combo = QComboBox()
        self.grouping_combo.addItems(["All", "More Similar", "Less Similar", "Not Good Similar"])
        self.grouping_combo.currentIndexChanged.connect(self.update_similar_frames_grid)
        grouping_layout.addWidget(self.grouping_combo)
        main_layout.addLayout(grouping_layout)
        # Splitter for result text and thumbnail preview
        splitter = QSplitter(Qt.Horizontal)
        # Result text
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        splitter.addWidget(self.result_text)
        # Thumbnail preview
        self.thumbnail_preview = QLabel()
        self.thumbnail_preview.setAlignment(Qt.AlignCenter)
        self.thumbnail_preview.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        splitter.addWidget(self.thumbnail_preview)
        main_layout.addWidget(splitter)
        # Similar frames grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.scroll_area.setWidget(self.grid_widget)
        main_layout.addWidget(self.scroll_area)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.video_path = ""
        self.thumbnail_path = ""
        self.similarity_threshold = 0.90
    def create_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        # File menu
        file_menu = QMenu("&File", self)
        menu_bar.addMenu(file_menu)
        export_action = QAction("Export Results to CSV", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        # View menu
        view_menu = QMenu("&View", self)
        menu_bar.addMenu(view_menu)
        toggle_dark_mode_action = QAction("Toggle Dark Mode", self)
        toggle_dark_mode_action.triggered.connect(self.toggle_dark_mode)
        view_menu.addAction(toggle_dark_mode_action)
        # Batch menu
        batch_menu = QMenu("&Batch", self)
        menu_bar.addMenu(batch_menu)
        batch_process_action = QAction("Batch Process Videos", self)
        batch_process_action.triggered.connect(self.batch_process_videos)
        batch_menu.addAction(batch_process_action)
    def set_style(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    text-align: center;
                    text-decoration: none;
                    font-size: 14px;
                    margin: 4px 2px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QLineEdit, QTextEdit {
                    background-color: #3b3b3b;
                    color: #ffffff;
                    border: 1px solid #555555;
                }
                QProgressBar {
                    border: 2px solid #555555;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #f0f0f0;
                    color: #000000;
                }
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    text-align: center;
                    text-decoration: none;
                    font-size: 14px;
                    margin: 4px 2px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QLineEdit, QTextEdit {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1px solid #ddd;
                }
                QProgressBar {
                    border: 2px solid grey;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
            """)
    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.set_style()
    def play_pause_video(self):
        if self.video_player.state() == QMediaPlayer.PlayingState:
            self.video_player.pause()
            self.play_button.setText("Play")
        else:
            self.video_player.play()
            self.play_button.setText("Pause")
    def set_position(self, position):
        self.video_player.setPosition(position)
    def update_threshold(self, value):
        self.similarity_threshold = value / 100
        self.threshold_label.setText(f"{self.similarity_threshold:.2f}")
    def export_results(self):
        if not hasattr(self, 'similar_frames'):
            QMessageBox.warning(self, "Error", "No results to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "CSV Files (*.csv)")
        if file_path:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Frame Number', 'Time', 'Similarity'])
                for frame_number, similarity in self.similar_frames:
                    time = self.format_time(frame_number / self.matcher.fps)
                    writer.writerow([frame_number, time, similarity])
            QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")
    def batch_process_videos(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with Videos")
        if folder_path:
            video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            if not video_files:
                QMessageBox.warning(self, "Error", "No video files found in the selected folder.")
                return
            for video_file in video_files:
                video_path = os.path.join(folder_path, video_file)
                self.video_path = video_path
                self.generate_thumbnail()
                self.start_processing()
            QMessageBox.information(self, "Batch Processing Complete", f"Processed {len(video_files)} videos.")
    def generate_thumbnail(self):
        if not self.video_path:
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        if ret:
            thumbnail_path = os.path.splitext(self.video_path)[0] + "_thumbnail.jpg"
            cv2.imwrite(thumbnail_path, frame)
            self.thumbnail_path = thumbnail_path
            self.update_thumbnail_preview()
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
        self.matcher = VideoThumbnailMatcher(self.video_path, self.thumbnail_path, 
                                             similarity_threshold=self.similarity_threshold)
        self.matcher.progress_update.connect(self.update_progress)
        self.matcher.result_ready.connect(self.show_results)
        self.matcher.start()
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    def show_results(self, result):
        if 'error' in result:
            self.result_text.setText(f"Error: {result['error']}")
            return
        best_frame_number, best_similarity, best_frame_path = result['best_frame']
        self.similar_frames = result['similar_frames']
        best_time = self.format_time(best_frame_number / self.matcher.fps)
        result_text = f"Best frame found at {best_time} with similarity {best_similarity:.4f}\n"
        result_text += f"Best frame saved as: {best_frame_path}\n\n"
        result_text += "Similar frames for human verification:\n"
        # Set video for playback
        self.video_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.video_path)))
        self.seek_slider.setRange(0, self.video_player.duration())
        self.update_similar_frames_grid()
        for frame_number, similarity in self.similar_frames:
            time = self.format_time(frame_number / self.matcher.fps)
            result_text += f"Frame at {time} - Similarity: {similarity:.4f}\n"
        self.result_text.setText(result_text)
    def update_similar_frames_grid(self):
        # Clear previous grid items
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)
        grouping = self.grouping_combo.currentText()
        filtered_frames = self.similar_frames
        if grouping == "More Similar":
            filtered_frames = [f for f in self.similar_frames if f[1] >= 0.5]
        elif grouping == "Less Similar":
            filtered_frames = [f for f in self.similar_frames if 0.4 <= f[1] < 0.5]
        elif grouping == "Not Good Similar":
            filtered_frames = [f for f in self.similar_frames if f[1] < 0.4]
        row = 0
        col = 0
        for frame_number, similarity in filtered_frames:
            time = self.format_time(frame_number / self.matcher.fps)
            frame = self.get_frame(self.video_path, frame_number)
            if frame is not None:
                frame_widget = self.create_frame_widget(frame, frame_number, time, similarity)
                self.grid_layout.addWidget(frame_widget, row, col)
                col += 1
                if col == 4:  # 4 columns in the grid
                    col = 0
                    row += 1
    def format_time(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"
    def get_frame(self, video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        return None
    def create_frame_widget(self, frame, frame_number, time, similarity):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        # Convert frame to QPixmap and resize
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Add image to layout
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_label)
        # Add time and similarity info
        info_label = QLabel(f"Time: {time}\nSimilarity: {similarity:.4f}")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        # Add download button
        download_btn = QPushButton("Download HD")
        download_btn.setIcon(QIcon('icons/save.png'))
        download_btn.clicked.connect(lambda: self.download_frame_hd(frame_number, time))
        layout.addWidget(download_btn)
        return widget
    def download_frame_hd(self, frame_number, time):
        if not self.video_path:
            QMessageBox.warning(self, "Error", "No video file selected.")
            return
        options_dialog = HDOptionsDialog(self)
        if options_dialog.exec_() != QDialog.Accepted:
            return
        format_option = options_dialog.format_combo.currentText()
        resolution_option = options_dialog.resolution_combo.currentText()
        apply_denoise = options_dialog.denoise_check.isChecked()
        apply_sharpen = options_dialog.sharpen_check.isChecked()
        if format_option == "PNG (Lossless)":
            file_extension = ".png"
            save_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
        elif format_option == "JPEG (High Quality)":
            file_extension = ".jpg"
            save_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        else:  # TIFF
            file_extension = ".tiff"
            save_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
        file_path, _ = QFileDialog.getSaveFileName(self, "Save HD Frame", f"frame_hd_{time.replace(':', '_')}{file_extension}", f"{format_option} (*{file_extension})")
        if file_path:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            if ret:
                if resolution_option != "Original":
                    width = int(resolution_option.split("(")[1].split("x")[0])
                    height = int(resolution_option.split("x")[1].split(")")[0])
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
                if apply_denoise:
                    frame = self.denoise_image(frame)
                if apply_sharpen:
                    frame = self.sharpen_image(frame)
                cv2.imwrite(file_path, frame, save_params)
                QMessageBox.information(self, "Frame Saved", f"HD Frame saved as: {file_path}")
                self.result_text.append(f"HD Frame saved as: {file_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to read the frame from the video.")
    def denoise_image(self, image):
        # Apply Non-local Means Denoising
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    def sharpen_image(self, image):
        # Create a sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                                       [-1, -1, -1]])
        # Apply the sharpening kernel
        sharpened = cv2.filter2D(image, -1, kernel)
        # Blend the sharpened image with the original
        return cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
    def enhance_image(self, image):
        # Apply some basic image enhancements
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        return enhanced
    def clear_all(self):
        # Clear video and thumbnail paths
        self.video_path = ""
        self.thumbnail_path = ""
        # Reset button texts
        self.video_button.setText("Select Video")
        self.thumbnail_button.setText("Select Thumbnail")
        # Clear URL input
        self.url_input.clear()
        # Clear result text
        self.result_text.clear()
        # Reset progress bar
        self.progress_bar.setValue(0)
        # Clear thumbnail preview
        self.thumbnail_preview.clear()
        # Clear video player
        self.video_player.setMedia(QMediaContent())
        # Clear similar frames grid
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)
        # Delete downloaded files
        if hasattr(self, 'downloader'):
            if os.path.exists(self.downloader.video_path):
                os.remove(self.downloader.video_path)
            if os.path.exists(self.downloader.thumbnail_path):
                os.remove(self.downloader.thumbnail_path)
        # Delete best frame if it exists
        if hasattr(self, 'matcher'):
            best_frame_path = os.path.join(os.path.dirname(self.video_path), f"best_frame_{self.matcher.best_frame_number}.jpg")
            if os.path.exists(best_frame_path):
                os.remove(best_frame_path)
        # Reset other attributes
        self.similar_frames = []
        if hasattr(self, 'matcher'):
            del self.matcher
        if hasattr(self, 'downloader'):
            del self.downloader
        QMessageBox.information(self, "Clear All", "All content has been cleared and reset.")
def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()
