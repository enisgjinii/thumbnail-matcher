# YouTube Video Thumbnail Matcher
The YouTube Video Thumbnail Matcher is a desktop application that allows you to find the best matching frame in a YouTube video based on the provided thumbnail image. It uses a combination of structural similarity (SSIM) and Histogram of Oriented Gradients (HOG) to compare the thumbnail with the video frames and identify the most similar one.
## Features
- Download YouTube videos and their thumbnails directly from the application
- Select local video and thumbnail files
- Analyze the video and find the best matching frame based on the thumbnail
- Display similar frames for human verification
- Playback the video within the application
- Export the results to a CSV file
- Toggle dark mode for better visibility
- Batch process multiple videos in a folder
- Download high-quality (HD) frames with various options (format, resolution, denoising, sharpening)
## Requirements
- Python 3.6 or higher
- PyQt5
- OpenCV
- scikit-image
- scipy
- numpy
- yt-dlp
- google-api-python-client
## Installation
1. Clone the repository:
git clone https://github.com/your-username/youtube-video-thumbnail-matcher.git
2. Install the required dependencies:

        pip install -r requirements.txt
3. Replace `'YOUR_API_KEY_HERE'` in the `index.py` file with your actual YouTube API key.
## Usage
1. Run the application:
python index.py
2. Use the provided features to download, select, and analyze your YouTube videos and thumbnails.
## Contributing
If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
## License
This project is licensed under the [MIT License](LICENSE).