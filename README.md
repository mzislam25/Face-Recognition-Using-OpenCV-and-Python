# Face Recognition Using OpenCV and Python
This project is an implementation of face recognition using OpenCV and Python, with the use of Haar cascades for face detection. The program can detect faces from webcam and draw a rectangle around each detected face.

## Getting Started
To run the program, you will need to have Python 3 and OpenCV installed on your machine. You can install the prerequisites:

```pip install requirements.txt```

## Usage
To run the program, open a command prompt or terminal and navigate to the directory containing the Python script. Then run the scripts sequentially:

## Command
01 ```python 01_face_detection.py```

02 ```python 02_face_training.py```

03 ```python 03_face_recognizer.py```

The program will open a window displaying the webcam. Press the 'q' key to exit the program.

## Customization
The program can be customized to work with different image or video sources. For example, to read from a video file instead of a camera, modify the cv2.VideoCapture function to specify the filename:

```
camera = cv2.VideoCapture('video.mp4')
```

The program can also be modified to perform additional processing on the detected faces, such as face feature extraction or comparison with a database of known faces.

## Contributing
Contributions to this project are welcome! If you find a bug or have a suggestion for an improvement, please create a new issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.