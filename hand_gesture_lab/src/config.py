import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# Camera config
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Pose & ROI config
ROI_PADDING_RATIO = 1.0  # Multiplier for shoulder distance to define ROI size
MIN_ROI_SIZE = 150

# Hand validation
MIN_HAND_CONFIDENCE = 0.5
MAX_JUMP_THRESHOLD = 0.2  # Max relative jump between consecutive frames
EMA_ALPHA = 0.5  # Smoothing factor (0 = no smoothing, 1 = max smoothing)

# Model config
SEQUENCE_LENGTH = 30
FEATURE_DIM = 63  # 21 keypoints * 3 coordinates (x, y, z)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 5  # Example: Swiping Left, Swiping Right, Stop Sign, Thumbs Up, No Gesture

GESTURE_CLASSES = {
    0: "Swiping Left",
    1: "Swiping Right",
    2: "Stop Sign",
    3: "Thumbs Up",
    4: "No Gesture"
}

# Inference config
PREDICTION_THRESHOLD = 0.8  # Softmax confidence needed to emit a gesture
DEBOUNCE_FRAMES = 15  # Frames to wait before emitting another gesture

# Training config
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
