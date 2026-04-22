import cv2
import os
import argparse
import torch

from . import config
from .pipeline import GesturePipeline
from .lstm_model import GestureLSTM
from .inference import InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Real-time Hand Gesture Recognition")
    parser.add_argument('--camera', type=int, default=config.CAMERA_INDEX, help='Camera index')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights')
    args = parser.parse_args()

    # Initialize Model & Inference Engine
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = GestureLSTM(
        input_size=config.FEATURE_DIM, 
        hidden_size=config.HIDDEN_SIZE, 
        num_layers=config.NUM_LAYERS, 
        num_classes=config.NUM_CLASSES
    )
    
    inference_engine = InferenceEngine(model, device=device)
    
    if args.weights and os.path.exists(args.weights):
        inference_engine.load_weights(args.weights)
    else:
        print("No weights provided or found. Running with random weights for demonstration.")

    # Initialize Pipeline
    pipeline = GesturePipeline(inference_engine=inference_engine)

    # Initialize Webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    print("Starting webcam stream. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Flip horizontally for selfie-view
            frame = cv2.flip(frame, 1)

            # Process frame
            processed_frame, predictions = pipeline.process_frame(frame, visualize=True)

            # Display
            cv2.imshow('Hand Gesture Recognition', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()

if __name__ == "__main__":
    main()
