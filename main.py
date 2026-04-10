import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from src.preprocess import create_data_generators
from src.model import build_model, unfreeze_top_layers
from src.train import train_phase1, train_phase2, plot_training_history
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    print_evaluation_summary
)
from src.predict import predict_single_image, generate_gradcam

CONFIG = {
    'DATA_DIR': 'data/chest_xray',
    'PHASE1_EPOCHS': 20,
    'PHASE2_EPOCHS': 10,
    'LEARNING_RATE': 1e-4,
    'FINETUNE_LR': 1e-5,
    'THRESHOLD': 0.5,
    'MODEL_PATH': 'models/best_model_phase2.keras',
}

def print_banner():
    print("""
========================================
  AI Medical Image Analysis System
  Pneumonia Detection from Chest X-Ray
  Model: MobileNetV2 Transfer Learning
========================================
    """)
    print(f"TensorFlow: {tf._version_}")
    print(f"GPU: {bool(tf.config.list_physical_devices('GPU'))}")

def run_training():
    if not os.path.exists(CONFIG['DATA_DIR']):
        print(f"ERROR: Dataset not found at {CONFIG['DATA_DIR']}")
        print("Please download dataset first!")
        sys.exit(1)
    train_gen, val_gen, test_gen = create_data_generators(
        CONFIG['DATA_DIR']
    )
    model = build_model(learning_rate=CONFIG['LEARNING_RATE'])
    history1 = train_phase1(
        model, train_gen, val_gen,
        epochs=CONFIG['PHASE1_EPOCHS']
    )
    model = unfreeze_top_layers(
        model,
        num_layers=30,
        learning_rate=CONFIG['FINETUNE_LR']
    )
    history2 = train_phase2(
        model, train_gen, val_gen,
        epochs=CONFIG['PHASE2_EPOCHS']
    )
    plot_training_history(history1, history2)
    print("\nTraining complete!")
    print(f"Best model saved to {CONFIG['MODEL_PATH']}")

def run_evaluation():
    if not os.path.exists(CONFIG['MODEL_PATH']):
        print("ERROR: No model found. Run training first!")
        sys.exit(1)
    print(f"Loading model: {CONFIG['MODEL_PATH']}")
    model = tf.keras.models.load_model(CONFIG['MODEL_PATH'])
    _, _, test_gen = create_data_generators(CONFIG['DATA_DIR'])
    metrics = evaluate_model(model, test_gen)
    plot_confusion_matrix(metrics)
    plot_roc_curve(metrics)
    print_evaluation_summary(metrics)
    print("\nEvaluation complete!")

def run_prediction(image_path):
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    if not os.path.exists(CONFIG['MODEL_PATH']):
        print("ERROR: No model found. Run training first!")
        sys.exit(1)
    model = tf.keras.models.load_model(CONFIG['MODEL_PATH'])
    predict_single_image(model, image_path)
    generate_gradcam(model, image_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Medical Image Analysis"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'predict', 'all'],
        required=True
    )
    parser.add_argument('--image', type=str, default=None)
    return parser.parse_args()

if _name_ == "_main_":
    print_banner()
    args = parse_args()
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    if args.mode == 'train':
        run_training()
    elif args.mode == 'evaluate':
        run_evaluation()
    elif args.mode == 'predict':
        if args.image is None:
            print("ERROR: --image required for predict mode")
            sys.exit(1)
        run_prediction(args.image)
    elif args.mode == 'all':
        run_training()
        run_evaluation()
    print("\nDone! Check outputs/ folder for results.")