import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.preprocess import load_and_preprocess_image, IMG_SIZE

def predict_single_image(model, image_path, threshold=0.5):
    img = load_and_preprocess_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    probability = float(model.predict(img_batch, verbose=0)[0][0])
    if probability >= threshold:
        prediction = "PNEUMONIA"
        confidence = probability * 100
        risk_level = "HIGH" if probability > 0.80 else "MEDIUM"
    else:
        prediction = "NORMAL"
        confidence = (1 - probability) * 100
        risk_level = "LOW"
    print("\n" + "="*50)
    print("  AI DIAGNOSIS RESULT")
    print("="*50)
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Diagnosis : {prediction}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  Risk Level: {risk_level}")
    print("="*50)
    print("  WARNING: Always consult a qualified physician.")
    print("="*50)
    return {
        'prediction': prediction,
        'confidence': round(confidence, 2),
        'probability': round(probability, 4),
        'risk_level': risk_level,
        'image_path': image_path
    }

def generate_gradcam(model, image_path,
                     save_path="outputs/gradcam_sample.png"):
    os.makedirs("outputs", exist_ok=True)
    img_array = load_and_preprocess_image(image_path)
    img_batch = np.expand_dims(img_array, axis=0)
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    try:
        base_model = model.layers[1]
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                base_model.get_layer('out_relu').output,
                model.output
            ]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(
                img_batch, training=False
            )
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        if tf.math.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(
            heatmap_colored, cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(
            heatmap_colored, cv2.COLOR_BGR2RGB
        )
        superimposed = cv2.addWeighted(
            original, 0.6, heatmap_colored, 0.4, 0
        )
        prob = float(model.predict(img_batch, verbose=0)[0][0])
        diagnosis = "PNEUMONIA" if prob > 0.5 else "NORMAL"
        confidence = prob*100 if prob > 0.5 else (1-prob)*100
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Grad-CAM | Diagnosis: {diagnosis} "
            f"({confidence:.1f}% confidence)",
            fontsize=14, fontweight='bold'
        )
        axes[0].imshow(original, cmap='bone')
        axes[0].set_title("Original X-Ray")
        axes[0].axis('off')
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title("Heatmap")
        axes[1].axis('off')
        axes[2].imshow(superimposed)
        axes[2].set_title("AI Focus Region")
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Grad-CAM saved to {save_path}")
    except Exception as e:
        print(f"Grad-CAM error: {e}")