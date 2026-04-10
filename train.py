import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(train_gen):
    classes = train_gen.classes
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weights = {0: weights[0], 1: weights[1]}
    print(f"Class Weights - NORMAL: {class_weights[0]:.4f}, PNEUMONIA: {class_weights[1]:.4f}")
    return class_weights

def get_callbacks(phase=1):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            filepath=f"models/best_model_phase{phase}.keras",
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(
            filename=f"logs/training_phase{phase}.csv",
            separator=',',
            append=False
        ),
    ]
    return callbacks

def train_phase1(model, train_gen, val_gen, epochs=20):
    print("\n" + "="*50)
    print("PHASE 1 - Training Classification Head")
    print("="*50)
    class_weights = get_class_weights(train_gen)
    callbacks = get_callbacks(phase=1)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    print("Phase 1 complete!")
    return history

def train_phase2(model, train_gen, val_gen, epochs=10):
    print("\n" + "="*50)
    print("PHASE 2 - Fine Tuning MobileNetV2")
    print("="*50)
    class_weights = get_class_weights(train_gen)
    callbacks = get_callbacks(phase=2)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    print("Phase 2 complete!")
    return history

def plot_training_history(history1, history2=None):
    os.makedirs("outputs", exist_ok=True)
    acc = history1.history['accuracy']
    val_acc = history1.history['val_accuracy']
    loss = history1.history['loss']
    val_loss = history1.history['val_loss']

    if history2:
        acc += history2.history['accuracy']
        val_acc += history2.history['val_accuracy']
        loss += history2.history['loss']
        val_loss += history2.history['val_loss']

    epochs = range(1, len(acc) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, acc, 'b-o', label='Training', markersize=4)
    axes[0].plot(epochs, val_acc, 'r-o', label='Validation', markersize=4)
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, loss, 'b-o', label='Training', markersize=4)
    axes[1].plot(epochs, val_loss, 'r-o', label='Validation', markersize=4)
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/training_curves.png', dpi=150)
    plt.close()
    print("Training curves saved to outputs/training_curves.png")