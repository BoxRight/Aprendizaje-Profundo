import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path

# Explicit splits from train_explicit.py
TEST_PERSONS = [8, 14, 20, 42, 58, 61, 65]
ACTIVITY_NAMES = {
    0: "Walk",
    1: "Sit",
    2: "Stand",
    3: "Pick",
    4: "Drink",
    5: "Fall",
}

def evaluate_detailed():
    # Load data
    data = np.load("processed/full_dataset.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    persons = data["persons"].astype(np.int32)
    
    if X.ndim == 3:
        X = X[..., np.newaxis]
        
    # Filter test set
    test_mask = np.isin(persons, TEST_PERSONS)
    X_test = X[test_mask]
    y_test = y[test_mask]
    p_test = persons[test_mask]
    
    # Load model
    model = tf.keras.models.load_model("models_explicit/best_model_explicit.keras")
    
    # Predict
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 1. Multiclass Classification Report
    report = classification_report(y_test, y_pred, target_names=[ACTIVITY_NAMES[i] for i in range(6)], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # 2. Binary Fall Analysis (Class 5 is Fall)
    y_test_bin = (y_test == 5).astype(int)
    y_pred_bin = (y_pred == 5).astype(int)
    
    bin_report = classification_report(y_test_bin, y_pred_bin, target_names=["Non-Fall", "Fall"], output_dict=True)
    bin_cm = confusion_matrix(y_test_bin, y_pred_bin)
    bin_acc = (y_test_bin == y_pred_bin).mean()
    
    # 3. Subject Variability
    subject_f1s = []
    subject_fall_f1s = []
    unique_subjects = np.unique(p_test)
    
    for s in unique_subjects:
        s_mask = (p_test == s)
        s_y_true = y_test[s_mask]
        s_y_pred = y_pred[s_mask]
        
        # Multiclass F1 for subject
        s_report = classification_report(s_y_true, s_y_pred, labels=range(6), output_dict=True, zero_division=0)
        subject_f1s.append(s_report['macro avg']['f1-score'])
        
        # Binary Fall F1 for subject
        s_y_true_bin = (s_y_true == 5).astype(int)
        s_y_pred_bin = (s_y_pred == 5).astype(int)
        if np.any(s_y_true_bin == 1):
            s_bin_report = classification_report(s_y_true_bin, s_y_pred_bin, labels=[0, 1], output_dict=True, zero_division=0)
            subject_fall_f1s.append(s_bin_report['1']['f1-score'])
    
    # Aggregate results
    results = {
        "multiclass": report,
        "confusion_matrix": cm.tolist(),
        "binary_fall": {
            "accuracy": bin_acc,
            "report": bin_report,
            "confusion_matrix": bin_cm.tolist()
        },
        "subject_variability": {
            "subjects": unique_subjects.tolist(),
            "macro_f1_mean": np.mean(subject_f1s),
            "macro_f1_std": np.std(subject_f1s),
            "fall_f1_mean": np.mean(subject_fall_f1s) if subject_fall_f1s else 0,
            "fall_f1_std": np.std(subject_fall_f1s) if subject_fall_f1s else 0
        }
    }
    
    # Save to JSON for easy reading
    with open("models_explicit/detailed_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Detailed evaluation saved to models_explicit/detailed_evaluation.json")

if __name__ == "__main__":
    evaluate_detailed()
