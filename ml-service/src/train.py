"""Train a PCA (Eigenfaces) + SVM classifier for the Schwarzenegger-vs-others
face verification problem, with hyperparameter search, imbalance-aware
evaluation, and MLflow experiment tracking.
"""
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from data import RANDOM_STATE, augment_positive_class, load_binary_face_dataset, split_dataset

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

# Two families of hyperparameters searched together: RBF (can fit curved
# boundaries but more prone to overfitting a small minority class) and
# linear (simpler decision boundary, often generalizes better when the
# feature count is high relative to sample count -- our exact situation).
PARAM_GRID = [
    {
        "pca__n_components": [30, 50, 100],
        "svm__kernel": ["rbf"],
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", 0.01],
    },
    {
        "pca__n_components": [30, 50, 100],
        "svm__kernel": ["linear"],
        "svm__C": [0.1, 1, 10],
    },
]


def build_pipeline():
    return Pipeline(
        [
            ("pca", PCA(whiten=True, random_state=RANDOM_STATE)),
            (
                "svm",
                SVC(
                    class_weight="balanced",  # up-weights the rare positive class
                    probability=True,  # needed for ROC/PR-AUC and confidence scores
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def plot_eigenfaces(pca, image_shape, n=12):
    """Reshape the top-n principal components back into image form and
    save a grid figure. These are the 'eigenfaces' -- the directions of
    greatest variance across all training faces.
    """
    n = min(n, pca.components_.shape[0])
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.3 * rows))
    for i, ax in enumerate(axes.ravel()):
        if i < n:
            ax.imshow(pca.components_[i].reshape(image_shape), cmap="gray")
            ax.set_title(f"eigenface {i + 1}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(ARTIFACTS_DIR, "eigenfaces.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["other", "schwarzenegger"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path, cm


def select_threshold(pipeline_template, X_train, y_train, groups, cv):
    """Pick a classification threshold via out-of-fold predictions on the
    training set, instead of trusting SVC's default 0.5 cutoff.

    Why this is necessary: with a 71:1 class imbalance, the default
    decision threshold can (and here, does) end up predicting the
    majority class for every single example -- 0 recall -- even though
    the model's *ranking* of examples (ROC-AUC, PR-AUC) is reasonably
    good. The fix isn't a better model, it's picking a better cutoff on
    the score the model already produces.

    Why out-of-fold and not just "pick whatever looks best on the test
    set": that would leak test-set information into a modeling decision
    and make the reported test metrics optimistic. cross_val_predict
    refits `pipeline_template` on each training fold and predicts on the
    held-out fold, so every probability used here comes from a model
    that never saw that sample -- the test set stays untouched until
    final reporting. `groups` (see augment_positive_class) keeps
    augmented siblings of the same source photo together in one fold, so
    this stays a fair estimate rather than partial memorization of
    near-duplicates.
    """
    oof_proba = cross_val_predict(
        clone(pipeline_template), X_train, y_train, groups=groups, cv=cv,
        method="predict_proba", n_jobs=-1,
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, oof_proba)
    # precision/recall have one more element than thresholds (the last
    # point is "classify everything as positive", which has no threshold).
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_idx = int(np.argmax(f1))
    best_threshold = float(thresholds[best_idx])

    print("Operating points from out-of-fold training predictions:")
    print(f"{'threshold':>10}  {'precision':>10}  {'recall':>10}  {'f1':>10}")
    # Print a handful of points spanning the curve for interview-ready discussion.
    sample_idxs = np.linspace(0, len(thresholds) - 1, min(8, len(thresholds))).astype(int)
    for i in sample_idxs:
        marker = "  <- chosen" if i == best_idx else ""
        print(f"{thresholds[i]:>10.3f}  {precision[i]:>10.3f}  {recall[i]:>10.3f}  {f1[i]:>10.3f}{marker}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall[:-1], precision[:-1])
    ax.scatter([recall[best_idx]], [precision[best_idx]], color="red", zorder=5, label="chosen threshold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (out-of-fold, train)")
    ax.legend()
    pr_curve_path = os.path.join(ARTIFACTS_DIR, "pr_curve.png")
    fig.savefig(pr_curve_path, dpi=120)
    plt.close(fig)

    return best_threshold, pr_curve_path


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X, y, image_shape = load_binary_face_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    n_positive_before = int(y_train.sum())
    X_train, y_train, groups = augment_positive_class(X_train, y_train, image_shape)
    print(
        f"Augmented positive class in training set: {n_positive_before} -> {int(y_train.sum())} "
        f"(test set untouched: {int(y_test.sum())} positives)"
    )

    pipeline = build_pipeline()
    # StratifiedGroupKFold, not plain StratifiedKFold: preserves the
    # positive/negative ratio per fold (Stratified) while also keeping
    # every augmented sibling of one source photo in the same fold
    # (Group) -- see augment_positive_class for why the latter matters.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # average_precision (PR-AUC) is a better model-selection metric than
    # accuracy or even ROC-AUC when positives are this rare, because it's
    # sensitive to precision at low recall -- exactly where an imbalanced
    # classifier tends to struggle.
    grid = GridSearchCV(
        pipeline, PARAM_GRID, scoring="average_precision", cv=cv, n_jobs=-1, verbose=1
    )

    mlflow.set_experiment("face-verification-schwarzenegger")
    with mlflow.start_run():
        grid.fit(X_train, y_train, groups=groups)
        best_model = grid.best_estimator_

        mlflow.log_params(grid.best_params_)
        mlflow.log_param("min_faces_per_other_person", 20)
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("n_test", len(y_test))
        mlflow.log_param("n_positive_train_before_augmentation", n_positive_before)
        mlflow.log_param("n_positive_train_after_augmentation", int(y_train.sum()))
        mlflow.log_param("n_positive_test", int(y_test.sum()))

        print(f"Best params: {grid.best_params_}")
        print(f"Best CV average_precision: {grid.best_score_:.4f}")
        print()

        threshold, pr_curve_path = select_threshold(
            build_pipeline().set_params(**grid.best_params_), X_train, y_train, groups, cv
        )
        mlflow.log_param("decision_threshold", round(threshold, 4))
        mlflow.log_artifact(pr_curve_path)

        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred_default = best_model.predict(X_test)  # SVC's built-in 0.5-equivalent cutoff
        y_pred = (y_proba >= threshold).astype(int)  # our tuned operating point

        metrics = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),  # threshold-independent
            "pr_auc": average_precision_score(y_test, y_proba),  # threshold-independent
            "recall_at_default_threshold": recall_score(y_test, y_pred_default, zero_division=0),
        }
        mlflow.log_metrics(metrics)

        print(f"Chosen decision threshold: {threshold:.4f}")
        print()
        print("-- At SVC's default cutoff (for comparison) --")
        print(classification_report(y_test, y_pred_default, target_names=["other", "schwarzenegger"], zero_division=0))
        print("-- At the tuned threshold --")
        print(classification_report(y_test, y_pred, target_names=["other", "schwarzenegger"], zero_division=0))
        print("Test metrics (tuned threshold):", {k: round(v, 4) for k, v in metrics.items()})

        cm_path, cm = plot_confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", cm)
        mlflow.log_artifact(cm_path)

        eigenfaces_path = plot_eigenfaces(best_model.named_steps["pca"], image_shape)
        mlflow.log_artifact(eigenfaces_path)

        mlflow.sklearn.log_model(best_model, "model")

        # Save locally too, so the FastAPI service can load it without
        # needing an MLflow tracking server at inference time. The
        # threshold travels with the model -- at inference we must use
        # the same tuned cutoff, not scikit-learn's default.
        model_path = os.path.join(MODELS_DIR, "face_classifier.joblib")
        joblib.dump(
            {"model": best_model, "image_shape": image_shape, "threshold": threshold},
            model_path,
        )
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
