"""Load the LFW subset used for this project and produce a binary-labeled,
stratified train/test split: 1 = Arnold Schwarzenegger, 0 = anyone else.
"""
import sys

import numpy as np
from scipy.ndimage import rotate
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

from inference import NoFaceDetectedError, detect_and_crop_face, preprocess_face

TARGET_NAME = "Arnold Schwarzenegger"
MIN_FACES_PER_OTHER_PERSON = 20  # keeps the "other" pool to well-represented people
TEST_SIZE = 0.2
RANDOM_STATE = 42  # fixed seed so results are reproducible run-to-run
TARGET_IMAGE_SHAPE = (50, 37)  # final size fed to the classifier


def load_binary_face_dataset():
    """Return (X, y, image_shape) for the Schwarzenegger-vs-others problem.

    X: (n_samples, n_features) flattened grayscale face images.
    y: (n_samples,) binary labels, 1 = TARGET_NAME, 0 = everyone else.
    image_shape: (height, width) of a single image, for reshaping later.

    Important: every image is run through the *same* detect_and_crop_face
    + preprocess_face functions the FastAPI service uses at inference
    time (src/inference.py), rather than using LFW's own pre-baked
    alignment (sklearn's fetch_lfw_people applies a fixed internal crop
    before resizing). Earlier this project trained on LFW's alignment
    but served predictions through a separately-parameterized Haar
    cascade crop -- classic training/serving skew. Since PCA/Eigenfaces
    operates directly on pixel positions (it is not translation- or
    scale-invariant), even a same-identity photo scores very differently
    under two different alignments -- confirmed directly: a live
    Schwarzenegger test photo scored a near-zero match confidence before
    this fix. Routing training data through the exact inference
    pipeline eliminates that mismatch by construction.
    """
    raw = fetch_lfw_people(min_faces_per_person=MIN_FACES_PER_OTHER_PERSON, resize=1.0, color=True)

    names = raw.target_names
    target_idx = np.where(names == TARGET_NAME)[0]
    if len(target_idx) == 0:
        raise RuntimeError(
            f"{TARGET_NAME!r} not found in LFW at "
            f"min_faces_per_person={MIN_FACES_PER_OTHER_PERSON}"
        )
    target_label = target_idx[0]

    features, labels = [], []
    n_dropped = 0
    for image_float, label in zip(raw.images, raw.target):
        image_bgr = (image_float[:, :, ::-1] * 255).astype(np.uint8)  # RGB->BGR, [0,1]->[0,255]
        try:
            face_gray = detect_and_crop_face(image_bgr)
        except NoFaceDetectedError:
            n_dropped += 1
            continue
        features.append(preprocess_face(face_gray, TARGET_IMAGE_SHAPE)[0])
        labels.append(1 if label == target_label else 0)

    print(
        f"Ran {len(raw.images)} images through the inference detection pipeline: "
        f"{n_dropped} dropped (no face detected), {len(features)} kept.",
        file=sys.stderr,
    )

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=int)
    return X, y, TARGET_IMAGE_SHAPE


def split_dataset(X, y):
    """Stratified train/test split preserving the positive class ratio."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def augment_positive_class(X_train, y_train, image_shape):
    """Expand the minority (Schwarzenegger) class in the *training* split
    only, via horizontal flips and small rotations. Also returns a
    `groups` array identifying which original source photo each row
    came from -- see the warning below for why this matters.

    Why augment at all: 42 total images (~34 in train after the split)
    is a small sample for an SVM operating on 1850-dimensional pixel
    vectors. Collecting more real photos isn't an option here, so we
    synthesize plausible variations of the ones we have -- a face
    flipped or tilted a few degrees is still recognizably the same
    identity, and gives the classifier more to generalize from. This
    must only ever touch the training split: augmenting before the
    train/test split would leak near-duplicate images across the split
    and inflate test performance.

    Why `groups` matters: augmented siblings of the same source photo
    (e.g. the original and its horizontal flip) are near-duplicates. If
    a later cross-validation split puts one sibling in a training fold
    and another in that fold's validation set, the model can partly
    "recognize" the near-duplicate instead of the underlying identity,
    which inflates cross-validated scores without reflecting real
    generalization. Callers should use these group IDs with
    StratifiedGroupKFold (not plain StratifiedKFold) so every variant of
    one source photo always lands in the same fold.
    """
    pos_mask = y_train == 1
    pos_images = X_train[pos_mask].reshape(-1, *image_shape)
    n_pos_originals = pos_images.shape[0]

    variants = [pos_images]  # start with the originals
    variants.append(pos_images[:, :, ::-1])  # horizontal flip
    for angle in (-8, 8):
        rotated = rotate(pos_images, angle, axes=(1, 2), reshape=False, mode="nearest")
        variants.append(rotated)
    # flipped + rotated, for a bit more variety
    flipped = pos_images[:, :, ::-1]
    variants.append(rotate(flipped, 5, axes=(1, 2), reshape=False, mode="nearest"))

    X_pos_aug = np.concatenate(variants, axis=0).reshape(-1, np.prod(image_shape))
    y_pos_aug = np.ones(X_pos_aug.shape[0], dtype=int)
    # Each of the 5 variant blocks reuses the same n_pos_originals source
    # indices, so tiling the group IDs ties every sibling back together.
    n_variant_blocks = len(variants)
    pos_groups = np.tile(np.arange(n_pos_originals), n_variant_blocks)

    # "Other" class rows weren't duplicated, so each gets its own unique
    # group -- offset so these IDs never collide with the positive groups.
    n_other = int((~pos_mask).sum())
    other_groups = np.arange(n_pos_originals, n_pos_originals + n_other)

    X_train_aug = np.vstack([X_train[~pos_mask], X_pos_aug])
    y_train_aug = np.concatenate([y_train[~pos_mask], y_pos_aug])
    groups_aug = np.concatenate([other_groups, pos_groups])

    # Shuffle so augmented positives aren't all clustered at the end
    # (matters for cross-validation fold assignment upstream).
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(len(y_train_aug))
    return X_train_aug[perm], y_train_aug[perm], groups_aug[perm]


if __name__ == "__main__":
    X, y, image_shape = load_binary_face_dataset()
    print(f"X shape: {X.shape}, y shape: {y.shape}, image_shape: {image_shape}")
    print(f"Positive (Schwarzenegger) count: {y.sum()} / {len(y)}")

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    train_ratio = y_train.sum() / len(y_train)
    test_ratio = y_test.sum() / len(y_test)
    print(f"Train: {X_train.shape[0]} samples, {y_train.sum()} positive ({train_ratio:.4f})")
    print(f"Test:  {X_test.shape[0]} samples, {y_test.sum()} positive ({test_ratio:.4f})")
