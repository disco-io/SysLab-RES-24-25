import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from PIL import Image

DATA_PATH = "cifar-10-batches-py"
LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_data():
    X_train, y_train = [], []
    for i in range(1, 6):
        file_path = os.path.join(DATA_PATH, f"data_batch_{i}")
        with open(file_path, "rb") as file:
            batch = pickle.load(file, encoding="bytes")
        X_train.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        y_train.append(np.array(batch[b"labels"]))
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    return X_train, y_train


def np_to_pil(img_np):
    return Image.fromarray(img_np)


def pil_to_np(img_pil):
    return np.array(img_pil)


# DITHERING & K-MEANS FUNCTIONS

K = 27


def distance(pix, mean):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pix, mean)))


def new_mean(group):
    n = len(group)
    r = sum(p[0] for p in group) / n
    g = sum(p[1] for p in group) / n
    b = sum(p[2] for p in group) / n
    return (r, g, b)


def pix_freq(PIX_LIST, IMG):
    freq = {}
    for i in range(IMG.width):
        for j in range(IMG.height):
            pix = PIX_LIST[i, j]
            freq[pix] = freq.get(pix, 0) + 1
    return freq


def classify(means, pixel_freq, curr_groups):
    new_groups = {k: [] for k in range(K)}
    for pix, freq in pixel_freq.items():
        closest_k = min(range(K), key=lambda k: distance(pix, means[k]))
        new_groups[closest_k] += [pix] * freq
    if curr_groups == {}:
        return new_groups, True
    return new_groups, any(new_groups[k] != curr_groups.get(k, []) for k in range(K))


def change_image(means, PIX_LIST, IMG):
    for i in range(IMG.width):
        for j in range(IMG.height):
            pix = PIX_LIST[i, j]
            closest_k = min(range(K), key=lambda k: distance(pix, means[k]))
            PIX_LIST[i, j] = tuple(map(int, means[closest_k]))


def k_means(IMG):
    PIX_LIST = IMG.load()
    pixel_freq_map = pix_freq(PIX_LIST, IMG)
    means = random.sample(list(pixel_freq_map.keys()), K)
    groups = {}
    while True:
        new_groups, changed = classify(means, pixel_freq_map, groups)
        means = [new_mean(group) for group in new_groups.values()]
        if not changed:
            break
        groups = new_groups
    final_means = [tuple(map(int, map(round, mean))) for mean in means]
    return final_means


def closest_color(pixel, palette):
    return min(palette, key=lambda c: distance(pixel, c))


def dither(IMG, palette):
    pix = IMG.load()
    width, height = IMG.size
    for y in range(height):
        for x in range(width):
            old_pix = pix[x, y]
            new_pix = closest_color(old_pix, palette)
            pix[x, y] = new_pix
            err = tuple(old - new for old, new in zip(old_pix, new_pix))

            def apply_error(dx, dy, factor):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    r, g, b = pix[nx, ny]
                    pix[nx, ny] = tuple(
                        max(0, min(255, int(round(v + err[i] * factor))))
                        for i, v in enumerate((r, g, b))
                    )

            apply_error(1, 0, 7 / 16)
            apply_error(-1, 1, 3 / 16)
            apply_error(0, 1, 5 / 16)
            apply_error(1, 1, 1 / 16)

    return IMG


X_train, y_train = load_data()

# Create a figure for plotting 20 images
fig, axes = plt.subplots(20, 3, figsize=(9, 60))
axes = axes.flatten()

for i in range(20):
    img_array = X_train[i]
    label = y_train[i]
    print("Label:", LABELS[label])

    IMG = np_to_pil(img_array)

    palette = k_means(IMG.copy())

    quantized_image = IMG.copy()
    change_image(palette, quantized_image.load(), quantized_image)

    dithered_image = IMG.copy()
    dithered_image = dither(dithered_image, palette)

    axes[i * 3].imshow(IMG)
    axes[i * 3].set_title(f"Original {LABELS[label]}")
    axes[i * 3 + 1].imshow(quantized_image)
    axes[i * 3 + 1].set_title(f"Quantized {LABELS[label]}")
    axes[i * 3 + 2].imshow(dithered_image)
    axes[i * 3 + 2].set_title(f"Dithered {LABELS[label]}")

    # Turn off axes for all subplots
    for ax in axes:
        ax.axis("off")

plt.tight_layout()
plt.show()
