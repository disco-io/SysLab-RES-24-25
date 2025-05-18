import os
import pickle
import numpy as np
import random
import math
from PIL import Image
from tqdm import tqdm  # progress bar

DATA_PATH = "cifar-10-batches-py"
OUT_PATH = "cifar-10-custom"
os.makedirs(OUT_PATH, exist_ok=True)

K = 27


def distance(pix, mean):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pix, mean)))


def new_mean(group):
    n = len(group)
    if n == 0:
        return (0, 0, 0)
    r = sum(p[0] for p in group) / n
    g = sum(p[1] for p in group) / n
    b = sum(p[2] for p in group) / n
    return (r, g, b)


def pix_freq(pix_list, img):
    freq = {}
    for i in range(img.width):
        for j in range(img.height):
            pix = pix_list[i, j]
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


def change_image(means, pix_list, img):
    for i in range(img.width):
        for j in range(img.height):
            pix = pix_list[i, j]
            closest_k = min(range(K), key=lambda k: distance(pix, means[k]))
            pix_list[i, j] = tuple(map(int, means[closest_k]))


def k_means(img):
    pix_list = img.load()
    pixel_freq_map = pix_freq(pix_list, img)
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


def dither(img, palette):
    pix = img.load()
    width, height = img.size
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

    return img


def process_and_save_batch(input_file, output_file):
    with open(input_file, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    data = batch[b"data"]
    labels = batch[b"labels"]
    filenames = batch[b"filenames"]

    new_data = []

    for i in tqdm(range(len(data)), desc=f"Processing {os.path.basename(input_file)}"):
        if i % 100 == 0:
            print(f"  Sanity check: {i}/{len(data)} images processed.")

        img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        pil_img = Image.fromarray(img)

        palette = k_means(pil_img.copy())
        quantized = pil_img.copy()
        change_image(palette, quantized.load(), quantized)

        dithered = dither(quantized.copy(), palette)

        dithered_np = np.array(dithered).transpose(2, 0, 1).reshape(-1)
        new_data.append(dithered_np)

    out_batch = {
        b"data": np.stack(new_data),
        b"labels": labels,
        b"filenames": filenames,
        b"batch_label": batch[b"batch_label"],
    }

    with open(output_file, "wb") as f:
        pickle.dump(out_batch, f)


# Process training batches
for i in range(1, 6):
    in_file = os.path.join(DATA_PATH, f"data_batch_{i}")
    out_file = os.path.join(OUT_PATH, f"data_batch_{i}")
    process_and_save_batch(in_file, out_file)

# Copy test_batch and meta file (unchanged)
for fname in ["test_batch", "batches.meta"]:
    in_path = os.path.join(DATA_PATH, fname)
    out_path = os.path.join(OUT_PATH, fname)
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        fout.write(fin.read())

print("All batches processed and saved to 'cifar-10-custom'")
