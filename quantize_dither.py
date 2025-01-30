from PIL import Image
import random
import math
from time import perf_counter

K = 27
IMGNAME = "deer128.jpg"
IMG = Image.open(IMGNAME)
IMG_OG = IMG.copy()
PIX_LIST = IMG.load()


def demo():
    img = Image.open("puppy.jpg")
    # Img.show()
    print(img.size)
    pix = img.load()
    # pix[0, 0] is top left corner
    print(pix[2, 5])
    pix[2, 5] = (
        255,
        255,
        255,
    )
    # img.show()
    img.save("my_image.png")


def initial_means():
    inits = []
    for _ in range(K):
        pix = random.choice(
            [(PIX_LIST[i, j]) for i in range(IMG.width) for j in range(IMG.height)]
        )
        if pix not in inits:
            inits.append(pix)
    return inits


def initial_k_plusplus():
    centers = [random.choice(range(IMG.width * IMG.height))]
    unique_pixels = set([centers[0]])

    pixels = [
        IMG.getpixel((i % IMG.width, i // IMG.width))
        for i in range(IMG.width * IMG.height)
    ]

    precomp = [distance(pixel, pixels[centers[0]]) ** 2 for pixel in pixels]

    while len(centers) < K:
        weights = [dist for dist in precomp]
        new_center = random.choices(range(IMG.width * IMG.height), weights=weights)[0]
        for point in range(IMG.width * IMG.height):
            if point not in unique_pixels:
                dist = distance(pixels[point], pixels[new_center]) ** 2
                precomp[point] = min(precomp[point], dist)
                unique_pixels.add(point)

        centers.append(new_center)

    means = [pixels[index] for index in centers]

    return means


def distance(pix, mean):
    r1, r2 = pix[0], mean[0]
    g1, g2 = pix[1], mean[1]
    b1, b2 = pix[2], mean[2]
    squared_error = (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2
    return math.sqrt(squared_error)


def new_mean(group):
    num_pix = len(group)
    mr = sum(pix[0] for pix in group) / num_pix
    mg = sum(pix[1] for pix in group) / num_pix
    mb = sum(pix[2] for pix in group) / num_pix
    return (mr, mg, mb)


def pix_freq(PIX_LIST):
    pixel_freq = {}

    for i in range(IMG.width):
        for j in range(IMG.height):
            pix = PIX_LIST[i, j]
            if pix in pixel_freq:
                pixel_freq[pix] += 1
            else:
                pixel_freq[pix] = 1

    return pixel_freq


def classify(means, pixel_freq, curr_groups):
    groups_changed = False
    new_groups = {k: [] for k in range(K)}

    for pix, freq in pixel_freq.items():
        min_distance = float("inf")
        assigned_mean = None

        for k, mean in enumerate(means):
            d = distance(pix, mean)
            if d < min_distance:
                min_distance = d
                assigned_mean = k

        new_groups[assigned_mean] += [pix] * freq

    if curr_groups == {}:
        groups_changed = True
        return new_groups, groups_changed

    for k in range(K):
        if new_groups[k] != curr_groups.get(k, []):
            groups_changed = True
            break

    return new_groups, groups_changed


def change_image(means, PIX_LIST):
    for i in range(IMG.width):
        for j in range(IMG.height):
            pix = PIX_LIST[i, j]
            min_distance = float("inf")
            assigned_mean = None

            for k, mean in enumerate(means):
                d = distance(pix, mean)
                if d < min_distance:
                    min_distance = d
                    assigned_mean = k

            PIX_LIST[i, j] = means[assigned_mean]


def k_means():
    keep_going = True
    means = initial_k_plusplus()
    pixel_freq = pix_freq(PIX_LIST)
    curr_groups = {}
    count = 0
    while keep_going:
        new_groups, groups_changed = classify(means, pixel_freq, curr_groups)
        curr_groups = new_groups
        new_means = [new_mean(group) for group in new_groups.values()]
        if groups_changed:
            means = new_means
        else:
            keep_going = False
        count += 1
    final_means = [(int(round(r)), int(round(g)), int(round(b))) for (r, g, b) in means]
    return final_means


def closest_color(pixel, color_palette):
    min_dist = float("inf")
    closest = None

    for color in color_palette:
        dist = distance(pixel, color)
        if dist < min_dist:
            min_dist = dist
            closest = color

    return closest


def dither(image, color_palette):
    width, height = image.size
    pix = image.load()
    og_image = image.copy()
    og_pix = og_image.load()

    for y in range(0, height):
        for x in range(0, width):
            old_pix = pix[x, y]

            old_pix = (
                max(0, min(255, old_pix[0])),
                max(0, min(255, old_pix[1])),
                max(0, min(255, old_pix[2])),
            )

            new_pix = closest_color(old_pix, color_palette)
            error_r, error_g, error_b = (
                old_pix[0] - new_pix[0],
                old_pix[1] - new_pix[1],
                old_pix[2] - new_pix[2],
            )

            pix[x, y] = new_pix
            new_pix = closest_color(old_pix, color_palette)

            pix[x, y] = new_pix
            error_r, error_g, error_b = (
                old_pix[0] - new_pix[0],
                old_pix[1] - new_pix[1],
                old_pix[2] - new_pix[2],
            )

            pix[x, y] = new_pix
            if 0 <= x + 1 < width and 0 <= y < height:
                pix[x + 1, y] = (
                    round(pix[x + 1, y][0] + error_r * 7 / 16),
                    round(pix[x + 1, y][1] + error_g * 7 / 16),
                    round(pix[x + 1, y][2] + error_b * 7 / 16),
                )
            if 0 <= x + 1 < width and 0 <= y + 1 < height:
                pix[x + 1, y + 1] = (
                    round(pix[x + 1, y + 1][0] + error_r * 1 / 16),
                    round(pix[x + 1, y + 1][1] + error_g * 1 / 16),
                    round(pix[x + 1, y + 1][2] + error_b * 1 / 16),
                )
            if 0 <= x < width and 0 <= y + 1 < height:
                pix[x, y + 1] = (
                    round(pix[x, y + 1][0] + error_r * 5 / 16),
                    round(pix[x, y + 1][1] + error_g * 5 / 16),
                    round(pix[x, y + 1][2] + error_b * 5 / 16),
                )
            if 0 <= x - 1 < width and 0 <= y + 1 < height:
                pix[x - 1, y + 1] = (
                    round(pix[x - 1, y + 1][0] + error_r * 3 / 16),
                    round(pix[x - 1, y + 1][1] + error_g * 3 / 16),
                    round(pix[x - 1, y + 1][2] + error_b * 3 / 16),
                )
    return image


# QUANTIZE
q_start = perf_counter()
final_means = k_means()
change_image(final_means, PIX_LIST)
q_end = perf_counter()

# DITHER
d_start = perf_counter()
IMG = dither(IMG_OG, final_means)
d_end = perf_counter()

IMG.show()
IMG.save(IMGNAME[:-4] + "_" + str(K) + "_out.png")

print(K, IMGNAME)
print("Quantize Time: ", q_end - q_start)
print("Dithering Time: ", d_end - d_start)
