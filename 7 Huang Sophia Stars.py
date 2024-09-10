import sys
import math
import random

# submitting spec
i1, i2, i3, i4, i5, i6 = (
    int(sys.argv[1]),
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
    int(sys.argv[6]),
)


import math

RAW_LIST = []
STARS_LIST = []
STARS_SORTED = {}

with open("star_data.csv", "r") as file:
    next(file)
    for line in file:
        values = line.strip().split(",")
        temperature = math.log(float(values[0]))
        luminosity = math.log(float(values[1]))
        radius = math.log(float(values[2]))
        absolute_magnitude = float(values[3])
        star_type = int(values[4])
        star_color = str(values[5])
        spectral_class = str(values[6])

        RAW_LIST.append(
            (
                temperature,
                luminosity,
                radius,
                absolute_magnitude,
                star_type,
                star_color,
                spectral_class,
            )
        )

        star = (temperature, luminosity, radius, absolute_magnitude)
        STARS_LIST.append(star)

        if star_type not in STARS_SORTED:
            STARS_SORTED[star_type] = []
        STARS_SORTED[star_type].append(star)

K = 6


def initial_means():
    inits = []
    for _ in range(K):
        inits.append(random.choice(STARS_LIST))
    return inits


def distance(star, mean):
    t1, t2 = star[0], mean[0]
    l1, l2 = star[1], mean[1]
    r1, r2 = star[2], mean[2]
    a1, a2 = star[3], mean[3]
    squared_error = (t1 - t2) ** 2 + (l1 - l2) ** 2 + (r1 - r2) ** 2 + (a1 - a2) ** 2
    return math.sqrt(squared_error)


def new_mean(group):
    num_stars = len(group)
    mt = sum(star[0] for star in group) / num_stars
    ml = sum(star[1] for star in group) / num_stars
    mr = sum(star[2] for star in group) / num_stars
    ma = sum(star[3] for star in group) / num_stars
    return (mt, ml, mr, ma)


def classify(means, STARS_LIST, curr_groups):
    groups_changed = False
    new_groups = {k: [] for k in range(K)}

    for _, star in enumerate(STARS_LIST):
        min_distance = float("inf")
        assigned_mean = None

        for j, mean in enumerate(means):
            d = distance(star, mean)
            if d < min_distance:
                min_distance = d
                assigned_mean = j

        new_groups[assigned_mean].append(star)

    if curr_groups == {}:
        groups_changed = True
        return new_groups, groups_changed

    for k in range(K):
        if new_groups[k] != curr_groups.get(k, []):
            groups_changed = True
            break

    return new_groups, groups_changed


def k_means():
    keep_going = True
    # means = initial_means()
    means = [
        STARS_LIST[i1],
        STARS_LIST[i2],
        STARS_LIST[i3],
        STARS_LIST[i4],
        STARS_LIST[i5],
        STARS_LIST[i6],
    ]
    curr_groups = {}
    count = 0
    while keep_going:
        new_groups, groups_changed = classify(means, STARS_LIST, curr_groups)
        curr_groups = new_groups
        new_means = [new_mean(group) for group in new_groups.values()]
        if groups_changed:
            means = new_means
        else:
            keep_going = False
        count += 1
    for k, mean in enumerate(means):
        print("\nMean:", mean)
        for star in curr_groups[k]:
            star_type = next(
                key for key, value in STARS_SORTED.items() if star in value
            )
            print(star, "star Type:", star_type)
    print("\nk-means ended with", count, "iterations")


k_means()
