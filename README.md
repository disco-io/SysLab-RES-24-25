# Color Quantization of CIFAR-10 Image Dataset for Image Classification using CNNs

![Poster](https://raw.githubusercontent.com/disco-io/SysLab-RES-24-25/main/presentation/Poster.png)

---

## TJHSST Computer Systems Research Lab with Dr. Selma Yilmaz

Convolutional Neural Networks (CNNs) are highly effective for image classification but often require significant computational resources, making them difficult to deploy on low-power devices. This project introduces Mini Color NN, a novel approach that combines k-means vector quantization and Floyd-Steinberg dithering to reduce the color complexity of images in the CIFAR-10 dataset while preserving critical visual information. The resulting Mini Color CIFAR-10 dataset reduces storage requirements by 1.7× and accelerates training time by 1.3×, with only a minor drop in classification accuracy (from 92.88% to 92.01%). These results demonstrate that significant computational efficiency can be achieved without compromising CNN performance, supporting broader deployment of deep learning models in edge computing and resource-constrained environments. This work contributes a publicly available, lightweight dataset and offers a practical preprocessing strategy for efficient image-based machine learning.

---

## How to Run Code

After cloning and `cd` into the repository, run `CIFAR10-CNN.py` and wait a few minutes for training to begin. To use the Original Dataset, set `DATA_PATH` to `"cifar-10-batches-py"` or `"cifar-10-custom"` for the Mini Color Version.

Quit the program at any time to view training progress across epochs. To resume, run the Python file again and follow instructions in the terminal.
