import cv2
import numpy as np
import os

# resizing images so dimensions are divisible by 2^levels
def resize_to_power_of_two(imgA, imgB, levels):
    factor = 2 ** levels
    height = min(imgA.shape[0], imgB.shape[0])
    width = min(imgA.shape[1], imgB.shape[1])

    new_h = (height // factor) * factor
    new_w = (width // factor) * factor

    imgA = cv2.resize(imgA, (new_w, new_h), interpolation=cv2.INTER_AREA)
    imgB = cv2.resize(imgB, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return imgA, imgB

# Gaussian pyramid
def build_gaussian_pyramid(image, levels, filter_1d):
    pyramid = [image.astype(np.float64)]

    kernel = filter_1d.reshape(1, -1)

    for lvl in range(1, levels):
        prev = pyramid[-1]
        h, w, c = prev.shape
        reduced = []

        for ch in range(c):
            tmp = cv2.filter2D(prev[:, :, ch], -1, kernel, borderType=cv2.BORDER_REPLICATE)
            tmp = cv2.filter2D(tmp, -1, kernel.T, borderType=cv2.BORDER_REPLICATE)
            decimated = tmp[::2, ::2]
            reduced.append(decimated)

        reduced = np.stack(reduced, axis=2)
        pyramid.append(reduced)

    return pyramid

# expanding image
def expand_image(image_small, target_h, target_w, filter_1d):
    h, w, c = image_small.shape
    expanded = np.zeros((target_h, target_w, c), dtype=np.float64)

    temp = np.zeros_like(expanded)
    temp[::2, ::2, :] = image_small

    kernel = filter_1d.reshape(1, -1)

    for ch in range(c):
        tmp = cv2.filter2D(temp[:, :, ch], -1, kernel, borderType=cv2.BORDER_REPLICATE)
        tmp = cv2.filter2D(tmp, -1, kernel.T, borderType=cv2.BORDER_REPLICATE)
        expanded[:, :, ch] = tmp[:target_h, :target_w]

    return expanded

# Laplacian pyramid
def build_laplacian_pyramid(image, levels, filt_decim, filt_interp):
    G = build_gaussian_pyramid(image, levels, filt_decim)
    L = []

    for lvl in range(levels - 1):
        expanded = expand_image(
            G[lvl + 1],
            G[lvl].shape[0],
            G[lvl].shape[1],
            filt_interp
        )
        L.append(G[lvl] - expanded)

    L.append(G[-1])
    return L


# reconstruction of the image
def reconstruct_from_laplacian(laplacian_pyramid, filt_interp):
    image = laplacian_pyramid[-1]

    for lvl in range(len(laplacian_pyramid) - 2, -1, -1):
        target = laplacian_pyramid[lvl].shape
        image = expand_image(image, target[0], target[1], filt_interp)
        image = image + laplacian_pyramid[lvl]

    return image


# normalization for visualization
def normalize_for_display(img):
    out = img.copy()
    for c in range(out.shape[2]):
        ch = out[:, :, c]
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            out[:, :, c] = (ch - mn) / (mx - mn)
        else:
            out[:, :, c] = 0
    return out


# main
def main():
    levels = 3

    filt_decim = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    filt_interp = 2 * filt_decim

    apple = cv2.imread("apple.jpeg")
    orange = cv2.imread("orange.jpeg")

    if apple is None or orange is None:
        raise FileNotFoundError("Images not found.")

    apple = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB) / 255.0
    orange = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB) / 255.0

    apple, orange = resize_to_power_of_two(apple, orange, levels)

    h, w, _ = apple.shape

    # masks
    maskA = np.zeros((h, w), dtype=np.float64)
    maskA[:, :w // 2] = 1.0
    maskB = 1.0 - maskA

    # Gaussian blur masks
    sigma = 0.05 * min(h, w)
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    maskA_blur = cv2.GaussianBlur(maskA, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    maskB_blur = cv2.GaussianBlur(maskB, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)

    # mask pyramids
    GMA = build_gaussian_pyramid(maskA_blur[:, :, None], levels, filt_decim)
    GMB = build_gaussian_pyramid(maskB_blur[:, :, None], levels, filt_decim)

    # image pyramids
    LA = build_laplacian_pyramid(apple, levels, filt_decim, filt_interp)
    LB = build_laplacian_pyramid(orange, levels, filt_decim, filt_interp)

    # combine
    LAB = []
    for lvl in range(levels):
        maskA_lvl = np.repeat(GMA[lvl], 3, axis=2)
        maskB_lvl = np.repeat(GMB[lvl], 3, axis=2)
        LAB.append(LA[lvl] * maskA_lvl + LB[lvl] * maskB_lvl)

    # reconstruct
    result = reconstruct_from_laplacian(LAB, filt_interp)
    result = np.clip(result, 0, 1)

    cv2.imwrite("blended_image.png", cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
