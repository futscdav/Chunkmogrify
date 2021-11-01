"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from 
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html

requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from: 
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

import PIL
import PIL.Image
import os
import scipy
import scipy.ndimage
import dlib
import cv2
import numpy as np

from PIL import Image
from argparse import ArgumentParser

# download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor('./resources/shape_predictor_68_face_landmarks.dat')

def get_landmark(filepath):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)

    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))


    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    # lm is a shape=(68,2) np.array
    return lm

    
def align_face(filepath):
    """
    :param filepath: str
    :return: PIL Image
    """

    lm = get_landmark(filepath)
    
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2


    # read image
    img = PIL.Image.open(filepath)

    output_size=1024
    transform_size=4096
    enable_padding=True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    return img


def get_landmark_npy(img):
    """get landmark with dlib
        :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    if len(dets) == 0:
        raise RuntimeError("No faces found")

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
    

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    # lm is a shape=(68,2) np.array
    return lm

def align_face_npy(img, output_size=1024):
    lm = get_landmark_npy(img)
    
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2


    img = Image.fromarray(img)

    transform_size=4096
    enable_padding=True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    return np.array(img)


def align_face_npy_with_params(img, output_size=1024):
    lm = get_landmark_npy(img)
    
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2


    img = Image.fromarray(img)

    transform_size=4096
    enable_padding=True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    shrunk_image = img

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    actual_crop = (0, 0, 0, 0)
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        actual_crop = crop
        img = img.crop(crop)
        quad -= crop[0:2]

    # # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    actual_padding = (0, 0, 0, 0)
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        actual_padding = pad
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    padded_img = img

    # # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    return np.array(img), [shrink, actual_crop, actual_padding, quad, padded_img, shrunk_image]


def unalign_face_npy(aligned_image, alignment_params):
    # Shrinking of the original image means that the face was too large to be represented
    # in the output size anyway, so it doesn't make sense to reverse it.

    shrink, crop, padding, quad, padded_img, shrunk_image = alignment_params

    def build_perspective(srcpts, dstpts):
        srcpts = np.array(srcpts)
        dstpts = np.array(dstpts)
        A = \
        [
            # x1
            [srcpts[0, 0], srcpts[0, 1], 1, 0, 0, 0, -srcpts[0, 0] * dstpts[0, 0], -srcpts[0, 1] * dstpts[0, 0]],
            [0, 0, 0, srcpts[0, 0], srcpts[0, 1], 1, -srcpts[0, 0] * dstpts[0, 1], -srcpts[0, 1] * dstpts[0, 1]],
            # x2
            [srcpts[1, 0], srcpts[1, 1], 1, 0, 0, 0, -srcpts[1, 0] * dstpts[1, 0], -srcpts[1, 1] * dstpts[1, 0]],
            [0, 0, 0, srcpts[1, 0], srcpts[1, 1], 1, -srcpts[1, 0] * dstpts[1, 1], -srcpts[1, 1] * dstpts[1, 1]],
            # x3
            [srcpts[2, 0], srcpts[2, 1], 1, 0, 0, 0, -srcpts[2, 0] * dstpts[2, 0], -srcpts[2, 1] * dstpts[2, 0]],
            [0, 0, 0, srcpts[2, 0], srcpts[2, 1], 1, -srcpts[2, 0] * dstpts[2, 1], -srcpts[2, 1] * dstpts[2, 1]],
            # x4
            [srcpts[3, 0], srcpts[3, 1], 1, 0, 0, 0, -srcpts[3, 0] * dstpts[3, 0], -srcpts[3, 1] * dstpts[3, 0]],
            [0, 0, 0, srcpts[3, 0], srcpts[3, 1], 1, -srcpts[3, 0] * dstpts[3, 1], -srcpts[3, 1] * dstpts[3, 1]],
        ]
        b = [dstpts[0, 0], dstpts[0, 1], dstpts[1, 0], dstpts[1, 1], dstpts[2, 0], dstpts[2, 1], dstpts[3, 0], dstpts[3, 1]]

        coeffs = np.linalg.solve(np.array(A), np.array(b))
        xform = \
            [
                [coeffs[0], coeffs[1], coeffs[2]],
                [coeffs[3], coeffs[4], coeffs[5]],
                [coeffs[6], coeffs[7], 1]
            ]
        return np.array(xform)

    # Transform back to the unaligned quad.
    c = build_perspective(
        [[0, 0], [0, 1024], [1024, 1024], [1024, 0]],
        quad + 0.5, 
        )
    c = np.linalg.inv(c)

    aligned_pil = PIL.Image.fromarray(aligned_image)
    fill_mask = PIL.Image.fromarray(np.ones_like(aligned_image, dtype=np.uint8) * 255)
    # Inverse to `unaligned = aligned_pil.transform((1024, 1024), PIL.Image.PERSPECTIVE, c.reshape(9)[0:8], Image.BICUBIC)``
    unaligned = aligned_pil.transform(
        (padded_img.width, padded_img.height), 
        Image.PERSPECTIVE, c.reshape(9)[0:8], Image.BICUBIC
    )
    unaligned_mask = fill_mask.transform(
        (padded_img.width, padded_img.height),
        Image.PERSPECTIVE, c.reshape(9)[0:8], Image.BICUBIC
    )
    # "Unpad"
    unaligned = np.array(unaligned)[padding[1]:unaligned.height-padding[3], padding[0]:unaligned.width-padding[2], :]
    unaligned_mask = np.array(unaligned_mask)[padding[1]:unaligned_mask.height-padding[3], padding[0]:unaligned_mask.width-padding[2], :]
    # Ideally get rid of the blur added with padding, but that's not as trivial..

    # Uncrop.
    canvas = np.empty((shrunk_image.height, shrunk_image.width, unaligned.shape[2]), dtype=unaligned.dtype)
    mask = np.zeros((shrunk_image.height, shrunk_image.width, unaligned_mask.shape[2]), dtype=unaligned_mask.dtype)

    if crop[0] == 0 and crop[1] == 0 and crop[2] == 0 and crop[3] == 0:
        # no crop, TODO: split x and y
        canvas = unaligned
        mask = unaligned_mask
    else:
        canvas[crop[1]:crop[3], crop[0]:crop[2], :] = unaligned
        mask[crop[1]:crop[3], crop[0]:crop[2], :] = unaligned_mask
    mask = mask[:, :, 0:1]

    x, y, w, h = cv2.boundingRect(mask)
    unaligned = cv2.seamlessClone(canvas, np.array(shrunk_image), mask, (int(x + w / 2), int(y + h /2)), 1)

    aligned = np.array(unaligned)
    return aligned

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if os.path.exists(args.input):
        if args.output is None:
            d, b = os.path.split(args.input)
            args.output = os.path.join(d, os.path.splitext(b)[0] + '_aligned' + os.path.splitext(b)[1])

        aligned = align_face(args.input)
        aligned.save(args.output)