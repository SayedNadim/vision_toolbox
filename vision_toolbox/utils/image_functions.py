# =================================================================================================
#                            Image (Array) Helper functions                                       #
# =================================================================================================
# for image reading purpose
import cv2
import skimage.io as sio
import imageio
import numpy as np
from PIL import Image
import skimage.feature as feature
import matplotlib.pyplot as plt


# array image functions
def allowed_image_extensions(filename):
    """
    Returns image files if they have the allowed image extensions
    :param filename: image file
    :return: image file
    """
    img_ext = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tiff', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in img_ext)


def image_reader(path, is_color=True, lib='skimage'):
    """
    Reads an image from the given path and returns the color or gray image using preferred library.
    :param path: Path of the image file
    :param is_color: Condition for the image is to be read as color image or grayscale image
    :param lib: Preferred image reading library. Available: cv2, skimage, pil and imageion. Note that, color image from
                cv2 does not comply with other libraries.
    :return: image value (numpy array)
    """
    if lib == 'skimage':
        if is_color:
            img = sio.imread(path)
        else:
            img = sio.imread(path, as_gray=True)
            img = img[:, :, np.newaxis]
    # TODO make cv2 image = image from other libraries. Currently all libraries match except cv2.
    #  I think, this is due to my implementation issue, maybe...
    elif lib == 'cv2':
        if is_color:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img[:, :, np.newaxis]
    elif lib == 'imageio':
        if is_color:
            img = imageio.imread(path)
        else:
            img = imageio.imread(path)
            gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
            img = gray(img)
            img = img[:, :, np.newaxis]
    elif lib == 'pil':
        if is_color:
            img = np.array(Image.open(path))
        else:
            img = np.array(Image.open(path).convert('L'))
            img = img[:, :, np.newaxis]

    else:
        raise KeyError(
            "Invalid library selected. Available libraries are: \'skimage\', \'cv2\',\'PIL\', and \'imageio\' ")

    return img.astype(np.int)


def canny_edge_map(image):
    image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    image = np.array(image)
    out = np.uint8(feature.canny(image, sigma=1, ) * 255)
    return out


def get_crop_params(image, crop_size, crop_method='random_crop'):
    crop_height, crop_width = crop_size
    start_x, end_x, start_y, end_y = 0, 0, 0, 0
    if image.shape[0] <= crop_height or image.shape[1] <= crop_width:
        image = cv2.resize(image, (crop_width * 2, crop_height * 2))

    if crop_method == 'random_crop':
        max_x = image.shape[1] // 2 - crop_width
        max_y = image.shape[0] // 2 - crop_height

        start_x = np.random.randint(0, max_x)
        end_x = start_x + crop_width
        start_y = np.random.randint(0, max_y)
        end_y = start_y + crop_height
    elif crop_method == 'center_crop':
        start_x = (image.shape[1] // 2 - crop_width // 2)
        end_x = start_x + crop_width
        start_y = (image.shape[0] // 2 - crop_height // 2)
        end_y = start_y + crop_height

    return {'top_left': (start_x, start_y), 'right_bottom': (end_x, end_y)}


def resize(img, height=256, width=256, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    img = cv2.resize(img, (height, width))
    return img


def crop_image(image, crop_pos=None, crop_size=(256, 256), random_crop=True, save_image=False):
    if random_crop:
        if crop_pos is None:
            crop_pos = get_crop_params(image, crop_size=crop_size)
        top_left = crop_pos.get('top_left')
        right_bottom = crop_pos.get('right_bottom')
        cropped_image = image[top_left[1]:right_bottom[1], top_left[0]:right_bottom[0], ...]
    else:
        cropped_image = resize(image, height=crop_size[0], width=crop_size[1], centerCrop=True)
    if save_image:
        imageio.imsave('./image.png', image)
        imageio.imsave('./cropped_image.png', cropped_image)
    return cropped_image


# =================================================================================================#

def vertical_stack_array(image_list):
    return np.vstack(image_list)


def horizontal_stack_array(image_list):
    return np.hstack(image_list)


# x = np.random.random((256,256,3))
# in_array = [x, x, x, x, x, x]
# vstacked = vertical_stack_array(in_array)
# hstacked = horizontal_stack_array(in_array)
#
# plt.figure()
# plt.imshow(hstacked)
# plt.show()
