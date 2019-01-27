import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import atexit
import cv2
import matplotlib.pyplot as plt

is_window_shown = False


def make_collage(images, match_size=True, v_stack=False, uint_clip=True):
    font_path = os.path.join(os.path.dirname(__file__), 'assets', 'OpenSans.ttf')

    padding = 5
    font_box = 30
    font_size = 16
    max_dimension = 0
    num_images = 0

    tiles = []
    for name, img in images.items():
        num_images += 1
        if img.shape[0] > max_dimension:
            max_dimension = img.shape[0]
        if img.shape[1] > max_dimension:
            max_dimension = img.shape[1]

    # put images on collage
    pos = 0
    for name, img in images.items():
        tile_height = (max_dimension + (padding * 2))
        tile_width = max_dimension + padding * 3 + font_box
        tile_np = np.empty((tile_width, tile_height, 3))
        tile_np.fill(255)

        x = padding
        y = padding

        img_wrong_dim = (img.shape[0] != max_dimension or img.shape[1] != max_dimension)

        if match_size and img_wrong_dim:
            # scale to max dimensions
            # col_img = resize(img, (max_dimension, max_dimension), anti_aliasing=False, preserve_range=True)
            col_img = np.array(
                Image.fromarray(
                    np.clip(img, 0, 255).astype(np.uint8)
                ).resize((max_dimension, max_dimension))
            )
            # col_img = rescale(img, 4.0, anti_aliasing=False, preserve_range=True)
        elif img_wrong_dim:
            # put image in center of max dimension block
            col_img = np.empty((max_dimension, max_dimension, 3))
            col_img.fill(255)
            ci_x = (max_dimension - img.shape[0]) // 2
            ci_y = (max_dimension - img.shape[1]) // 2
            col_img[ci_x:ci_x + img.shape[0], ci_y:ci_y + img.shape[1], :] = img
        else:
            col_img = img

        # put image on collage
        tile_np[y:y + max_dimension, x:x + max_dimension, :] = col_img

        W = max_dimension
        H = font_box
        text = Image.new('RGB', (W, H), color=(210, 210, 210))
        draw = ImageDraw.Draw(text)
        font = ImageFont.truetype(font_path, font_size)
        w, h = draw.textsize(name, font=font)
        draw.text(((W - w) / 2, 3), name, fill="black", font=font)

        text = np.array(text)
        tile_np[y + max_dimension:y + max_dimension + font_box, x:x + max_dimension, :] = np.array(text)
        pos += 1
        tiles.append(tile_np)

    axis = 0 if v_stack else 1
    collage_np = np.concatenate(tuple(tiles), axis=axis)
    if uint_clip:
        return np.clip(collage_np, 0, 255).astype(np.uint8)
    else:
        return collage_np


def make_stacked_collage(images_list, match_size=True, uint_clip=True):
    collage = None
    for images in images_list:
        c = make_collage(images, match_size, uint_clip=uint_clip)
        if collage is None:
            collage = c
        else:
            collage = np.concatenate((collage, c), 0)
    return collage


def show_img(img, method='cv'):
    global is_window_shown
    width, height, chan = img.shape

    img_to_show = np.clip(img.copy(), 0, 255).astype(np.uint8)
    if method == 'plt':
        fig = plt.figure(figsize=(height / 100, width / 100))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img_to_show, cmap=plt.get_cmap("bone"))
        plt.show()
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imshow('NEURAL PIXELS', img[:, :, ::-1])
        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # Esc key to stop
            print('\nESC pressed, stopping')
            raise KeyboardInterrupt
        # show again to initialize the window if first
        if not is_window_shown:
            cv2.imshow('NEURAL PIXELS', img[:, :, ::-1])
            k = cv2.waitKey(10) & 0xFF
            if k == 27:  # Esc key to stop
                print('\nESC pressed, stopping')
                raise KeyboardInterrupt
        is_window_shown = True


def on_exit():
    if is_window_shown:
        cv2.destroyAllWindows()


atexit.register(on_exit)
