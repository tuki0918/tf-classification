import argparse
import math
import os

from PIL import Image, ImageOps

FLAGS = None


def resize(path, img_size=64):
    """
    指定した画像をリサイズ（正方形）する
    :param path:
    :param img_size:
    :return:
    """
    img = Image.open(path, 'r')
    img = ImageOps.fit(img, (img_size, img_size), Image.ANTIALIAS)

    w, h = img.size
    p1 = math.floor((img_size - w) / 2)
    p2 = math.floor((img_size - h) / 2)

    canvas = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    canvas.paste(img, (p1, p2))
    canvas.save(path, 'JPEG', quality=100, optimize=True)


def find_all_files(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            yield os.path.join(root, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='images',
                        help='***')
    FLAGS, unparsed = parser.parse_known_args()

    if os.path.isdir(FLAGS.path):
        for path in find_all_files(FLAGS.path):
            if not path.endswith('DS_Store'):
                resize(path)
