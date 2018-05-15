# plot utils

import matplotlib.pyplot as plt
from io import BytesIO
import base64

# for creating custom `cmap`
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb

from dataset_utils import load_dataset


svgMetaData = """<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Created with matplotlib (http://matplotlib.org/), modified to stack multiple svg elemements -->
<svg version="1.1" width="22" height="22" viewBox="0 0 22 22" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
 <defs>
  <style type="text/css">
    *{stroke-linecap:butt;stroke-linejoin:round;}
    .sprite { display: none;}
    .sprite:target { display: inline; }
  </style>
 </defs>
"""
# .myimg {  filter: invert(100%); }

svgImgTag = """
<g class="sprite" id="{}">
    <image class="myimg" id="img_{}" width="20" height="20" xlink:href="data:image/png;base64,{}"/>
</g>
"""

current_dpi = plt.gcf().get_dpi()
fig = plt.figure(figsize=(28 / current_dpi, 28 / current_dpi))


def create_cm(basecolor):
    colors = [(1, 1, 1), to_rgb(basecolor), to_rgb(basecolor)]  # R -> G -> B
    return LinearSegmentedColormap.from_list(colors=colors, name=basecolor)


basecolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
cmaps = []
for basecolor in basecolors:
    cmaps.append(create_cm(basecolor))


def gen_figure_data(data, classId, img_size, for_COIL=False):
    figFile = BytesIO()
    if for_COIL:
        plt.imsave(figFile,
                   data.reshape(img_size, img_size).T, cmap='gray')
    else:
        plt.imsave(figFile,
                   data.reshape(img_size, img_size), cmap=cmaps[classId])
    plt.gcf().clear()
    figFile.seek(0)
    return base64.b64encode(figFile.getvalue()).decode('utf-8')


def gen_svg_stack(dataset_name, X, classIds, n, img_size, for_COIL=False):
    outfile = './plots/{}.svg'.format(dataset_name)
    with open(outfile, "w") as svgFile:
        svgFile.write(svgMetaData)
        for i in range(n):
            figData = gen_figure_data(X[i], classIds[i], img_size, for_COIL)
            svgFile.write(svgImgTag.format(i, i, figData))
        svgFile.write("</svg>")


if __name__ == '__main__':
    datasets = {
        'MNIST': 28,
        'MNIST-SMALL': 8,
        'COIL20': 32
    }
    for dataset_name, img_size in datasets.items():
        X, y, labels = load_dataset(dataset_name)
        gen_svg_stack(dataset_name, X, y, len(y), img_size,
                      for_COIL=(dataset_name == 'COIL20'))
