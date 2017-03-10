# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import os

import numpy as np
import scipy.misc

from stylize import stylize

import math
from argparse import ArgumentParser

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
OUT_DIR = "output"


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
                        dest='content', help='content image',
                        metavar='CONTENT', required=True)
    parser.add_argument('--styles',
                        dest='styles',
                        nargs='+', help='one or more style images',
                        metavar='STYLE', required=True)
    parser.add_argument('--output',
                        dest='output', help='output path',
                        metavar='OUTPUT')
    parser.add_argument('--iterations', type=int,
                        dest='iterations', help='iterations (default %(default)s)',
                        metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
                        dest='print_iterations', help='statistics printing frequency',
                        metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-buffer',
                        dest='checkpoint_buffer', help='save iteration progress to single to single file',
                        action='store_true', default=False)
    parser.add_argument('--checkpoint-output',
                        dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
                        metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int,
                        dest='width', help='output width',
                        metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
                        dest='style_scales',
                        nargs='+', help='one or more style scales',
                        metavar='STYLE_SCALE')
    parser.add_argument('--network',
                        dest='network', help='path to network parameters (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight', type=float, nargs="+",
                        dest='content_weight', help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight', help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-blend-weights', type=float,
                        dest='style_blend_weights', help='style blending weights',
                        nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight', help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float, nargs="+",
                        dest='learning_rate', help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--initial',
                        dest='initial', help='initial image',
                        metavar='INITIAL')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error(
            "Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_image = imread(options.content)
    style_images = [imread(style) for style in options.styles]

    width = options.width
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                                    content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape
    

    scale_images = []
    if len(options.style_scales) > 1 and len(style_images) == 1:
        style_blend_weights = []
        for style_scale in options.style_scales:
            style_size = style_scale * target_shape[1] / style_images[0].shape[1]
            scale_images.append([scipy.misc.imresize(style_images[0], style_size), style_scale])
            style_blend_weights.append(1.0)
    else:
        for i in range(len(style_images)):
            style_scale = STYLE_SCALE
            if options.style_scales is not None:
                style_scale = options.style_scales[i]

            style_size = style_scale * target_shape[1] / style_images[i].shape[1]

            if not options.style_scales:
                options.style_scales = style_size

            style_images[i] = scipy.misc.imresize(style_images[i], style_size)
            scale_images.append([style_images[i], style_scale])

        style_blend_weights = options.style_blend_weights
        if style_blend_weights is None:
            # default is equal weights
            style_blend_weights = [1.0 / len(style_images) for _ in style_images]
        else:
            total_blend_weight = sum(style_blend_weights)
            style_blend_weights = [weight / total_blend_weight
                                   for weight in style_blend_weights]

    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])

    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    for scale_image in scale_images:
        for learning_rate in options.learning_rate:
            for content_weight in options.content_weight:

                for iteration, image in stylize(
                    network=options.network,
                    initial=initial,
                    content=content_image,
                    styles=[scale_image[0]],
                    iterations=options.iterations,
                    content_weight=content_weight,
                    style_weight=options.style_weight,
                    style_blend_weights=style_blend_weights,
                    tv_weight=options.tv_weight,
                    learning_rate=learning_rate,
                    print_iterations=options.print_iterations,
                    checkpoint_iterations=options.checkpoint_iterations
                ):
                    output_file = None
                    if iteration is not None:
                        if options.checkpoint_output:
                            output_file = options.checkpoint_output % iteration
                        if options.checkpoint_buffer:
                            output_file = os.path.join(OUT_DIR, "000.jpg")
                    else:
                        if options.output:
                            output_file = options.output
                        else:
                            if not os.path.exists(OUT_DIR):
                                os.makedirs(OUT_DIR)

                            # load the correct parameters for this run
                            import copy
                            iteration_options = copy.deepcopy(options)
                            iteration_options.content_weight = content_weight
                            iteration_options.learning_rate = learning_rate
                            iteration_options.style_scale = scale_image[1]

                            output_file = auto_output_filename(iteration_options)
                            output_file = os.path.join(OUT_DIR, output_file)

                    if output_file:
                        imsave(output_file, image)

                    if not iteration:
                        print "Created output at: {}".format(output_file)


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def auto_output_filename(options):
    content, ext = os.path.splitext(os.path.basename(options.content))
    style = os.path.splitext(os.path.basename(options.styles[0]))[0]
    name = ('{1}_'
            '{2}_'
            's{0.style_weight}_'
            'S{0.style_scale}_'
            'c{0.content_weight}_'
            't{0.tv_weight}_'
            'l{0.learning_rate}_'
            'i{0.iterations}'
            '{3}'
            ).format(options, content, style, ext)
    return name


if __name__ == '__main__':
    main()
