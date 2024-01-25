# Copyright © 2023 Apple Inc.

import os
import argparse
import re

import mlx.core as mx
import numpy as np
from PIL import Image
from PIL import PngImagePlugin
from tqdm import tqdm

from stable_diffusion import StableDiffusion

def save_image(x, args):
    # Save them to disc
    im = Image.fromarray(np.array(x))      # Converting the 'mlx.core.array' object to a NumPy array before passing to Image.fromarray

    print(f"Prompt = {args.prompt}")
    metadata = PngImagePlugin.PngInfo()
    if args.save_prompt:
        metadata.add_text("Prompt", args.prompt)
    else:
        metadata = None

    im.save(args.output, pnginfo=metadata)
    print(f"image saved to file {args.output}")

    new_im = Image.open(args.output)
    print(f"Opened new file with name: {new_im.filename}")

    metadata = new_im.info
    print(f"Metadata saved with image file:")
    for k,v in metadata.items():
        print(f"Key: {k}; value: {v}")

    #print(f"Metadata saved prompt is {new_im.info['Prompt']}")

def save_interim_image(x, limit, count, args):
    # Save them to disc
    im = Image.fromarray(np.array(x))      # Converting the 'mlx.core.array' object to a NumPy array before passing to Image.fromarray

    print(f"Prompt = {args.prompt}")
    metadata = PngImagePlugin.PngInfo()
    if args.save_prompt:
        metadata.add_text("Prompt", args.prompt)
    else:
        metadata = None

    file_name = args.output.split('.')[0]
    file_ext = args.output.split('.')[-1]

    interim_file = f"{file_name}_{str(limit)}_{str(count)}.{file_ext}"
    im.save(interim_file, pnginfo=metadata)
    print(f"image saved to file {interim_file}")

    new_im = Image.open(interim_file)
    print(f"Opened new file with name: {new_im.filename}")

    metadata = new_im.info
    print(f"Metadata saved with image file:")
    for k,v in metadata.items():
        print(f"Key: {k}; value: {v}")

def get_latest_file(folder):

    # Get list of files in the folder
    files = os.listdir(folder)

    # Find the latest file by modification time
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder, x)))

    print(latest_file)
    # outimg2imgF_124.png

    # Extract letter
    match = re.search(r'outimg2img(.)_', latest_file)
    if match:
        letter = match.group(1)

    # Increment and substitute letter
    if letter == 'Z':
        letter = 'A'
    else:
        letter = chr(ord(letter) + 1)

    new_filename = latest_file.replace(match.group(1), letter)

    print(new_filename)
    # outimg2imgG_124.png

def filename_sequence():

    filename = "outimg2imgE_56.png"

    # Extract the letter part
    match = re.search(r'outimg2img(.)_', filename)
    if match:
        letter = match.group(1)

    # Increment the letter
    if letter == 'Z':
        letter = 'A'
    else:
        letter = chr(ord(letter) + 1)

    # Substitute the new letter back into the filename
    new_filename = filename.replace(match.group(1), letter)

    print(new_filename)
    # outimg2imgF_56.png

def reshape_image(original_image):

    img = mx.array(original_image)

    target_size_W = 64*round((img.shape[0] + 32) / 64)
    target_size_H = 64*round((img.shape[1] + 32) / 64)

    pad_amt_W = (target_size_W - img.shape[0]) // 2
    pad_amt_H = (target_size_H - img.shape[1]) // 2

    pad_margins = ((pad_amt_W, pad_amt_W), (pad_amt_H, pad_amt_H), (0, 0))

    padded_img = mx.pad(img, pad_width=pad_margins)

    print(f"\033[33mOriginal shape {img.shape}; padded shape {padded_img.shape}\033[0m")

    return padded_img

def decode_to_images(x_t, args):
    # Decode them into images
    decoded = []
    for i in tqdm(range(0, args.n_images, args.decoding_batch_size)):
        decoded.append(sd.decode(x_t[i : i + args.decoding_batch_size]))
        mx.eval(decoded[-1])

    # Arrange them on a grid
    x = mx.concatenate(decoded, axis=0)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    print(f"\n\033[33mBHWC = {B}, {H}, {W}, {C}\033[0m")
    x = x.reshape(args.n_rows, B // args.n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(args.n_rows * H, B // args.n_rows * W, C)
    x = (x * 255).astype(mx.uint8)

    return x

def parser():
    # note the use of `action="store_true"` to define the default when just `-gt` etc are entered.
    parser = argparse.ArgumentParser(
        description="Generate images from an image and a textual prompt using stable diffusion." \
        " If the original image is larger than [512,512], the number of final images may need to be reduced to 8 or fewer." \
        " Values of strength closer to 1 produce images that diverge more from the original." \
        " To see the emergent images as they denoise, set the '-sd' parameter to a number like 10." \
        " To save the text-prompt as metadata with the image files set -sp to true." \
        " Saving the last N consecutive sets of images allows selection of less refined results."
    )
    parser.add_argument("image", help = "original image to use as the basis for the noising")
    parser.add_argument("prompt", help = "text to define the final images")
    parser.add_argument("--strength", type=float, default=0.9, help = "value in (0,1); larger means more variation in the output image")
    parser.add_argument("--n_images", type=int, default=4, help = "total number of images arranged in n_rows")
    parser.add_argument("--n_rows", type=int, default=1, help = "the number of rows in the grid of final images")
    parser.add_argument("--steps", type=int, default=50, help = "maximum number of steps N")
    parser.add_argument("--cfg", type=float, default=7.5, help = "configuration number N")
    parser.add_argument("--negative_prompt", default="", help = "things to avoid in the final images")
    parser.add_argument("--decoding_batch_size", type=int, default=1)
    parser.add_argument("-o", "--output", default="out.png", help = "base.ext filename for outputs")
    parser.add_argument("-sd", "--show_denoising", type=int, default = 999, help = "show denoising images every N iterations")
    parser.add_argument("-sp", "--save_prompt", action="store_true", default = False, help = "save the main test-prompt as metadata")
    parser.add_argument("--save_last_N", type=int, default = 1, help = "save all the last N consecutive sets of images")
    parser.add_argument("-pp", "--print_parser", action="store_true", default = False, help = "print the argument Namespace at inception")
    parser.add_argument("-gt", "--generate_triangle", action="store_true", default = False, help = "create a triangular display of progressive strength")
    args = parser.parse_args()

    if args.print_parser:
        print(args)

    return args

if __name__ == "__main__":

    sd = StableDiffusion()

    args = parser()

    # Read the image
    img = mx.array(np.array(Image.open(args.image)))
    save_image(img, args) # test the save routine

    img = (img[:, :, :3].astype(mx.float32) / 255) * 2 - 1
    img = reshape_image(img)
    print(f"Reshaped original image now {img.shape}")

    # experimental triangular display generator
    if args.generate_triangle:
        # define the strength loop values [added zero here, which forces better alignment]
        strength_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        # use the strength as defined in args
        strength_list = [args.strength]

    for args.strength in strength_list:
    # Noise to produce the latents by encoding img.
    # This sd. function is in stable_diffusion/__init__.py
        latents = sd.generate_latents_from_image(
            img,
            args.prompt,
            strength=args.strength,
            n_images=args.n_images,
            cfg_weight=args.cfg,
            num_steps=args.steps,
            negative_text=args.negative_prompt,
        )
        # Denoise the latents to produce new images
        # By the time we get here all the noising is finished

        count = 0
        limit = int(args.steps * args.strength)
        for x_t in tqdm(latents, total=limit):
            count += 1
            mx.simplify(x_t)
            mx.simplify(x_t)
            mx.eval(x_t)
            if ((count % args.show_denoising == 0) or ((args.save_last_N != 1) and ((limit - count) < args.save_last_N))):
                x = decode_to_images(x_t, args)
                save_interim_image(x, limit, count, args)

        x = decode_to_images(x_t, args)
        save_image(x, args)

