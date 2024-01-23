# Create triangular array of emergent images

from PIL import Image
import argparse
import re

def parser():
    # note the use of `action="store_true"` to define the default when just `-gt` etc are entered.
    parser = argparse.ArgumentParser(
        description="Generate arrays of images from a set of SD-generated images." \
        " If the original image is much larger than [512,512], there may be resource issues."
    )
    parser.add_argument("seed_image", help = "original image used as the basis for the noising; used to size images")
    parser.add_argument("target_image", help = "the final image used in final display")
    parser.add_argument("--total", type=int, default=55, help = "total number of images to process; 10 rows of 10 require 55")
    parser.add_argument("-o", "--output", default="out.png", help = "base.ext filename for outputs")
    parser.add_argument("-pp", "--print_parser", action="store_true", default = False, help = "print the argument Namespace at inception")
    parser.add_argument("-ro", "--right_only", action="store_true", default = False, help = "only show the right-hand denoising layers")
    parser.add_argument("-ft", "--full_triangle", action="store_true", default = False, help = "create a full pyramidal display from left to right")
    args = parser.parse_args()

    if args.print_parser:
        print(args)

    return args

# utility to calculate number of rows without using sqrt
def row_number(total):
    for r in range(1,100):
        prod = r*(r+1)
        if prod == 2*total:
            return r
    raise ValueError("No matching value found!")

def get_digits(text):

    num_list = re.findall(r'\d+', text)
    print(num_list)

    return num_list

def save_target(destination):
    path = destination.split('/')
    path = '/'.join(path[:-1])

    digits = get_digits(destination)
    next = int(digits[-1]) + 1

    #input("press",)
    goal = f"{path}/tpyramid_pattern{next}.png"
    canvas.save(goal)
    print(f"Saved target triangular image to {goal}")

if __name__ == "__main__":

    args = parser()

    if args.print_parser:
        print(args)

    total_images = args.total

    img = Image.open(args.seed_image)
    size_w = img.size[0] # height of each image
    size_h = img.size[1] # width of each image
    print(f"height = {size_h}; width = {size_w}")
    print(f"Source base image: {args.seed_image}")
    print(f"Target image: {args.target_image}")
    #input("\npress",)

    base_name = args.seed_image.split('.')[0]
    print(f"base name = {base_name}")
    target_name = args.target_image.split('.')[0]
    print(f"Target name = {target_name}")
    #input("press",)

    # we are not currently using either of these
    list1 = ["SS", "TT", "UU", "VV", "WW", "XX", "YY", "ZZ", "AA", "BB"]
    list2 = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]

    print(f"args.right_only = {args.right_only}")
    print(f"args.full_triangle = {args.full_triangle}")
    # this must be selected; otherwise ft is chosen by default
    if args.right_only:
        # Create blank canvas
        rows = row_number(args.total)
        canvas = Image.new('RGB', (rows*size_w, rows*size_h), 'white')

        for row in range(1, rows+1):
            for col in range(1, row+1):
                filename = f"{base_name}_{row*20}_{col*20}.png"
                print(f"right-only: {filename}")
                img = Image.open(filename)
                x = (row-1)*size_h
                y = (col-1)*size_w
                #print(f"(x, y) = ({y},{x})")
                canvas.paste(img, (y,x))
                img = Image.open(f"{args.target_image}")
                img = img.resize((5*size_w, 5*size_h))
                canvas.paste(img, ((rows-5)*size_w,0))

        save_target(args.target_image)

    elif args.full_triangle:
        # Create blank canvas for full triangular left to right depiction

        rows = row_number(args.total)
        print(f"Number of rows required = {rows}")

        canvas = Image.new('RGB', ((2*rows-1)*size_w, rows*size_h), 'white')
        print(f"Canvas size = {canvas.size}")

        for row in range(1, rows + 1):
            for col in range(1, row + 1):
                filename = f"{base_name}_{row*20}_{col*20}.png"
                print(f"full triangle: {filename}")
                img = Image.open(filename)
                x = (row-1)*size_h
                y = (rows+col-2)*size_w
                #print(f"(x, y) = ({y},{x})")
                canvas.paste(img, (y,x))
                img = Image.open(args.seed_image)
                img = img.resize((5*size_w, 5*size_h))
                canvas.paste(img, (0,0))
                img = Image.open(f"{args.target_image}")
                img = img.resize((5*size_w, 5*size_h))
                canvas.paste(img, ((2*rows-6)*size_w,0))

        for col in range(1, rows +1):
            for row in range(1, col + 1):
                #print(f"(row, col) = ({row},{col})")
                filename = f"{base_name}_{row*20}_{20}.png"
                print(filename)
                img = Image.open(filename)
                x = (rows-col+row)*size_h
                y = (rows-1-row)*size_w
                #print(f"(x, y) = ({(rows-row+col-3)*size_w-y},{x}), (row, col) = ({row},{col})")
                canvas.paste(img, ((rows-row+col-3)*size_w-y,x))

        save_target(args.target_image)
