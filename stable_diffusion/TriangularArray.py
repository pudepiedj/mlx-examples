# Create triangular array of emergent images

from PIL import Image
from math import sqrt

total_images = 55
img = Image.open('stable_diffusion/images2images/Triangular/testimageSTRENGTHNew5.png')
size_h = img.size[0] # height of each image
size_w = img.size[1] # width of each image

'''# Create blank canvas
rows = int(sqrt(2*total_images + 0.25) - 0.5)
canvas = Image.new('RGB', (rows*size, rows*size), 'white')

list1 = ["SS", "TT", "UU", "VV", "WW", "XX", "YY", "ZZ", "AA", "BB"]
list2 = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]

for row in range(1, rows+1):
    for col in range(1, row+1):
        filename = f"stable_diffusion/images2images/Triangular/testimageSTRENGTHNew2_{row*20}_{col*20}.png"
        print(filename)
        img = Image.open(filename)
        x = (row-1)*size
        y = (col-1)*size
        print(f"(x, y) = ({y},{x})")
        canvas.paste(img, (y,x))
        img = Image.open('stable_diffusion/images2images/image102.png')
        img = img.resize((4*size, 4*size))
        canvas.paste(img, (5184-3*size,0))

canvas.save('stable_diffusion/images2images/Triangular/testimageSTRENGTHNew5.png')'''

# Create blank canvas for left to right depiction
rows = int(sqrt(2*total_images + 0.25) - 0.5)
canvas = Image.new('RGB', ((2*rows-1)*size_w, rows*size_h), 'white')

list1 = ["SS", "TT", "UU", "VV", "WW", "XX", "YY", "ZZ", "AA", "BB"]
list2 = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]

for row in range(1, rows + 1):
    for col in range(1, row + 1):
        filename = f"stable_diffusion/images2images/Triangular/testimageSTRENGTHNew6_{row*20}_{col*20}.png"
        print(filename)
        img = Image.open(filename)
        x = (row-1)*size_h
        y = (rows+col-2)*size_w
        print(f"(x, y) = ({y},{x})")
        canvas.paste(img, (y,x))
        img = Image.open('stable_diffusion/images2images/Triangular/testimageSTRENGTHNew5.png')
        img = img.resize((5*size_w, 5*size_h))
        canvas.paste(img, (0,0))
        img = Image.open('stable_diffusion/images2images/Triangular/testimageSTRENGTHNew6.png')
        img = img.resize((5*size_w, 5*size_h))
        canvas.paste(img, ((2*rows-6)*size_w,0))

for col in range(1, rows +1):
    for row in range(1, col + 1):
        print(f"(row, col) = ({row},{col})")
        filename = f"stable_diffusion/images2images/Triangular/testimageSTRENGTHNew6_{row*20}_{20}.png"
        print(filename)
        img = Image.open(filename)
        x = (rows-col+row)*size_h
        y = (rows-1-row)*size_w
        print(f"(x, y) = ({(rows-row+col-3)*size_w-y},{x}), (row, col) = ({row},{col})")
        canvas.paste(img, ((rows-row+col-3)*size_w-y,x))

canvas.save('stable_diffusion/images2images/tpyramid_pattern6.png')