import sys

print(sys.executable)
from PIL import Image

img = Image.open("puppy.jpg")  # Just put the local filename in quotes.
img.show()  # Send the image to your OS to be displayed as a temporary file
print(
    img.size
)  # A tuple. Note: width first THEN height. PIL goes [x, y] with y counting from the top of the frame.
pix = (
    img.load()
)  # Pix is a pixel manipulation object; we can assign pixel values and img will change as we do so.
print(
    pix[2, 5]
)  # Access the color at a specific location; note [x, y] NOT [row, column].
pix[2, 5] = (
    255,
    255,
    255,
)  # Set the pixel to white. Note this is called on “pix”, but it modifies “img”.
img.show()  # Now, you should see a single white pixel near the upper left corner
img.save("my_image.png")  # Save the resulting image. Alter your filename as necessary.
