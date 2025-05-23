import numpy as np
import cv2 # OpenCV
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tkinter import filedialog
from PIL import Image

# Global to store cropping coordinates
crop_coords = []

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

crop_coords = []  # Global variable to store the selected coordinates

def line_select_callback(eclick, erelease):
    global crop_coords
    if eclick.xdata is None or erelease.xdata is None:
        crop_coords = []
    else:
        crop_coords = [
            (int(eclick.xdata), int(eclick.ydata)),
            (int(erelease.xdata), int(erelease.ydata))
        ]
    plt.close()

def crop_image_with_selector(image):
    global crop_coords
    crop_coords = []  # Reset before each selection

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Drag to select cropping area, then close window")

    toggle_selector = RectangleSelector(
        ax, line_select_callback,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )
    plt.show()

    return crop_coords


def extract_line_by_color(image, color, tolerance=30):
    """Extracts pixels matching a target color, keeping only the largest continuous blob."""
    # Convert image to RGB if needed
    if image.shape[2] == 4:  # RGBA → RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Convert to NumPy array and apply color mask
    lower = np.array([max(c - tolerance, 0) for c in color], dtype=np.uint8)
    upper = np.array([min(c + tolerance, 255) for c in color], dtype=np.uint8)
    mask = cv2.inRange(image, lower, upper)

    # Find connected components (contours)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No regions matched that color.")
        return np.array([])

    # Keep the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Create a clean mask with just the largest blob
    filtered_mask = np.zeros_like(mask)
    cv2.drawContours(filtered_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Get coordinates of nonzero pixels
    coords = np.column_stack(np.where(filtered_mask > 0))

    return coords



# Step 1: Load image
file_path = filedialog.askopenfilename(title="Select plot image")
image = np.array(Image.open(file_path).convert('RGB'))

# Step 2: Crop to axes
print("Please select the axes region...")
(x1, y1), (x2, y2) = crop_image_with_selector(image)
cropped = image[y1:y2, x1:x2]


# Step 3: Get line color
color_str = input("Enter line RGB color (e.g., '255,0,0' for red): ")
color = tuple(map(int, color_str.strip().split(',')))

# Step 4: Extract pixels
coords = extract_line_by_color(cropped, color)

# Flip coordinates (row, col) -> (x, y)
x = coords[:, 1]
y = coords[:, 0]
data = np.vstack((x, y)).T

# Step 5: Show results
plt.imshow(cropped)
plt.scatter(x, y, s=1, c='white')
plt.title("Extracted Line")
plt.show()

# Step 6: Save or return data
np.savetxt("extracted_data.csv", data, delimiter=",", header="x,y", comments='')
print("Data saved to extracted_data.csv")