import cv2
import matplotlib.pyplot as plt
from filters import apply_grayscale, apply_gaussian_blur, apply_sepia, apply_sharpening, apply_cartoon_effect

# Load an image
image_path = "image.jpg"  # Replace with the path to your test image
image = cv2.imread(image_path)

# Apply all filters one by one
filters = {
    "Grayscale": apply_grayscale(image),
    "Gaussian Blur": apply_gaussian_blur(image, intensity=15),
    "Sepia": apply_sepia(image),
    "Sharpening": apply_sharpening(image),
    "Cartoon Effect": apply_cartoon_effect(image),
}

# Display the original image
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Display each filtered image
for i, (filter_name, filtered_img) in enumerate(filters.items(), start=2):
    plt.subplot(2, 3, i)
    if filter_name == "Grayscale":
        plt.imshow(filtered_img, cmap="gray")  # Grayscale needs a different color map
    else:
        plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
    plt.title(filter_name)
    plt.axis("off")

# Show all images
plt.tight_layout()
plt.show()