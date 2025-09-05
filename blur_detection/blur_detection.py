import cv2
import matplotlib.pyplot as plt

def blur_score(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # read in grayscale
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def is_blurry(image_path, threshold=100):
    score = blur_score(image_path)
    return score < threshold, score

# Test on images
sharp_img = "sharp.png"   # put a sharp image here
blurred_img = "blur.png"  # put a blurred image here

for img in [sharp_img, blurred_img]:
    blurry, score = is_blurry(img)
    print(f"{img} → Blur Score: {score:.2f} → {'Blurry' if blurry else 'Sharp'}")

    # visualization
    image = cv2.imread(img)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Score: {score:.2f} | {'Blurry' if blurry else 'Sharp'}")
    plt.axis('off')
    plt.show()
