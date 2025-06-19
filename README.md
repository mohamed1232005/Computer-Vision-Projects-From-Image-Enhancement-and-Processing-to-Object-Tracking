# Computer-Vision-Projects-From-Image-Enhancement-and-Processing-to-Object-Tracking

# 📊 Computer Vision: Image Enhancement, Processing, and Object Tracking Projects

This repository includes five advanced computer vision projects showcasing various domains including spatial filtering, frequency analysis, morphological operations, content-based retrieval, and deep learning-based object tracking.

## 📘 Spatial Image Enhancement with Brightness and Contrast Adjustments

#Assignment 1

## Spatial Domain Filter

###Import required libraries

###Increase brightness and contrast

Brightness

Contrast

Use opencv functions : to handle uint8 arithmetic or convert to higher integer types to avoid overflow

imcreasing brightness with cv2

Together

###Apply histogram equalization

###Apply histogram equalization and Contrast Limited Adaptive Histogram Equalization (CLAHE) and compare results

Histogram Equalization for Color Images

Contrast Limited Adaptive Histogram Equalization (CLAHE) :improves the local contrast. More importantly it allows us to specify the size of the neighborhood that is considered "local".

###Apply gaussian filter with different kernel size and compare

Apply median filter

Apply sharpening filter

###Apply Adaptive thresholding and global thresholding and compare with results

```python
# Code Block 1
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.titlesize'] = 20
%matplotlib inline
matplotlib.rcParams['image.cmap'] = 'gray'
```

```python
# Code Block 2
img = cv2.imread("increase_brightness.jpg")
brightnessOffset = 50
brightHigh = img + brightnessOffset
plt.figure(figsize=[20,20])
plt.subplot(121);plt.imshow(img[...,::-1]);plt.title("Original Image");
plt.subplot(122);plt.imshow(brightHigh[...,::-1]);plt.title("High Brightness");
```

```python
# Code Block 3
print("Original Image Datatype : {}".format(img.dtype))
print("Brightness Image Datatype : {}\n".format(brightHigh.dtype))

print("Original Image Highest Pixel Intensity : {}".format(img.max()))
print("Brightness Image Highest Pixel Intensity : {}".format(brightHigh.max()))
```

```python
# Code Block 4
# Sample 2x2 matrix of type uint8
a = np.array([[100, 110],
              [120, 130]], dtype='uint8')
print(a)
```

```python
# Code Block 5
brightnessOffset = 50

# Add the offset for increasing brightness
brightHighOpenCV = cv2.add(img, np.ones(img.shape,dtype='uint8')*brightnessOffset)

brightHighInt32 = np.int32(img) + brightnessOffset
brightHighInt32Clipped = np.clip(brightHighInt32,0,255)

plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(img[...,::-1]);plt.title("original Image");
plt.subplot(132);plt.imshow(brightHighOpenCV[...,::-1]);plt.title("Using cv2.add function");
```

```python
# Code Block 6
# Add 130 so that the last element encounters overflow
print(a + 130)
```

```python
# Code Block 7
print(cv2.add(a,130))
```

```python
# Code Block 8
contrastPercentage = 30

# Multiply with scaling factor to increase contrast
contrastHighNormalized = (img * (1+contrastPercentage/100))/255
contrastHighNormalized01Clipped = np.clip(contrastHighNormalized,0,1)

# Display the outputs
plt.figure(figsize=[20,20])
plt.subplot(121);plt.imshow(img[...,::-1]);plt.title("Original Image");
plt.subplot(122);plt.imshow(contrastHighNormalized01Clipped[...,::-1]);plt.title("Normalized float to [0, 1]");
```

```python
# Code Block 9
brightnessOffset = 50
contrastPercentage = 30
# Brightness Adjustment using cv2.add
bright_img = cv2.add(img, np.ones(img.shape, dtype='uint8') * brightnessOffset)
contrast_img = np.float32(bright_img) * (1 + contrastPercentage / 100)
# Clip
contrast_img = np.clip(contrast_img, 0, 255).astype(np.uint8)


plt.figure(figsize=[20, 20])
plt.subplot(131); plt.imshow(img[..., ::-1]); plt.title("Original Image")
plt.subplot(132); plt.imshow(bright_img[..., ::-1]); plt.title("Brightness Adjusted")
plt.subplot(133); plt.imshow(contrast_img[..., ::-1]); plt.title("Brightness + Contrast Adjusted")
plt.show()
```

```python
# Code Block 10
img2 = cv2.imread("hist_equalization.jpg", cv2.IMREAD_GRAYSCALE)
plt.imshow(img2);
```

```python
# Code Block 11
# Equalize histogram
imEq = cv2.equalizeHist(img2)
#Display images
plt.figure()
ax = plt.subplot(1,2,1)
plt.imshow(img2, vmin=0, vmax=255)
ax.set_title("Original Image")
ax.axis('off')
ax = plt.subplot(1,2,2)
plt.imshow(imEq, vmin=0, vmax=255)
ax.set_title("Histogram Equalized")
ax.axis('off')
```

```python
# Code Block 12
plt.figure(figsize=(30,10))
plt.subplot(1,2,1)
plt.hist(img2.ravel(),256,[0,256]);

plt.subplot(1,2,2)
plt.hist(imEq.ravel(),256,[0,256]);
plt.show()
```

```python
# Code Block 13
img3 = cv2.imread("histogram _equalized.jpg", cv2.IMREAD_COLOR)
plt.imshow(img[...,::-1]);
```

```python
# Code Block 14
# Convert to HSV
imhsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)

# Perform histogram equalization only on the V channel
imhsv[:,:,2] = cv2.equalizeHist(imhsv[:,:,2])

# Convert back to BGR format
imEq = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)

#Display images
plt.figure()

ax = plt.subplot(1,2,1)
plt.imshow(img3[:,:,::-1], vmin=0, vmax=255)
ax.set_title("Original Image")
ax.axis('off')


ax = plt.subplot(1,2,2)
plt.imshow(imEq[:,:,::-1], vmin=0, vmax=255)
ax.set_title("Histogram Equalized")
ax.axis('off')
```

```python
# Code Block 15
# Convert to HSV
imhsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
imhsvCLAHE = imhsv.copy()

# Perform histogram equalization only on the V channel
imhsv[:,:,2] = cv2.equalizeHist(imhsv[:,:,2])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
imhsvCLAHE[:,:,2] = clahe.apply(imhsvCLAHE[:,:,2])

# Convert back to BGR format
imEq = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)
imEqCLAHE = cv2.cvtColor(imhsvCLAHE, cv2.COLOR_HSV2BGR)

#Display images
plt.figure(figsize=(40,40))

ax = plt.subplot(1,3,1)
plt.imshow(img3[:,:,::-1], vmin=0, vmax=255)
ax.set_title("Original Image")
ax.axis('off')


ax = plt.subplot(1,3,2)
plt.imshow(imEq[:,:,::-1], vmin=0, vmax=255)
ax.set_title("Histogram Equalized")
ax.axis('off')

ax = plt.subplot(1,3,3)
plt.imshow(imEqCLAHE[:,:,::-1], vmin=0, vmax=255)
ax.set_title("CLAHE")
ax.axis('off')
```

```python
# Code Block 16
img4 = cv2.imread("increase_brightness.jpg")
plt.imshow(img[...,::-1]);
```

```python
# Code Block 17
# Apply gaussian blur
dst1=cv2.GaussianBlur(img4,(5,5),0,0)
dst2=cv2.GaussianBlur(img4,(25,25),50,50)
# Display images
combined = np.hstack((img4, dst1,dst2))
```

```python
# Code Block 18
plt.figure(figsize=[20,10])
plt.subplot(131);plt.imshow(img[...,::-1]);plt.title("Original Image")
plt.subplot(132);plt.imshow(dst1[...,::-1]);plt.title("Gaussian Blur Result 1 : KernelSize = 5")
plt.subplot(133);plt.imshow(dst2[...,::-1]);plt.title("Gaussian Blur Result 2 : KernelSize = 25")
```

```python
# Code Block 19
img5 = cv2.imread("/content/images (8).jpeg", cv2.IMREAD_GRAYSCALE)
plt.imshow(img5);
```

```python
# Code Block 20
# Defining the kernel size
kernelSize = 5
# Performing Median Blurring
medianBlurred = cv2.medianBlur(img5,kernelSize)

# Display the original and median blurred image
plt.figure(figsize=[20,10])
plt.subplot(121);plt.imshow(img5[...,::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(medianBlurred[...,::-1]);plt.title("Median Blur Result : KernelSize = 5")
```

```python
# Code Block 21
img6 = cv2.imread("sharp_filter.jpeg", cv2.IMREAD_GRAYSCALE)
plt.imshow(img6);
```

```python
# Code Block 22
#Sharpen kernel
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")
```

```python
# Code Block 23
# Using 2D filter by applying the sharpening kernel
sharpenOutput = cv2.filter2D(img6, -1, sharpen)

plt.figure(figsize=[20,10])

plt.subplot(121);plt.imshow(img6[...,::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(sharpenOutput[...,::-1]);plt.title("Sharpening Result")
```

```python
# Code Block 24
# # Using 2D filter by applying the sharpening kernel
# sharpenOutput = cv2.filter2D(img6, -1, sharpen)

# plt.figure(figsize=[20,10])
# plt.subplot(121);plt.imshow(img6, cmap='gray');plt.title("Original Image")
# plt.subplot(122);plt.imshow(img6, cmap='gray');plt.title("Sharpening Result")
```

```python
# Code Block 25
img7 = cv2.imread("Threshold.jpeg", cv2.IMREAD_GRAYSCALE)
plt.imshow(img7);
```

```python
# Code Block 26
retval, img_thresh_gl_1 = cv2.threshold(img7, 50, 255, cv2.THRESH_BINARY)
retval, img_thresh_gl_2 = cv2.threshold(img7, 130, 255, cv2.THRESH_BINARY)

# Perform adaptive thresholding
img_thresh_adp = cv2.adaptiveThreshold(img7, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 7)

# Adjust figure size
plt.figure(figsize=(12, 6))

# spacing
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.subplot(221); plt.imshow(img7, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(222); plt.imshow(img_thresh_gl_1, cmap='gray'); plt.title("Thresholded (global: 50)"); plt.axis('off')
plt.subplot(223); plt.imshow(img_thresh_gl_2, cmap='gray'); plt.title("Thresholded (global: 130)"); plt.axis('off')
plt.subplot(224); plt.imshow(img_thresh_adp, cmap='gray'); plt.title("Thresholded (adaptive)"); plt.axis('off')

plt.show()
```

## 📘 Frequency Domain Filtering Using Discrete Fourier Transform

#Assignment_2

## Tune over different frequencies using **create_filter** function

## Update the **butterworth**, and **gaussian** parts in the following function

## Generate this plot:

![Image Description](https://drive.google.com/uc?id=1Ryif1rOhK2JVY70iw600_ha3geEzhIfP)

## Update the **create_lowpass_filter** function to support highpass and lowpass filters
### name the function **create_filter**

## Testing the new **create_filter** function

so , we have applied the  frequency domain filtering to an image using the Discrete Fourier Transform (DFT). It demonstrates low-pass, high-pass, band-pass, and band-stop filtering using ideal, Butterworth, and Gaussian methods. The filters help analyze and manipulate image frequency components, enhancing or suppressing specific details.

```python
# Code Block 1
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import urllib.request
```

```python
# Code Block 2
def load_image(image_url):
    """Load image in grayscale"""
    resp = urllib.request.urlopen(image_url)
    image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    return image
```

```python
# Code Block 3
# Provide the image URL
image_url = "https://drive.google.com/uc?id=1PKCyYW9FWhsQUchwjSlwQYsmoSASdFSy"
image = load_image(image_url)

# Display Original Image
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()
```

```python
# Code Block 4
# Compute the discrete Fourier Transform of the image
fourier = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift the zero-frequency component to the center of the spectrum
fourier_shift = np.fft.fftshift(fourier)

# calculate the magnitude of the Fourier Transform
magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))

# Scale the magnitude for display
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

cv2_imshow(magnitude)
```

```python
# Code Block 5
def apply_fourier_transform(image):
    """Compute the DFT and shift the zero-frequency component to the center"""
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    return dft_shifted

def inverse_fourier_transform(dft_shifted):
    """Reverse the Fourier transform and return the magnitude"""
    dft_shifted = np.fft.ifftshift(dft_shifted)  # Shift back
    img_back = cv2.idft(dft_shifted)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # Compute magnitude
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)  # Normalize for display


def create_filter(shape, filter_type, d0=30, w=10):
    """
    Create various frequency filters in the frequency domain.
    - filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    - d0: Cutoff frequency
    - w: Bandwidth for band filters
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.float32)

    for u in range(rows):
        for v in range(cols):
            d = np.sqrt((u - crow)**2 + (v - ccol)**2)  # Distance from center
            if filter_type == 'lowpass' and d <= d0:
                mask[u, v] = 1
            elif filter_type == 'highpass' and d > d0:
                mask[u, v] = 1
            elif filter_type == 'bandpass' and d0 - w / 2 <= d <= d0 + w / 2:
                mask[u, v] = 1
            elif filter_type == 'bandstop' and not (d0 - w / 2 <= d <= d0 + w / 2):
                mask[u, v] = 1

    return mask
```

```python
# Code Block 6
# Load image
path = "https://drive.google.com/uc?id=1PKCyYW9FWhsQUchwjSlwQYsmoSASdFSy"
image = load_image(path)

# Apply Fourier Transform
dft_shifted = apply_fourier_transform(image)

# Create Low-Pass Filter
lowpass_filter = create_filter(image.shape, 'lowpass', d0=50)

# Apply Filter
filtered_dft = dft_shifted * lowpass_filter

# Inverse Fourier Transform
img_lowpass = inverse_fourier_transform(filtered_dft)

# Display
cv2_imshow(img_lowpass)
```

```python
# Code Block 7
# Create High-Pass Filter
highpass_filter = create_filter(image.shape, 'highpass', d0=50)

# Apply Filter
filtered_dft = dft_shifted * highpass_filter

# Inverse Fourier Transform
img_highpass = inverse_fourier_transform(filtered_dft)

# Display
cv2_imshow(img_highpass)
```

```python
# Code Block 8
# Create Band-Pass Filter
bandpass_filter = create_filter(image.shape, 'bandpass', d0=80, w=30)

# Apply Filter
filtered_dft = dft_shifted * bandpass_filter

# Inverse Fourier Transform
img_bandpass = inverse_fourier_transform(filtered_dft)

# Display
cv2_imshow(img_bandpass)
```

```python
# Code Block 9
# Create Band-Stop Filter
bandstop_filter = create_filter(image.shape, 'bandstop', d0=80, w=30)

# Apply Filter
filtered_dft = dft_shifted * bandstop_filter

# Inverse Fourier Transform
img_bandstop = inverse_fourier_transform(filtered_dft)

# Display
cv2_imshow(img_bandstop)
```

```python
# Code Block 10
# Create Low-Pass Filters
def create_lowpass_filter(shape, filter_type, d0=30, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)

    for u in range(rows):
        for v in range(cols):
            d = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)  # Distance from center

            if filter_type == 'ideal':  # Ideal Low-Pass Filter
                mask[u, v] = 1 if d <= d0 else 0

            elif filter_type == 'butterworth':  # Butterworth Low-Pass Filter
                mask[u, v] = 1 / (1 + (d / d0) ** (2 * order))

            elif filter_type == 'gaussian':  # Gaussian Low-Pass Filter
                mask[u, v] = np.exp(-(d**2) / (2 * (d0 ** 2)))



    return mask


# Apply Fourier Transform
dft_shifted = apply_fourier_transform(image)

# Define filter types and frequency cutoff
filter_types = ['ideal', 'butterworth', 'gaussian']
d0 = 50  # Cutoff frequency

# Create figure for subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Apply different low-pass filters and display results
for i, f_type in enumerate(filter_types):
    # Create Low-Pass Filter
    lowpass_filter = create_lowpass_filter(image.shape, f_type, d0=d0)

    # Apply Filter
    filtered_dft = dft_shifted * lowpass_filter[:, :, np.newaxis]  # Expand dimensions

    # Inverse Fourier Transform
    img_lowpass = inverse_fourier_transform(filtered_dft)

    # Display in subplot
    ax = axes[i]
    ax.imshow(img_lowpass, cmap='gray')
    ax.set_title(f'{f_type.capitalize()} Low-Pass Filter (D0={d0})')
    ax.axis('off')

# Show the figure
plt.tight_layout()
plt.show()
```

```python
# Code Block 11
import math
```

```python
# Code Block 12
# Write the updated function here
# Create Low-Pass Filters
def create_lowpass_filter_after_updating(shape, filter_type, d0=30, order=2,  highpass = False):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)

    for u in range(rows):
        for v in range(cols):
            d = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)  # Distance from center

            if filter_type == 'ideal':  # Ideal Low-Pass Filter
                mask[u, v] = 1 if d <= d0 else 0

            elif filter_type == 'butterworth':  # Butterworth Low-Pass Filter
                mask[u, v] = 1 / (1 + (d / d0) ** (2 * order))

            elif filter_type == 'gaussian':  # Gaussian Low-Pass Filter
                mask[u, v] = np.exp(-(d**2) / (2 * (d0 ** 2)))

            if highpass == True:
              mask[u, v] = 1 - mask[u, v]

    return mask


# Apply Fourier Transform
dft_shifted = apply_fourier_transform(image)

# Define filter types and frequency cutoff
filter_types = ['ideal', 'butterworth', 'gaussian']
d0 = 50  # Cutoff frequency

# Create figure for subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Apply different low-pass filters and display results
for i, f_type in enumerate(filter_types):
    # Create Low-Pass Filter
    lowpass_filter = create_lowpass_filter(image.shape, f_type, d0=d0)

    # Apply Filter
    filtered_dft = dft_shifted * lowpass_filter[:, :, np.newaxis]  # Expand dimensions

    # Inverse Fourier Transform
    img_lowpass = inverse_fourier_transform(filtered_dft)

    # Display in subplot
    ax = axes[i]
    ax.imshow(img_lowpass, cmap='gray')
    ax.set_title(f'{f_type.capitalize()} Low-Pass Filter (D0={d0})')
    ax.axis('off')

# Show the figure
plt.tight_layout()
plt.show()
```

```python
# Code Block 13
# Apply Fourier Transform
dft_shifted = apply_fourier_transform(image)

# Define filter types and frequency cutoff
filter_types = ['ideal', 'butterworth', 'gaussian']
d0 = 50  # Cutoff frequency

# Create figure for Low-Pass and High-Pass filters
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Apply filters and display results
for i, f_type in enumerate(filter_types):
    # Low-Pass Filter
    lowpass_filter = create_lowpass_filter_after_updating(image.shape, f_type, d0=d0, highpass=False)
    filtered_dft_low = dft_shifted * lowpass_filter[:, :, np.newaxis]  # Expanding to match shape

    img_lowpass = inverse_fourier_transform(filtered_dft_low)

    # High-Pass Filter
    highpass_filter = create_lowpass_filter_after_updating(image.shape, f_type, d0=d0, highpass=True)
    filtered_dft_high = dft_shifted * highpass_filter[:, :, np.newaxis]  # Expanding to match shape

    img_highpass = inverse_fourier_transform(filtered_dft_high)

    # Display Low-Pass Filter Result
    axes[0, i].imshow(img_lowpass, cmap='gray')
    axes[0, i].set_title(f'{f_type.capitalize()} Low-Pass Filter (D0={d0})')
    axes[0, i].axis('off')

    # Display High-Pass Filter Result
    axes[1, i].imshow(img_highpass, cmap='gray')
    axes[1, i].set_title(f'{f_type.capitalize()} High-Pass Filter (D0={d0})')
    axes[1, i].axis('off')

# Show the figure
plt.tight_layout()
plt.show()
```

## 📘 Morphological Image Processing: Erosion, Dilation, Opening, and Closing

# Assignment 3

## Morphological Operations Implementation

### Data Loading

### Built-In Implementation
using built-in OpenCV functions

Observations of the Built-In :

Erosion removes details, while dilation expands objects.

Opening is useful for removing noise, while closing is effective for filling small gaps.

Edge handling in OpenCV is optimized, ensuring smooth results without excessive loss of detail.

Erosion-->: Shrinks white regions, removing fine details and making objects thinner. Small elements disappear, especially in the face image, where parts of the contour vanish.

Dilation-->: Expands white regions, thickening objects and merging details. The face image’s stripes blend together, and in the fish image, the eye appears larger.

Opening-->: Removes small noise while keeping the main structure. In the elephant image, small decorative details are lost, but the shape remains.

Closing-->: Fills small holes and connects broken parts. The face image’s stripes become smoother, and gaps in the elephant’s patterns are reduced.

Overall Comparison: Erosion removes details, dilation expands objects, opening cleans noise, and closing fills gaps. OpenCV handles edges well, ensuring smooth results.

### From Scratch Implementation

#### Erosion

#### Dilation

#### Opening

#### Closing

#### Applying custom implementations

Observations of Custom Implementaion:

####Erosion:

For 3×3 Kernel: The fine details of objects shrink as expected, but some small structures disappear completely.

For 5×5 Kernel: More aggressive shrinking occurs, leading to excessive removal of details, especially in thin or small regions.

Compared to OpenCV: The erosion effect is slightly stronger, likely due to strict kernel matching in the custom implementation.

#### Dilation :

For 3×3 Kernel: The objects expand moderately, and small gaps are reduced.

For 5×5 Kernel: The objects become thicker, and details start merging, particularly in the elephant and face images.

Compared to OpenCV: The results are similar, but the custom implementation might over-expand certain areas due to edge handling.

#### Opening:
For 3×3 Kernel: Small noise is reduced effectively, but some fine structures are also lost.

For 5×5 Kernel: The image gets too simplified, and many thin parts vanish completely.

Compared to OpenCV: The custom implementation is slightly more aggressive, likely due to stricter erosion.

####Closing :
For 3×3 Kernel: Small gaps are filled, but some parts don’t fully connect.

For 5×5 Kernel: Larger gaps are filled, but some regions appear over-closed.

Compared to OpenCV: The effect is stronger in the custom version, possibly due to dilation being applied more broadly.

####Conclusion:

OpenCV's implementation is optimized for better edge handling, smoother transitions, and efficiency.

The custom implementation is more aggressive, making erosion and dilation stronger than necessary in some cases.

For better accuracy, fine-tuning the kernel behavior in the custom approach is necessary to match OpenCV’s results more closely.

```python
# Code Block 1
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

```python
# Code Block 2
images = ['img_1.jpg', 'img_2.jpg', 'img_3.png']
```

```python
# Code Block 3
kernel = np.ones((3,3), np.uint8)
```

```python
# Code Block 4
kernel
```

```python
# Code Block 5
operations = ['Original', 'Eroded', 'Dilated', 'Opened', 'Closed']
```

```python
# Code Block 6
for img in images:
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #  Applying OpenCV's built-in morphological operations
    eroded_built_in= cv2.erode(image, kernel, iterations=1)
    dilated_built_in = cv2.dilate(image, kernel, iterations=1)
    opened_built_in= cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed_built_in = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv_images = [image, eroded_built_in, dilated_built_in, opened_built_in, closed_built_in]


    plt.figure(figsize=(12, 8))

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(cv_images[i], cmap='gray')
        plt.title(operations[i] + ' (OpenCV)')
        plt.xticks([]), plt.yticks([])

    plt.show()
```

```python
# Code Block 7
# Performs erosion on a binary image using a given structuring element (kernel).
# Erosion removes pixels at object boundaries, making objects shrink.

def Custom_Erosion(image, kernel):

    # Convert image to binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    h, w = binary_image.shape  # Get image dimensions
    kh, kw = kernel.shape  # Get kernel dimensions
    pad_h, pad_w = kh // 2, kw // 2  # Calculate padding size

    # Pad the image with zeros (black) to handle edges
    padded_img = np.pad(binary_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Create an output image initialized with zeros (black background)
    eroded_img = np.zeros_like(binary_image)

    # Slide the kernel over the image
    for i in range(h):
        for j in range(w):
            # Ensure that only white pixels are checked (avoid total removal)
            if np.all(padded_img[i:i+kh, j:j+kw] * kernel > 0):
                eroded_img[i, j] = 255  # Set pixel to white

    return eroded_img
```

```python
# Code Block 8
def Custom_Dilation(image, kernel):
# Dilation expands pixels at object boundaries, making objects larger.

    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2


    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    dilated_img = np.zeros_like(image)


    for i in range(h):
        for j in range(w):
            if np.any(padded_img[i:i+kh, j:j+kw] == 255):
                dilated_img[i, j] = 255

    return dilated_img
```

```python
# Code Block 9
def Custom_Opening(image, kernel):
  # Erosion followed by Dilation
    return Custom_Dilation(Custom_Erosion(image, kernel), kernel)
```

```python
# Code Block 10
def Custom_Closing(image, kernel):
  # Dilation followed by Erosion
    return Custom_Erosion(Custom_Dilation(image, kernel), kernel)
```

```python
# Code Block 11
kernel_sizes = [(3, 3), (5, 5)]
```

```python
# Code Block 12
for img2 in images:
    # Load image in grayscale
    image = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    for k_size in kernel_sizes:
        kernel = np.ones(k_size, np.uint8)  # Define kernel of current size

        # Apply custom implementations
        eroded = Custom_Erosion(image, kernel)
        dilated = Custom_Dilation(image, kernel)
        opened = Custom_Opening(image, kernel)
        closed = Custom_Closing(image, kernel)



        operations = ['Original', 'Eroded', 'Dilated', 'Opened', 'Closed']
        custom_images = [image, eroded, dilated, opened, closed]

        plt.figure(figsize=(12, 8))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(custom_images[i], cmap='gray')
            plt.title(f"{operations[i]} (Custom, {k_size[0]}x{k_size[1]})")
            plt.xticks([]), plt.yticks([])

        plt.show()
```

## 📘 Content-Based Image Retrieval Using SIFT, K-Means, and TF-IDF

This project presents a detailed experimental pipeline for image retrieval using SIFT descriptors and visual vocabulary encoding. It includes:

- Feature extraction using SIFT
- KMeans clustering with cuML for GPU acceleration
- Visual word histogram construction
- Comparison of Bag-of-Words (BoW) and TF-IDF encodings
- Similarity scoring using cosine similarity
- Parameter evaluation for number of images, centroids, and representations

### Summary of Experiments:

| Experiment | Parameter Tuned                     | Results Summary                                           |
|-----------|--------------------------------------|-----------------------------------------------------------|
| 1         | Number of images for clustering      | 1000 images achieved optimal trade-off                    |
| 2         | Number of centroids (k)              | 150 centroids balanced precision and efficiency           |
| 3         | Encoding strategy                    | TF-IDF outperformed BoW for most texture categories       |

Libraries used:
- `cv2` for SIFT
- `cupy`, `cuml.cluster.KMeans` for GPU clustering
- `sklearn.metrics.pairwise.cosine_similarity`
- `matplotlib` for visualization

## 📘 Real-Time Object Tracking in Video Using YOLOv10n, YOLOv11n, and YOLOv12n

#Assiggnment 5

## Object Tracking On Video using three differentversions of the YOLO model :
###(YOLOv10n, YOLOv11n, and YOLOv12n)

### Libraries

###Yolo Models

###Getting Frame Count

###Running Inference on All Models

### FPS Comparison Plot

### Video Display Helper Function

### Displaying Each Model Output

YOLOv10n

YOLOv11n

YOLOv12n

Visual inspection confirmed that all models effectively tracked moving vehicles with similar accuracy.

### Summary Table

Among the three YOLO models tested:

YOLOv11n offers the best trade-off between speed and tracking stability for this scenario.

### Recap

Models: yolov10n.pt, yolov11n.pt, yolov12n.pt

Tracking performed using Ultralytics YOLO with .track() function

Inference speed measured as:

-Total frames processed: 942

-Timing each run using Python’s time module

Output video saved and visualized for comparison

### Observations

YOLOv11n had the best performance in terms of speed (4.94 FPS).

YOLOv12n was the slowest (4.48 FPS), likely due to a heavier architecture.

YOLOv10n provided a balance but was slightly slower than YOLOv11n.

```python
# Code Block 1
# Install dependencies
!pip install ultralytics opencv-python

import os
import time
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import HTML, display
from base64 import b64encode
```

```python
# Code Block 2
# Input video
video_path = "/content/test1.mp4"

# Output directory
output_dir = "/content/tracked_outputs"
os.makedirs(output_dir, exist_ok=True)

# Uploaded models
models = {
    "YOLOv10n": "/mnt/data/yolov10n.pt",
    "YOLOv11n": "/mnt/data/yolo11n.pt",
    "YOLOv12n": "/mnt/data/yolo12n.pt"
}
```

```python
# Code Block 3
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
```

```python
# Code Block 4
results_summary = []

for name, path in models.items():
    print(f"Tracking with {name}")
    model = YOLO(path)

    start = time.time()
    model.track(
        source=video_path,
        save=True,
        save_dir=output_dir,
        name=f"{name.lower()}_result",
        persist=True,
        conf=0.25
    )
    end = time.time()

    total_time = end - start
    fps = frame_count / total_time

    results_summary.append({
        "Model": name,
        "Total Time (s)": round(total_time, 2),
        "Total Frames": frame_count,
        "Avg FPS": round(fps, 2),
        "Output File": f"{output_dir}/{name.lower()}_result/track.mp4"
    })

df_results = pd.DataFrame(results_summary)
```

```python
# Code Block 5
df_results = pd.DataFrame({
    "Model": ["YOLOv10n", "YOLOv11n", "YOLOv12n"],
    "Total Time (s)": [203.51, 190.74, 210.47],
    "Total Frames": [942, 942, 942],
    "Avg FPS": [4.63, 4.94, 4.48],
    "Output File": [
        "/content/runs/detect/yolov10n_result/track.mp4",
        "/content/runs/detect/yolov11n_result/track.mp4",
        "/content/runs/detect/yolov12n_result/track.mp4"
    ]
})
```

```python
# Code Block 6
# Fixing YOLOv10n result by converting .avi to .mp4
input_path = "/content/runs/detect/yolov10n_result/test1.avi"
output_path = "/content/runs/detect/yolov10n_result/track.mp4"

os.system(f"ffmpeg -i {input_path} -vcodec libx264 {output_path}")
```

```python
# Code Block 7
# YOLOv11n
input_path = "/content/runs/detect/yolov11n_result/test1.avi"
output_path = "/content/runs/detect/yolov11n_result/track.mp4"
os.system(f"ffmpeg -i {input_path} -vcodec libx264 {output_path}")

# YOLOv12n
input_path = "/content/runs/detect/yolov12n_result/test1.avi"
output_path = "/content/runs/detect/yolov12n_result/track.mp4"
os.system(f"ffmpeg -i {input_path} -vcodec libx264 {output_path}")
```

```python
# Code Block 8
def plot_fps_bar_chart(df):
    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df["Avg FPS"], edgecolor='black')
    plt.title("YOLO Model Inference Speed Comparison (FPS)")
    plt.xlabel("Model")
    plt.ylabel("Average FPS")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

plot_fps_bar_chart(df_results)
```

```python
# Code Block 9
def display_video(video_path, model_name):
    with open(video_path, 'rb') as f:
        video_data = f.read()
    data_url = "data:video/mp4;base64," + b64encode(video_data).decode()
    return HTML(f"""
    <h4>{model_name} - Output</h4>
    <video width=500 controls>
        <source src="{data_url}" type="video/mp4">
    </video><br>
    """)
```

```python
# Code Block 10
# YOLOv10n
display(display_video(df_results["Output File"][0], "YOLOv10n"))
```

```python
# Code Block 11
# YOLOv11n
display(display_video(df_results["Output File"][1], "YOLOv11n"))
```

```python
# Code Block 12
# YOLOv12n
display(display_video(df_results["Output File"][2], "YOLOv12n"))
```

```python
# Code Block 13
df_results.sort_values(by="Avg FPS", ascending=False)
```

