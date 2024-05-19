# FastSAM Object Segmentation and Extraction

This project uses FastSAM for object segmentation and extraction. The project allows you to segment and extract objects from images using state-of-the-art techniques.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction
FastSAM is a cutting-edge tool for object segmentation and extraction. It leverages advanced models to accurately segment and extract objects from images. This project makes it easy to use FastSAM for various image processing tasks.

## Features
- High accuracy object segmentation
- Efficient and fast processing
- Easy-to-use interface for image segmentation and extraction

## Installation
To install the required dependencies and set up the project, follow these steps:

1. Clone the FastSAM repository:

    ```bash
    git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
    ```

2. Install the required Python packages:

    ```bash
    pip install -r FastSAM/requirements.txt
    pip install git+https://github.com/openai/CLIP.git roboflow supervision
    ```

3. Download the pre-trained weights:

    ```bash
    wget -P FastSAM/weights https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt
    ```

## Usage
To use FastSAM for object segmentation and extraction, follow these steps:

1. Ensure you have the necessary dependencies installed and the pre-trained weights downloaded as described in the Installation section.

2. Run your segmentation and extraction scripts using FastSAM. For example:

```python
cd FastSAM
from fastsam import FastSAM, FastSAMPrompt
import supervision as sv
import roboflow
from roboflow import Roboflow
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = FastSAM('./weights/FastSAM.pt')

#path to your image
path='/content/pexels-vladimirsrajber-18631420.jpg'

DEVICE='cuda:0'

everything_results = model(path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
prompt_process = FastSAMPrompt(path, everything_results, device=DEVICE)
ann = prompt_process.everything_prompt()
output_filename = f'output_images6.jpg'
output_path = os.path.join('./output/', output_filename)
prompt_process.plot(annotations=ann ,output_path=output_path)
 ```
```bash
- After Segmentation of image Run this code to extract each object from segmented image
# Load the original image
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# Iterate through each result in everything_results
for idx, result in enumerate(everything_results):
    # Assuming each result contains multiple masks
    for mask_idx, mask in enumerate(result.masks):
        mask_array = mask.data[0].cpu().numpy()
        binary_mask = np.where(mask_array > 0.5, 1, 0)

        white_background = np.ones_like(image) * 255

        # Combine the object with the white background
        new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]

        # Save the image
        output_filename = os.path.join(output_folder, f'object_{idx}_{mask_idx}.png')
        plt.imsave(output_filename, new_image.astype(np.uint8), format='png', dpi=300)
```
## Example Image
- Here is an example of an original image used in this project:

<img src="https://i.postimg.cc/mZHnsbXH/orginal-image.jpg" alt="Original Image" width="300"/>

- This is the segmented Image

<img src="https://i.postimg.cc/CKNdQTDB/Segmented-image.jpg" alt="Segmented_Image" width="300"/>

- one sample result of extracted object from segmented image

<img src="https://i.postimg.cc/7PMGCzbM/object-0-1.png" alt="Segmented_Image" width="300"/>



## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
