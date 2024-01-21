Pneumonia classification : The Dataset is taken from RSNA Pneumonia Detection challenge from Kaggle. It has 26684 X-ray images out of which 20672 images are without Pneumonia and 6012 images with Pneumonia.

1. First step is to perform Preprocessing of the images
   ->Original image shape is 1024 x 1024.
   -> Resizing image to 224 x 224.
   ->Standardizing the pixel values into the interval [0,1] by scaling with 1/256.
   -> Split the dataset into 24000 train images and 2684 validation images and then store the converted images in folders corresponding to the class i.e. 0 or 1.

![Screenshot 2024-01-20 211514](https://github.com/shanunrandev123/Pneumonia_classifier/assets/49170258/55d3562f-61c2-4373-a5ec-8919578cde74)
