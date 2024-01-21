Pneumonia classification : The Dataset is taken from RSNA Pneumonia Detection challenge from Kaggle. It has 26684 X-ray images out of which 20672 images are without Pneumonia and 6012 images with Pneumonia.

1. First step is to perform Preprocessing of the images
   ->Original image shape is 1024 x 1024.
   -> Resizing image to 224 x 224.
   ->Standardizing the pixel values into the interval [0,1] by scaling with 1/256.
   -> Split the dataset into 24000 train images and 2684 validation images and then store the converted images in folders corresponding to the class i.e. 0 or 1.

![Screenshot 2024-01-20 211514](https://github.com/shanunrandev123/Pneumonia_classifier/assets/49170258/55d3562f-61c2-4373-a5ec-8919578cde74)

   -> Compute training mean and standard deviation for normalization. Dataset is very large(>3GB) so i computed sigma x and sigma x**2 for each image x and then add those to globally defined variables sums and sum_squared.

2. Dataset Creation : I used Torchvision Dataset folder and Z- normzalized images using the formula (X - mean/std deviation)
   Applied some Data Augmentation techniques -> Random rotations, random translations, random scales, random resize crops,

3. Training : Employed Pytorch lightning which is a torch wrapper and used ResNet18 architecture
   Changed input channels from 3 to 1(as i am working with medical images which arent in RGB format)
   Changed output dimension from 1000 to 1
   Loss Function -> Used BCEWithLogitLoss : directly applied on logits(raw predictions); negative output <=> No Pneumonia
   Optimizer -> Used Adam optimizer with lr = 1e-4 and trained for 30 epochs
   
