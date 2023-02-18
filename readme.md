1. Image Fundamentals
1.1. Quantization & Interpolation
1.1.1. For two cases as without and with histogram equalization (uniform histogram), display the quantized image in
(4, 8, 16, 32, 64, 128) Levels and its histograms. Also, the optimum mean square error obtained for each case.
Discuss and report the results for the gray Elaine image. It should be noted, you can use rgb2gray, histeq and
immse functions for this problem.

1.1.2. Write a program which can, firstly, downsample an image by a factor of 2, with and without using the averaging
filter, and also, up-sample the previously downsampled images by a factor of 2, using the pixel replication and
bilinear interpolation methods, respectively. Display (zoom of image) and discuss the results obtained with
different methods for the Goldhill image. Note, you can use immse function for this problem.


1.1.3. The initial image consists of eight bits of data for each pixel. Create new images using 5, 4, 3, 2 and 1 bit only
for each pixel. How many bits are needed to preserve image quality? Does it change from place to place in the
image? Discuss about the results. (Test Image Grayscale of Barbara).
