# multiresolution-image-blending
Multiresolution blending is described in the paper: P. Burt, E. Adelson, A Multiresolution Spline With Application to Image Mosaics,
ACM Transactions on Graphics, Vol. 2. No. 4, October 1983, Pages 217-236

The procedure can be divided into three main parts:

1.	Decomposition
•	Loading the images
•	Adjusting image dimensions so they are divisible by 2^nd. This ensures correct downsampling at each pyramid level without dimension errors.
•	Creating masks for the apple and orange images:
	  Apple mask: left half white, right half black
	  Orange mask: inverse of apple mask, MB = 1 - MA
•	Applying Gaussian filtering to the masks with sigma = 5% of the image dimension to smooth the transition
•	Constructing Gaussian pyramids for both masks (GMA and GMB). Each level is a downsampled, filtered version of the previous level.
•	Constructing Laplacian pyramids for both images. The Laplacian pyramid is obtained by subtracting the expanded next Gaussian level from the current level:
  L = Gi - expand(Gi+1)
  It represents image details at different frequency levels.

2.	Construction of the combined Laplacian pyramid
•	For each pyramid level, the combined Laplacian image is computed as:
  LAB = LA * GMA + LB * GMB
  Low frequencies ensure smooth transitions, high frequencies preserve sharp details.

3.	Reconstruction of the blended image
•	The final blended image is reconstructed by:
  R = expand(R) + LAB
•	This results in a seamless blend by combining low- and high-frequency information through pyramids.



