# FISH simulator

This FISH image simulator is able to quickly generate thousands of images with various noise background and various RNA spots signal. 
The noise is simulated with a Poisson noise and a Perlin noise. The FISH signal is simulated with random positoning spots convolve by a Gaussian kernel to simulate the PSF. 
Motion blur is also simulated with elastic transform.

## Install dependencies with Conda 
DeepSpot requires Python >= 3.6 and TensorFlow >= 2.2.

To use a tested environment :

`conda env create spotsim.yaml`

Then activate the environment:

`conda activate spotsim`

## Usage

`python spot_simulator.py`

## Parameters

* root : path where images are output
* spot_number : number of spot per image (use a random for a variable number of spot per image)
* image_number : number of images to simulate
* spot_size : default [4,9], when spot_size is big, the spot is small
* kernel_size : gaussian kernel size for PSF simulation, the spot is wider with high values
* perlin_level : Perlin noise shape, small level yeld dense noise, higher level yeld a more diffuse Perlin noise
* poisson_noise_level : max intensity of Poisson noise
* full_noise_level : max intensity of the sum of the two previous noise
* out_image_size : size of image to simulate, default 256*256