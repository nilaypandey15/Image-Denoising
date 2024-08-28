def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet,estimate_sigma 
from skimage.util import random_noise
#import skimage.metrics.peak_signal_noise_ratio as psnr
#from skimage.metrics import peak_signal_noise_ratio
import skimage.io
import cv2
import numpy as np
from math import log10, sqrt

#img1=skimage.io.imread('dog.png')
img=cv2.imread("image_denoising-1.png") 
#cv2_im = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img1=skimage.img_as_float(img)
# img=img1
sigma=0.1
imgn=random_noise(img,var=sigma**2)


# Adjust sigma estimation parameters
sigma_est = estimate_sigma(imgn, average_sigmas=True)

#img_bayes = denoise_wavelet(img, sigma=0.1)
# Experiment with denoising parameters
img_bayes = denoise_wavelet(imgn, method='BayesShrink', mode='soft', wavelet_levels=4, wavelet='bior2.2')
img_visushrink = denoise_wavelet(imgn, method='VisuShrink', mode="soft", sigma=sigma_est/2, wavelet_levels=6, wavelet='sym4')
# Combine denoised images
img_combined = (img_bayes + img_visushrink) / 2
psnr_combined = PSNR(img, img_combined)

psnr_noisy=PSNR(img,imgn)
psnr_bayes=PSNR(img,img_bayes)
psnr_visu=PSNR(img,img_visushrink)

plt.figure(figsize=(30,30))

plt.subplot(2,2,1)
plt.imshow(img,cmap="gray")
plt.title('Original Image',fontsize=30)

plt.subplot(2,2,2)
plt.imshow(imgn,cmap="gray")
plt.title('Noisy Image',fontsize=30)

plt.subplot(2,2,3)
plt.imshow(img_combined,cmap="gray")
plt.title('combined image',fontsize=30)


#plt.subplot(2,2,3)
#plt.imshow(img_bayes,cmap="gray")
#plt.title('Denoised Image using Bayes',fontsize=30)

plt.subplot(2,2,4)
plt.imshow(img_visushrink,cmap="gray")
plt.title('Denoised Image using Visushrink',fontsize=30)
plt.show()



print('PSNR[Original Vs. Noisy Image]:',psnr_noisy)
print('PSNR[Original Vs Denoised(Visushrink)]:',psnr_visu)
print('PSNR[Original Vs. Denoised(Bayes)]:',psnr_bayes)
print('PSNR[Original Vs. combined image]:',psnr_combined)




