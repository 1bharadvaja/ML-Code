import cv2
import tensorflow as tf

original1 = cv2.imread("/Users/thema/OneDrive/Desktop/ML_Code/png_data/testing/HR/222RCADefault_0007bscan_0572.png")
compressed1 = cv2.imread("/Users/thema/OneDrive/Desktop/ML_Code/b2a.png")

original2 = cv2.imread("/Users/thema/OneDrive/Desktop/ML_Code/png_data/testing/HR/20305Default_0014bscan_0602.png")
compressed2 = cv2.imread("/Users/thema/OneDrive/Desktop/ML_Code/b2a2.png")

original3 = cv2.imread("/Users/thema/OneDrive/Desktop/ML_Code/png_data/testing/HR/MarginDefault_0014bscan_0142.png")
compressed3 = cv2.imread("/Users/thema/OneDrive/Desktop/ML_Code/b2a3.png")

psnr1 = cv2.PSNR(original1, compressed1)
psnr2 = cv2.PSNR(original2, compressed2)
psnr3 = cv2.PSNR(original3, compressed3)
print(psnr1, psnr2, psnr3)

    # Read images (of size 255 x 255) from file.
im1 = tf.image.decode_image(tf.io.read_file('path/to/im1.png'))
im2 = tf.image.decode_image(tf.io.read_file('path/to/im2.png'))
tf.shape(im1)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`
tf.shape(im2)  # `img2.png` has 3 channels; shape is `(255, 255, 3)`
    # Add an outer batch for each image.
im1 = tf.expand_dims(im1, axis=0)
im2 = tf.expand_dims(im2, axis=0)
    # Compute SSIM over tf.uint8 Tensors.
ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)

    # Compute SSIM over tf.float32 Tensors.
im1 = tf.image.convert_image_dtype(im1, tf.float32)
im2 = tf.image.convert_image_dtype(im2, tf.float32)
ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    # ssim1 and ssim2 both have type tf.float32 and are almost equal.
#compressed = compressed.astype(np.float64) / 255.
#original = original.astype(np.float64) / 255.
#mse = np.mean((compressed - original) ** 2)
#psnr = 10 * math.log10(1. / mse)
#print(psnr) 
