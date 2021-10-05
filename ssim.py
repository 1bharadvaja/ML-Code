import tensorflow as tf
    # Read images (of size 255 x 255) from file.
im1 = tf.image.decode_image(tf.io.read_file('/Users/thema/Desktop/ML_Code/png_data/testing/HR/222RCADefault_0007bscan_0572.png'))
im2 = tf.image.decode_image(tf.io.read_file('/Users/thema/Desktop/ML_Code/b2a.png'))

im3 = tf.image.decode_image(tf.io.read_file('/Users/thema/Desktop/ML_Code/png_data/testing/HR/20305Default_0014bscan_0602.png'))
im4 = tf.image.decode_image(tf.io.read_file('/Users/thema/Desktop/ML_Code/b2a2.png'))

im5 = tf.image.decode_image(tf.io.read_file('/Users/thema/Desktop/ML_Code/png_data/testing/HR/MarginDefault_0014bscan_0142.png'))
im6 = tf.image.decode_image(tf.io.read_file('/Users/thema/Desktop/ML_Code/b2a3.png'))

tf.shape(im1)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`
tf.shape(im2)  # `img2.png` has 3 channels; shape is `(255, 255, 3)`
    # Add an outer batch for each image.
im1 = tf.expand_dims(im1, axis=0)
im2 = tf.expand_dims(im2, axis=0)

im3 = tf.expand_dims(im3, axis=0)
im4 = tf.expand_dims(im4, axis=0)

im5 = tf.expand_dims(im5, axis=0)
im6 = tf.expand_dims(im6, axis=0)
    # Compute SSIM over tf.uint8 Tensors.
ssim1_1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
ssim1_2 = tf.image.ssim(im3, im4, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
ssim1_3 = tf.image.ssim(im5, im6, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)

    # Compute SSIM over tf.float32 Tensors.
im1 = tf.image.convert_image_dtype(im1, tf.float32)
im2 = tf.image.convert_image_dtype(im2, tf.float32)
ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
print(ssim1_1, ssim1_2, ssim1_3)