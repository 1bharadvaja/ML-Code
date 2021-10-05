import cv2
import tensorflow as tf

original1 = cv2.imread("/Users/thema/Desktop/ML_Code/png_data/testing/HR/222RCADefault_0007bscan_0572.png")
compressed1 = cv2.imread("/Users/thema/Desktop/ML_Code/b2a.png")

original2 = cv2.imread("/Users/thema/Desktop/ML_Code/png_data/testing/HR/20305Default_0014bscan_0602.png")
compressed2 = cv2.imread("/Users/thema/Desktop/ML_Code/b2a2.png")

original3 = cv2.imread("/Users/thema/Desktop/ML_Code/png_data/testing/HR/MarginDefault_0014bscan_0142.png")
compressed3 = cv2.imread("/Users/thema/Desktop/ML_Code/b2a3.png")

psnr1 = cv2.PSNR(original1, compressed1)
psnr2 = cv2.PSNR(original2, compressed2)
psnr3 = cv2.PSNR(original3, compressed3)
print(psnr1, psnr2, psnr3)
