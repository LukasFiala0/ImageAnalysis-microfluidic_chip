import cv2
import matplotlib.pyplot as plt

img = cv2.imread('obr_test2.tif', 0)
img = cv2.medianBlur(img, 23)

# Jednoduché binární prahování
_, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

# Inverzní binární prahování
_, binary_inv = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)

# Truncate prahování
_, trunc = cv2.threshold(img, 10, 255, cv2.THRESH_TRUNC)

# ToZero prahování
_, tozero = cv2.threshold(img, 10, 255, cv2.THRESH_TOZERO)

# Inverzní ToZero prahování
_, tozero_inv = cv2.threshold(img, 10, 255, cv2.THRESH_TOZERO_INV)

# Otsuovo prahování
_, otsu = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Zobrazení výsledků
images = [binary, binary_inv, trunc, tozero, tozero_inv, otsu]
titles = ['Binary', 'Binary Inv', 'Trunc', 'To Zero', 'To Zero Inv', 'Otsu']

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig('Prahove_techniky.png')
plt.show()
