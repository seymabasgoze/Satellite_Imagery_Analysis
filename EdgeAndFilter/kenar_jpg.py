import skimage.color
from skimage.feature import canny
from skimage import data, morphology
from skimage.color import rgb2gray
import scipy.ndimage as nd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

plt.rcParams["figure.figsize"] = (8, 5)  # pencere boyut

old_img = cv2.imread(r"C:\Users\stajyer\Downloads\eski.tiff", cv2.IMREAD_UNCHANGED)
new_img = cv2.imread(r"C:\Users\stajyer\Downloads\yeni.tiff", cv2.IMREAD_UNCHANGED)

o_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
n_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)


def mse(img1, img2):  # iki resmin karşılaştırılması fonksiyonu
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h * w))
    return mse, diff


error, diff = mse(o_gray, n_gray)
print("2 resmin eşleme hatası: ", error)
plt.imshow(diff)
plt.title('Fark')
plt.show()

fark_resmi = (diff * 255).astype(np.uint8)  # tiff tipindeki dosya üzerinde işlem yapmak için
plt.imshow(fark_resmi)
plt.title('uint uygulanmış fark')
plt.show()

# KENAR
edges = canny(fark_resmi)
plt.imshow(edges, interpolation='gaussian')
plt.title('kenar tespiti')
plt.show()

# REGION
fill_im = nd.binary_fill_holes(edges)
plt.imshow(fill_im)
plt.title('Region based deneme')
plt.show()

# SOBEL FONKS
elevation_map = cv2.Sobel(fark_resmi, cv2.CV_64F, 1, 0, ksize=5)
plt.imshow(elevation_map)
plt.title('sobel deneme')
plt.show()

# İŞARET
markers = np.zeros_like(fark_resmi)
markers[fark_resmi < 0.1171875] = 1  # 30/255
markers[fark_resmi > 0.5859375] = 2  # 150/255
plt.imshow(markers)
plt.title('işaretler')
plt.show()

# ?
segmentation = watershed(elevation_map, markers=1)
plt.imshow(segmentation)
plt.title('segmentasyon uygulama')
plt.show()

# LABEL
segmentation = nd.binary_fill_holes(segmentation - 1)
label_rock, _ = nd.label(segmentation)
image_label_overlay = skimage.color.label2rgb(label_rock, image=fark_resmi)
plt.imshow(image_label_overlay)
plt.title('laber2rgb fonks.')
plt.show()

# KIYAS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16), sharey=True)
ax1.imshow(fark_resmi)
ax1.contour(segmentation, [0.8], linewidths=1.8, colors='w')
ax2.imshow(image_label_overlay)
plt.show()  # iki filtre kıyaslama (fark-label2rgb)

# OPENCV
a = (segmentation * 255).astype(np.uint8)
cv2.imshow("fark", fark_resmi)
cv2.imshow("ele", elevation_map)
# cv2.imshow("segmen",segmentation)
cv2.imshow("l", image_label_overlay)
cv2.imshow("segmen", a)
# fig.subplots_adjust(matplotlib.pyplot.margins())


cv2.waitKey(0)
cv2.destroyAllWindows()
