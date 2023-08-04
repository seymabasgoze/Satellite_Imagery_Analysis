import rasterio as rio
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import rasterio


def nothing(x):
    pass


firs_path = r"C:\Users\stajyer\Downloads\eski.tiff"  # eski tarihli resim yolu (old pic)
second_path = r"C:\Users\stajyer\Downloads\yeni.tiff"  # yeni tarihli resim yolu (new pic)

set_data1 = rio.open(firs_path)
set_data2 = rio.open(second_path)

print("İlk resim veriler: ", set_data1.meta)  # meta ile verileri al
height = set_data1.bounds.right - set_data1.bounds.left
print("width: ".format(height))

print("İkinci resim veriler: ", set_data2.meta)  # meta ile verileri al
height = set_data2.bounds.right - set_data1.bounds.left
print("width: ".format(height))

b, g, r = set_data1.read()
fig = plt.imshow(g)
fig.set_cmap('inferno')
plt.colorbar()
# plt.show()

b, g, r = set_data2.read()
fig = plt.imshow(r)
plt.colorbar()
plt.show()

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

print("Image matching Error between the two images:", error)

cv2.namedWindow('canny')
slice1Copy = np.uint8(n_gray)
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'canny', 0, 1, nothing)

# add lower and upper threshold slidebars to "canny"
cv2.createTrackbar('lower', 'canny', 0, 255, nothing)
cv2.createTrackbar('upper', 'canny', 0, 255, nothing)

# Infinite loop until we hit the escape key on keyboard
while (1):

    # get current positions of four trackbars
    lower = cv2.getTrackbarPos('lower', 'canny')
    upper = cv2.getTrackbarPos('upper', 'canny')
    s = cv2.getTrackbarPos(switch, 'canny')

    if s == 0:
        edges = n_gray
    else:
        edges = cv2.Canny(n_gray, lower, upper)

    cv2.imshow("edge", edges)
    cv2.imshow("fark", diff)  # farkı gösterecek olan pencere
    cv2.imshow("eski", old_img)
    cv2.imshow("yeni", new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

