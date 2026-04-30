import cv2

input_path = "/home/allgo/mydrive/get_object/bag/test0.jpg"
output_path = "/home/allgo/mydrive/get_object/bag/test0_crop.png"

image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError("Failed to read image: %s" % input_path)

h, w = image.shape[:2]
y1, y2 = 0, 700
x1, x2 = 1000, 1500
if y1 >= y2 or x1 >= x2:
    raise ValueError(
        "Invalid crop for image size (%d, %d): y[%d:%d], x[%d:%d]"
        % (h, w, y1, y2, x1, x2)
    )

imageout = image[y1:y2, x1:x2]
ok = cv2.imwrite(output_path, imageout)
if not ok:
    raise IOError("Failed to write cropped image: %s" % output_path)

print("Saved:", output_path)
