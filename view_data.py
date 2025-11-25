"""
    python view_data.py path/to/image.jpg
"""

import cv2
import sys


# Đọc đường dẫn ảnh từ command line
image_path = sys.argv[1]

# Đọc ảnh
image = cv2.imread(image_path)

# Resize ảnh nếu quá lớn
h, w = image.shape[:2]
max_width = 1200
max_height = 800

if w > max_width or h > max_height:
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = cv2.resize(image, (new_w, new_h))

# Hiển thị ảnh
cv2.imshow('Image Viewer', image)

cv2.waitKey(0)

cv2.destroyAllWindows()
