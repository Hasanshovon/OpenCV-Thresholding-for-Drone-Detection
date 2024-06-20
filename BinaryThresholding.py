import os 
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

# Get the directory
path = os.getcwd()
path = os.path.join(path, "data")

# Get all file names
file_list = os.listdir(path)
column = 2
rows = len(file_list)

fig = plt.figure(figsize=(10, 5 * rows))
fig.suptitle("Binary Thresholding for Drone Detection", fontsize=16)
images = []

for i, file in enumerate(file_list):
    name = os.path.join(path, file)
    img = cv.imread(name, cv.IMREAD_GRAYSCALE)
    ret, thresh1 = cv.threshold(img, 130, 255, cv.THRESH_BINARY)
    images.append(img)
    
    if img is not None:
        ax1 = fig.add_subplot(rows, column, 2*i + 1)
        ax1.imshow(img, cmap='gray')
        ax1.set_title(f'Original - {file}')
        ax1.axis("off")
        
        ax2 = fig.add_subplot(rows, column, 2*i + 2)
        ax2.imshow(thresh1, cmap='gray')
        ax2.set_title(f'Thresholded - {file}')
        ax2.axis("off")
    else:
        print(f'Failed to read file {name}')

#plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust the top to make room for the main title
plt.show()
