import numpy as np
import matplotlib.pyplot as plt

height_picture, width_picture = 28, 28
centers = np.load("B/centroidsNonScaled.npy")

for picture in range(len(centers)):
    picture_matrix = np.array(np.zeros([height_picture, width_picture]))
    # init row and col index to fill in picture matrix with the proper pixel
    # -> picture are shaped in 1-D list in our data set and we have to shaped them in 28*28 matrix
    row = 0
    col = 0
    for pixel in range(len(centers[picture])):
        if (col == width_picture):  # reinit at the end of the row
            col = 0
            row += 1

        picture_matrix[col, row] = centers[picture][pixel]
        col += 1
    plt.imshow(picture_matrix,cmap="gray_r")
    plt.show()

