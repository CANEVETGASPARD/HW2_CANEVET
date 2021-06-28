import numpy as np
import matplotlib.pyplot as plt

def display_pictures(height,width,pictures):
    for picture in range(len(pictures)):
        picture_matrix = np.array(np.zeros([height, width]))
        # init row and col index to fill in picture matrix with the proper pixel
        # -> picture are shaped in 1-D list in our data set and we have to shaped them in 28*28 matrix
        row = 0
        col = 0
        for pixel in range(len(pictures[picture])):
            if (col == width):  # reinit at the end of the row
                col = 0
                row += 1

            picture_matrix[row, col] = pictures[picture][pixel]
            col += 1
        plt.imshow(picture_matrix,cmap="gray_r")
        plt.show()

if __name__ == "__main__":
    height_picture, width_picture = 28, 28
    centers = np.load("B/centroids.npy")
    display_pictures(height_picture,width_picture,centers)

