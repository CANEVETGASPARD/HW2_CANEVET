import pandas as pd
import numpy as np
from centroid_display_and_saving import display_pictures

if __name__ == "__main__":
    height_picture, width_picture = 28, 28
    mnist_data = pd.read_csv("B/mnist-datasets/mnist_test.csv", header=None)
    mnist_data_array = mnist_data.to_numpy()
    display_pictures(height_picture, width_picture,mnist_data_array[:100], 'only exploring')



