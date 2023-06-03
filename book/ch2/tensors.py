import numpy as np
import matplotlib.pyplot as plt
import book.ch2.mnist_ex as mnist_ex


def header(msg):
    print(f"\n{msg}")
    print("-" * len(msg))


# Rank 0 tensors
header("Rank 0")
x = np.array(12)
print(x)
print(f"Dimensions {x.ndim}")

# Rank 1 tensors
header("Rank 1")
x = np.array([2, 3, 6, 9, 15])
print(x)
print(f"Dimensions = {x.ndim}")

# Rank 2 tensors
header("Rank 2")
x = np.array([[1, 2, 3, 4, 5],
              [10, 20, 30, 40, 50],
              [100, 200, 300, 400, 500]])
print(x)
print(f"Dimensions = {x.ndim}")

(train_images, train_labels), (test_images, test_labels) = mnist_ex.load_mnist()

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()