import matplotlib.pyplot as plt


def display_digit(digit):
    image = digit.reshape([28, 28])
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
