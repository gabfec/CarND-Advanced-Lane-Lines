import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def display_before_after(before_img, after_img, before_text="Original", after_text="After"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(before_img)
    ax1.set_title(before_text, fontsize=20)
    ax2.imshow(after_img)
    ax2.set_title(after_text, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def display_img(image):
    f, ax1 = plt.subplots(1, 1, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    #ax1.set_title(before_text, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()