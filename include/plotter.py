
import matplotlib.pyplot as plt
import numpy as np
from . import statistics as stat

def plot_pixel_class_distribution(dataset, class_color_description):
    hist_by_pixel, hist_by_images, total_num_pixels = stat.calc_class_distribution_data(dataset, class_color_description)

    class_names = list(hist_by_pixel.keys())

    f, axs = plt.subplots(1,2,figsize=(30,5))
    x_pos = [i for i, _ in enumerate(class_names)]
    bar1 = axs[0].barh(x_pos, hist_by_pixel.values())
    axs[0].set_ylabel("Class")
    axs[0].set_xlabel("Num of Pixels")
    axs[0].set_yticks(x_pos)
    axs[0].set_yticklabels(class_names)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[0].text(0.7, 0.95, f'Total Pixel Number = {total_num_pixels:,.0f}', transform=axs[0].transAxes, fontsize=12, verticalalignment='top', bbox=props)
    for rect in bar1:
        width = rect.get_width()
        if width < 1_500_000:
            ha = 'left'
            pos = width + 10000
            color = 'black'
        else:
            ha = 'right'
            pos = width - 10000
            color = 'white'
        ratio = width / total_num_pixels
        axs[0].text(pos, rect.get_y() + rect.get_height() / 2.0, f'{width:,.0f} / {ratio*100:.2f}%', ha=ha, va='center', color=color, fontweight='bold')


    bar2 = axs[1].barh(x_pos, hist_by_images.values())
    axs[1].set_ylabel("Class")
    axs[1].set_xlabel("Number of Images")
    axs[1].set_yticks(x_pos)
    axs[1].set_yticklabels(class_names)

    for rect in bar2:
        width = rect.get_width()
        if width < 500:
            ha = 'left'
            pos = width + 7
            color = 'black'
        else:
            ha = 'right'
            pos = width - 7
            color = 'white'
        axs[1].text(pos, rect.get_y() + rect.get_height() / 2.0, f'{width:,.0f}', ha=ha, va='center', color=color, fontweight='bold')
    plt.show()

def plot_training_history(history, save_path, show_plot):
    plt.figure(figsize=(20, 10))
    # Plot training & validation loss values
    plt.subplot(231)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation iou_score values
    plt.subplot(232)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(233)
    plt.plot(history.history['recall_m'])
    plt.plot(history.history['val_recall_m'])
    plt.title('Model recall mean')
    plt.ylabel('recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(234)
    plt.plot(history.history['precision_m'])
    plt.plot(history.history['val_precision_m'])
    plt.title('Model precision mean')
    plt.ylabel('precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(235)
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.title('Model F1 score mean')
    plt.ylabel('f1')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show_plot:
        plt.show()