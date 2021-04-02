import io
import torch

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import multilabel_confusion_matrix
from torchvision import transforms as torch_transforms


def tensor_confusion_matrix(y_pred, y_true, display_labels):
    """
    Creates per class confusion matrices witch are later concatenated

    Args:
        y_pred (torch.Tensor ( N x C )): Sigmoid multi-label predictions
        y_true (torch.Tensor ( N x C )): Multi-label targets
        display_labels (list): The string representation of the labels
    Returns:
        (torch.Tensor): Image tensor of the concatenated confusion matrices
    """

    # Creates an array of 2x2 arrays of size num_classes
    cm_array = multilabel_confusion_matrix(y_true.cpu(), y_pred.cpu())

    # Create transform to convert PIL to Tensor for Tensorboard
    to_tensor = torch_transforms.ToTensor()

    ims = []
    for i, cm in enumerate(cm_array):
        # Plotting one confusion matrix with sklearn
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal', values_format='d')

        # Add the corresponding label as titel of the plot
        plt.title(display_labels[i])

        # Write figure into buffer and open it as PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)

        # Copy image because buffer is closed afterwards
        ims.append(im.copy())
        buf.close()

    widths, heights = zip(*(i.size for i in ims))

    total_width = sum(widths)
    max_height = max(heights)

    # Create new image with (num_classes * width)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    # Pasting the confusion matrices into new_im
    for im in ims:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    # Transform to tensor for Tensorboard logger
    new_im = to_tensor(new_im)

    return new_im
