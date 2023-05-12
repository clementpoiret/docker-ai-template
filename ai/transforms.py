import torch
from torchvision import transforms


class ResizeAndCrop(torch.nn.Module):
    """Resize and crop the input to the given height and width."""

    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def __resize(self, image):
        """Resize the image so that the height is equal to the given height,
        while the width is scaled accordingly."""
        return transforms.Resize(
            (self.height,
             int(self.height / image.size[1] * image.size[0])))(image)

    def __crop(self, image):
        """When needed, crop the image (random crop) to fix the width to self.width."""
        if image.size[0] <= self.width:
            return image
        else:
            return transforms.RandomCrop((self.height, self.width))(image)

    def forward(self, image):
        return self.__crop(self.__resize(image))


class PadIfNeeded(torch.nn.Module):
    """Pad the input from the center to the given height and width.
    Uses a reflection padding."""

    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, image):
        """Pad the image from the center to the given height and width.
        Uses a reflection padding.
        Ensure that the width is exactly self.width and the height is exactly self.height."""
        return transforms.Pad((int((self.width - image.size[0]) / 2),
                               int((self.height - image.size[1]) / 2),
                               int((self.width - image.size[0]) / 2) +
                               (self.width - image.size[0]) % 2,
                               int((self.height - image.size[1]) / 2) +
                               (self.height - image.size[1]) % 2),
                              padding_mode="reflect")(image)


class RandomImage(torch.nn.Module):
    """Replace a given image with random noise."""

    def __init__(self):
        super().__init__()

    def forward(self, image):
        return torch.rand_like(image)
