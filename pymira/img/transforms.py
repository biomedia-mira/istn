from pymira.img.processing import resample_image


class Resampler(object):
    """Resamples an image to given element spacing and size."""

    def __init__(self, out_spacing, out_size=None, is_label=False):
        """
        Args:
        :param out_spacing (tuple): Output element spacing.
        :param out_size (tuple, option): Output image size.
        :param is_label (boolean, option): Indicates label maps with nearest neighbor interpolation.
        """
        self.out_spacing = out_spacing
        self.out_size = out_size
        self.is_label = is_label

    def __call__(self, image):
        image_resampled = resample_image(image, self.out_spacing, self.out_size, self.is_label)

        return image_resampled


class Normalizer(object):
    """Normalizes image intensities with a given function."""

    def __init__(self, transform):
        """
        Args:
        :param transform (callable): Intensity normalization function.
        """
        self.transform = transform

    def __call__(self, image, mask=None):
        image_normalized = self.transform(image, mask)

        return image_normalized

