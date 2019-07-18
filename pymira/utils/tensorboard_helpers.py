import torchvision


def volume_to_batch_image(volume, normalize=True, dim='D', batch=0):
    """ Helper function that, given a 5 D tensor, converts it to a 4D
    tensor by choosing element batch, and moves the dim into the batch
    dimension, this then allows the slices to be tiled for tensorboard

    Args:
        volume: volume to be viewed

    Returns:
        3D tensor (already tiled)
    """
    if batch >= volume.shape[0]:
        raise ValueError('{} batch index too high'.format(batch))
    if dim == 'D':
        image = volume[batch, :, :, :, :].permute(1, 0, 2, 3)
    elif dim == 'H':
        image = volume[batch, :, :, :, :].permute(2, 0, 1, 3)
    elif dim == 'W':
        image = volume[batch, :, :, :, :].permute(3, 0, 1, 2)
    else:
        raise ValueError('{} dim not supported'.format(dim))
    if normalize:
        return torchvision.utils.make_grid(normalize_to_0_1(image))
    else:
        return torchvision.utils.make_grid(image)

def normalize_to_0_1(volume):
    """
        Normalize the image to (0,1) to be viewed as a greyscale
    Args:
        volume: 4D tensor (B, C, H, W)

    Returns:
        Normalized volume
    """
    max_val = volume.max()
    min_val = volume.min()
    return (volume - min_val) / (max_val - min_val)
