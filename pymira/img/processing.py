import numpy as np
import SimpleITK as sitk


def zero_mean_unit_var(image, mask=None, fill_value=0):
    """Normalizes an image to zero mean and unit variance."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    mean = np.mean(img_array[msk_array>0])
    std = np.std(img_array[msk_array>0])

    if std > 0:
        img_array = (img_array - mean) / std
        img_array[msk_array==0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def range_matching(image, mask=None, low_percentile=4, high_percentile=96, fill_value=0):
    """Normalizes an image by mapping the low_percentile to zero, and the high_percentile to one."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    lo_p = np.percentile(img_array[msk_array>0], low_percentile)
    hi_p = np.percentile(img_array[msk_array>0], high_percentile)

    img_array = (img_array - lo_p) / (hi_p - lo_p)
    img_array[msk_array == 0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def zero_one(image, mask=None, fill_value=0):
    """Normalizes an image by mapping the min to zero, and max to one."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    min_value = np.min(img_array[msk_array>0])
    max_value = np.max(img_array[msk_array>0])

    img_array = (img_array - min_value) / (max_value - min_value)
    img_array[msk_array == 0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def threshold_zero(image, mask=None, fill_value=0):
    """Thresholds an image at zero."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array > 0
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    img_array[msk_array == 0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def same_image_domain(image1, image2):
    """Checks whether two images cover the same physical domain."""

    same_size = image1.GetSize() == image2.GetSize()
    same_spacing = image1.GetSpacing() == image2.GetSpacing()
    same_origin = image1.GetOrigin() == image2.GetOrigin()
    same_direction = image1.GetDirection() == image2.GetDirection()

    return same_size and same_spacing and same_origin and same_direction


def reorient_image(image):
    """Reorients an image to standard radiology view."""

    dir = np.array(image.GetDirection()).reshape(len(image.GetSize()), -1)
    ind = np.argmax(np.abs(dir), axis=0)
    new_size = np.array(image.GetSize())[ind]
    new_spacing = np.array(image.GetSpacing())[ind]
    new_extent = new_size * new_spacing
    new_dir = dir[:, ind]

    flip = np.diag(new_dir) < 0
    flip_diag = flip * -1
    flip_diag[flip_diag == 0] = 1
    flip_mat = np.diag(flip_diag)

    new_origin = np.array(image.GetOrigin()) + np.matmul(new_dir, (new_extent * flip))
    new_dir = np.matmul(new_dir, flip_mat)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing.tolist())
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(new_dir.flatten().tolist())
    resample.SetOutputOrigin(new_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    return resample.Execute(image)


def resample_image_to_ref(image, ref, is_label=False, pad_value=0):
    """Resamples an image to match the resolution and size of a given reference image."""

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        #resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(image)


def resample_image(image, out_spacing=(1.0, 1.0, 1.0), out_size=None, is_label=False, pad_value=0):
    """Resamples an image to given element spacing and output size."""

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    if out_size is None:
        out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    else:
        out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        #resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(sitk.Cast(image, sitk.sitkFloat32))

