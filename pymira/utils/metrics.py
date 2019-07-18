import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def multi_class_score(one_class_fn, predictions, labels, one_hot=False, unindexed_classes=0):
    result = {}
    shape = labels.shape
    for label_index in range(shape[1] + unindexed_classes):
        if one_hot:
            class_predictions = torch.round(predictions[:, label_index, :, :, :])
        else:
            class_predictions = predictions.eq(label_index)
            class_predictions = class_predictions.squeeze(1)  # remove channel dim
        class_labels = labels.eq(label_index).float()
        class_labels = class_labels.squeeze(1)  # remove channel dim
        class_predictions = class_predictions.float()

        result[str(label_index)] = one_class_fn(class_predictions, class_labels).mean()

    return result


# Inefficient to do this twice, TODO: update multi_class_score to handle this
def hausdorff_distance(predictions, labels, one_hot=False, unindexed_classes=0, spacing=[1, 1, 1]):
    def one_class_hausdorff_distance(pred, lab):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        batch = pred.shape[0]
        result = []
        for i in range(batch):
            pred_img = sitk.GetImageFromArray(pred[i].cpu().numpy())
            pred_img.SetSpacing(spacing)
            lab_img = sitk.GetImageFromArray(lab[i].cpu().numpy())
            lab_img.SetSpacing(spacing)
            hausdorff_distance_filter.Execute(pred_img, lab_img)
            result.append(hausdorff_distance_filter.GetHausdorffDistance())
        return torch.tensor(np.asarray(result))

    return multi_class_score(one_class_hausdorff_distance, predictions, labels, one_hot=one_hot,
                             unindexed_classes=unindexed_classes)


def average_surface_distance(predictions, labels, one_hot=False, unindexed_classes=0, spacing=[1, 1, 1]):
    def one_class_average_surface_distance(pred, lab):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        batch = pred.shape[0]
        result = []
        for i in range(batch):
            pred_img = sitk.GetImageFromArray(pred[i].cpu().numpy())
            pred_img.SetSpacing(spacing)
            lab_img = sitk.GetImageFromArray(lab[i].cpu().numpy())
            lab_img.SetSpacing(spacing)
            hausdorff_distance_filter.Execute(pred_img, lab_img)
            result.append(hausdorff_distance_filter.GetAverageHausdorffDistance())
        return torch.tensor(np.asarray(result))

    return multi_class_score(one_class_average_surface_distance, predictions, labels, one_hot=one_hot,
                             unindexed_classes=unindexed_classes)


def dice_score(predictions, labels, one_hot=False, unindexed_classes=0):
    """ returns the dice score

    Args:
        predictions: one hot tensor [B, num_classes, D, H, W]
        labels: label tensor [B, 1, D, H, W]
    Returns:
        dict: ['label'] = [B, score]
    """

    def one_class_dice(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()

        return (2. * true_positive) / (p_flat.sum() + l_flat.sum())

    return multi_class_score(one_class_dice, predictions, labels, one_hot=one_hot, unindexed_classes=unindexed_classes)


def precision(predictions, labels, one_hot=False, unindexed_classes=0):
    def one_class_precision(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()
        return true_positive / p_flat.sum()

    return multi_class_score(one_class_precision, predictions, labels, one_hot=one_hot,
                             unindexed_classes=unindexed_classes)


def recall(predictions, labels, one_hot=False, unindexed_classes=0):
    def one_class_recall(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()
        negative = 1 - p_flat
        false_negative = (negative * l_flat).sum()
        return true_positive / (true_positive + false_negative)

    return multi_class_score(one_class_recall, predictions, labels, one_hot=one_hot,
                             unindexed_classes=unindexed_classes)


class Logger():
    def __init__(self, name, loss_names):
        self.name = name
        self.loss_names = loss_names
        self.epoch_logger = {}
        self.epoch_summary = {}
        self.epoch_number_logger = []
        self.reset_epoch_logger()
        self.reset_epoch_summary()

    def reset_epoch_logger(self):
        for loss_name in self.loss_names:
            self.epoch_logger[loss_name] = []

    def reset_epoch_summary(self):
        for loss_name in self.loss_names:
            self.epoch_summary[loss_name] = []

    def update_epoch_logger(self, loss_dict):
        for loss_name, loss_value in loss_dict.items():
            if loss_name not in self.loss_names:
                raise ValueError('Logger was not constructed to log {}'.format(loss_name))
            else:
                self.epoch_logger[loss_name].append(loss_value.item())

    def update_epoch_summary(self, epoch, reset=True):
        for loss_name in self.loss_names:
            self.epoch_summary[loss_name].append(np.mean(self.epoch_logger[loss_name]))
        self.epoch_number_logger.append(epoch)
        if reset:
            self.reset_epoch_logger()

    def get_latest_dict(self):
        latest = {}
        for loss_name in self.loss_names:
            latest[loss_name] = self.epoch_summary[loss_name][-1]
        return latest

    def get_epoch_logger(self):
        return self.epoch_logger

    def write_epoch_logger(self, location, index, loss_names, loss_labels, colours, linestyles=None, scales=None,
                           clear_plot=True):
        if linestyles is None:
            linestyles = ['-'] * len(colours)
        if scales is None:
            scales = [1] * len(colours)
        if not (len(loss_names) == len(loss_labels) and len(loss_labels) == len(colours) and len(colours) == len(
                linestyles) and len(linestyles) == len(scales)):
            raise ValueError('Length of all arg lists must be equal but got {} {} {} {} {}'.format(len(loss_names),
                                                                                                   len(loss_labels),
                                                                                                   len(colours),
                                                                                                   len(linestyles),
                                                                                                   len(scales)))

        for name, label, colour, linestyle, scale in zip(loss_names, loss_labels, colours, linestyles, scales):
            if scale == 1:
                plt.plot(range(0, len(self.epoch_logger[name])), self.epoch_logger[name], c=colour,
                         label=label, linestyle=linestyle)
            else:
                plt.plot(range(0, len(self.epoch_logger[name])), [scale * val for val in self.epoch_logger[name]],
                         c=colour,
                         label='{} x {}'.format(scale, label), linestyle=linestyle)
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('{}/{}.png'.format(location, index))
        if clear_plot:
            plt.clf()

    def print_latest(self, loss_names=None):
        print_str = '{}\tEpoch: {}\t'.format(self.name, self.epoch_number_logger[-1])
        if loss_names is None:
            loss_names = self.loss_names
        for loss_name in loss_names:
            if loss_name not in self.loss_names:
                raise ValueError('Logger was not constructed to log {}'.format(loss_name))
            else:
                print_str += '{}: {:.6f}\t'.format(loss_name, self.epoch_summary[loss_name][-1])
        print(print_str)
