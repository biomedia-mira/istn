import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader


class ImageRegistrationDataset(Dataset):
    """Dataset for pairwise image registration."""

    def __init__(self, csv_file_img, csv_file_msk=None, normalizer=None, resampler=None):
        """
        Args:
        :param csv_file_img (string): Path to csv file with image filenames.
        :param csv_file_msk (string): Path to csv file with mask filenames.
        :param normalizer (callable, optional): Optional transform to be applied on each image.
        :param resampler (callable, optional): Optional transform to be applied on each image.
        """
        self.data = pd.read_csv(csv_file_img)

        if csv_file_msk:
            self.msk_data = pd.read_csv(csv_file_msk)

        self.samples = []
        for idx in range(len(self.data)):
            src_path = self.data.iloc[idx, 0]
            trg_path = self.data.iloc[idx, 1]

            print('Reading source image ' + src_path)
            source = sitk.ReadImage(src_path, sitk.sitkFloat32)

            print('Reading target image ' + trg_path)
            target = sitk.ReadImage(trg_path, sitk.sitkFloat32)

            source_msk = sitk.GetImageFromArray(np.ones(source.GetSize()[::-1]))
            target_msk = sitk.GetImageFromArray(np.ones(target.GetSize()[::-1]))

            if csv_file_msk:
                src_msk_path = self.msk_data.iloc[idx, 0]
                trg_msk_path = self.msk_data.iloc[idx, 1]

                print('Reading source mask ' + src_msk_path)
                source_msk = sitk.ReadImage(src_msk_path, sitk.sitkFloat32)
                source_msk.CopyInformation(source)

                print('Reading target mask ' + trg_msk_path)
                target_msk = sitk.ReadImage(trg_msk_path, sitk.sitkFloat32)
                target_msk.CopyInformation(target)

            if normalizer:
                source = normalizer(source, source_msk)
                target = normalizer(target, target_msk)

            if resampler:
                source = resampler(source)
                target = resampler(target)
                source_msk = resampler(source_msk)
                target_msk = resampler(target_msk)

            if len(source.GetSize()) == 3:
                source.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
                target.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
            else:
                source.SetDirection((1, 0, 0, 1))
                target.SetDirection((1, 0, 0, 1))

            source.SetOrigin(np.zeros(len(source.GetOrigin())))
            target.SetOrigin(np.zeros(len(target.GetOrigin())))
            source_msk.CopyInformation(source)
            target_msk.CopyInformation(target)

            sample = {'source': source, 'target': target, 'source_msk': source_msk, 'target_msk': target_msk}

            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.samples[item]

        source = torch.from_numpy(sitk.GetArrayFromImage(sample['source'])).unsqueeze(0)
        target = torch.from_numpy(sitk.GetArrayFromImage(sample['target'])).unsqueeze(0)
        source_msk = torch.from_numpy(sitk.GetArrayFromImage(sample['source_msk'])).unsqueeze(0)
        target_msk = torch.from_numpy(sitk.GetArrayFromImage(sample['target_msk'])).unsqueeze(0)

        return {'source': source, 'target': target, 'source_msk': source_msk, 'target_msk': target_msk}

    def get_sample(self, item):
        return self.samples[item]


class ImageSegRegDataset(Dataset):
    """Dataset for pairwise image registration with segmentation loss."""

    def __init__(self, csv_file_img, csv_file_seg, csv_file_msk=None, normalizer_img=None, resampler_img=None, normalizer_seg=None, resampler_seg=None):
        """
        Args:
        :param csv_file_img (string): Path to csv file with image filenames.
        :param csv_file_seg (string): Path to csv file with segmentation filenames.
        :param csv_file_msk (string): Path to csv file with mask filenames.
        :param normalizer_img (callable, optional): Optional transform to be applied on each image.
        :param resampler_img (callable, optional): Optional transform to be applied on each image.
        :param normalizer_seg (callable, optional): Optional transform to be applied on each segmentation.
        :param resampler_seg (callable, optional): Optional transform to be applied on each segmentation.
        """
        self.img_data = pd.read_csv(csv_file_img)
        self.seg_data = pd.read_csv(csv_file_seg)

        if csv_file_msk:
            self.msk_data = pd.read_csv(csv_file_msk)

        self.samples = []
        for idx in range(len(self.img_data)):
            src_path = self.img_data.iloc[idx, 0]
            trg_path = self.img_data.iloc[idx, 1]

            src_seg_path = self.seg_data.iloc[idx, 0]
            trg_seg_path = self.seg_data.iloc[idx, 1]

            print('Reading source image ' + src_path)
            source = sitk.ReadImage(src_path, sitk.sitkFloat32)

            print('Reading target image ' + trg_path)
            target = sitk.ReadImage(trg_path, sitk.sitkFloat32)

            print('Reading source segmentation ' + src_seg_path)
            source_seg = sitk.ReadImage(src_seg_path, sitk.sitkFloat32)

            print('Reading target segmentation ' + trg_seg_path)
            target_seg = sitk.ReadImage(trg_seg_path, sitk.sitkFloat32)

            source_msk = sitk.GetImageFromArray(np.ones(source.GetSize()[::-1]))
            target_msk = sitk.GetImageFromArray(np.ones(target.GetSize()[::-1]))

            if csv_file_msk:
                src_msk_path = self.msk_data.iloc[idx, 0]
                trg_msk_path = self.msk_data.iloc[idx, 1]

                print('Reading source mask ' + src_msk_path)
                source_msk = sitk.ReadImage(src_msk_path, sitk.sitkFloat32)
                source_msk.CopyInformation(source)

                print('Reading target mask ' + trg_msk_path)
                target_msk = sitk.ReadImage(trg_msk_path, sitk.sitkFloat32)
                target_msk.CopyInformation(target)

            if normalizer_img:
                source = normalizer_img(source, source_msk)
                target = normalizer_img(target, target_msk)

            if resampler_img:
                source = resampler_img(source)
                target = resampler_img(target)
                source_msk = resampler_img(source_msk)
                target_msk = resampler_img(target_msk)

            if normalizer_seg:
                source_seg = normalizer_seg(source_seg)
                target_seg = normalizer_seg(target_seg)

            if resampler_seg:
                source_seg = resampler_seg(source_seg)
                target_seg = resampler_seg(target_seg)

            if len(source.GetSize()) == 3:
                source.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
                target.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
            else:
                source.SetDirection((1, 0, 0, 1))
                target.SetDirection((1, 0, 0, 1))

            source.SetOrigin(np.zeros(len(source.GetOrigin())))
            target.SetOrigin(np.zeros(len(target.GetOrigin())))
            source_seg.CopyInformation(source)
            target_seg.CopyInformation(target)
            source_msk.CopyInformation(source)
            target_msk.CopyInformation(target)

            sample = {'source': source, 'target': target, 'source_seg': source_seg, 'target_seg': target_seg, 'source_msk': source_msk, 'target_msk': target_msk}

            self.samples.append(sample)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, item):
        sample = self.samples[item]

        source = torch.from_numpy(sitk.GetArrayFromImage(sample['source'])).unsqueeze(0)
        target = torch.from_numpy(sitk.GetArrayFromImage(sample['target'])).unsqueeze(0)
        source_seg = torch.from_numpy(sitk.GetArrayFromImage(sample['source_seg'])).unsqueeze(0)
        target_seg = torch.from_numpy(sitk.GetArrayFromImage(sample['target_seg'])).unsqueeze(0)
        source_msk = torch.from_numpy(sitk.GetArrayFromImage(sample['source_msk'])).unsqueeze(0)
        target_msk = torch.from_numpy(sitk.GetArrayFromImage(sample['target_msk'])).unsqueeze(0)

        return {'source': source, 'target': target, 'source_seg': source_seg, 'target_seg': target_seg, 'source_msk': source_msk, 'target_msk': target_msk}

    def get_sample(self, item):
        return self.samples[item]
