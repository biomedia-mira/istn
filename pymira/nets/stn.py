import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class STN2D(nn.Module):

    def __init__(self, input_size, input_channels, device):
        super(STN2D, self).__init__()
        num_features = torch.prod(((((torch.tensor(input_size) - 4) / 2 - 4) / 2) - 4) / 2)
        self.device = device
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc = nn.Linear(32 * num_features, 32)

        self.theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).view(2, 3)

        # Regressor for the 2 * 3 affine matrix
        # self.affine_regressor = nn.Linear(32, 2 * 3)

        # initialize the weights/bias with identity transformation
        # self.affine_regressor.weight.data.zero_()
        # self.affine_regressor.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Regressor for individual parameters
        self.translation = nn.Linear(32, 2)
        self.rotation = nn.Linear(32, 1)
        self.scaling = nn.Linear(32, 2)
        self.shearing = nn.Linear(32, 1)

        # initialize the weights/bias with identity transformation
        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0], dtype=torch.float))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        self.shearing.weight.data.zero_()
        self.shearing.bias.data.copy_(torch.tensor([0], dtype=torch.float))

    def forward(self, x):
        xs = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        xs = F.avg_pool2d(F.relu(self.conv2(xs)), 2)
        xs = F.avg_pool2d(F.relu(self.conv3(xs)), 2)
        xs = xs.view(xs.size(0), -1)
        xs = F.relu(self.fc(xs))
        # theta = self.affine_regressor(xs).view(-1, 2, 3)
        self.theta = self.affine_matrix(xs)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img)

    def warp_image(self, img):
        grid = F.affine_grid(self.theta, img.size()).to(self.device)
        wrp = F.grid_sample(img, grid)

        return wrp

    def affine_matrix(self, x):
        b = x.size(0)

        # trans = self.translation(x)
        trans = torch.tanh(self.translation(x)) * 0.1
        translation_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 0, 2] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 2] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 2] = 1.0

        # rot = self.rotation(x)
        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
        rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
        rotation_matrix[:, 0, 0] = torch.cos(rot.view(-1))
        rotation_matrix[:, 0, 1] = -torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 0] = torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 1] = torch.cos(rot.view(-1))
        rotation_matrix[:, 2, 2] = 1.0

        # scale = F.softplus(self.scaling(x), beta=np.log(2.0))
        # scale = self.scaling(x)
        scale = torch.tanh(self.scaling(x)) * 0.2
        scaling_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
        # scaling_matrix[:, 0, 0] = scale[:, 0].view(-1)
        # scaling_matrix[:, 1, 1] = scale[:, 1].view(-1)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = 1.0

        # shear = self.shearing(x)
        shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)
        shearing_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
        shearing_matrix[:, 0, 0] = torch.cos(shear.view(-1))
        shearing_matrix[:, 0, 1] = -torch.sin(shear.view(-1))
        shearing_matrix[:, 1, 0] = torch.sin(shear.view(-1))
        shearing_matrix[:, 1, 1] = torch.cos(shear.view(-1))
        shearing_matrix[:, 2, 2] = 1.0

        # Affine transform
        matrix = torch.bmm(shearing_matrix, scaling_matrix)
        matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        matrix = torch.bmm(matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        # matrix = torch.bmm(translation_matrix, rotation_matrix)
        # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        # matrix = torch.bmm(matrix, scaling_matrix)
        # matrix = torch.bmm(matrix, shearing_matrix)

        # No-shear transform
        # matrix = torch.bmm(scaling_matrix, rotation_matrix)
        # matrix = torch.bmm(matrix, translation_matrix)

        # Rigid-body transform
        # matrix = torch.bmm(rotation_matrix, translation_matrix)

        return matrix[:, 0:2, :]


class BSplineSTN2D(nn.Module):
    """
    B-spline implementation inspired by https://github.com/airlab-unibas/airlab
    """

    def __init__(self, input_size, input_channels, device, control_point_spacing=(20, 20)):
        super(BSplineSTN2D, self).__init__()
        # Cuda params
        self.device = device
        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float

        self.input_size = input_size
        self.control_point_spacing = np.array(control_point_spacing)
        self.stride = self.control_point_spacing.astype(dtype=int).tolist()

        area = self.control_point_spacing[0] * self.control_point_spacing[1]
        self.area = area.astype(float)
        cp_grid_shape = np.ceil(np.divide(self.input_size, self.control_point_spacing)).astype(dtype=int)

        # new image size after convolution
        self.inner_image_size = np.multiply(self.control_point_spacing, cp_grid_shape) - (
                self.control_point_spacing - 1)

        # add one control point at each side
        cp_grid_shape = cp_grid_shape + 2

        # image size with additional control points
        self.new_image_size = np.multiply(self.control_point_spacing, cp_grid_shape) - (self.control_point_spacing - 1)

        # center image between control points
        image_size_diff = self.inner_image_size - self.input_size
        image_size_diff_floor = np.floor((np.abs(image_size_diff) / 2)) * np.sign(image_size_diff)
        crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2) * np.sign(image_size_diff)
        self.crop_start = crop_start.astype(dtype=int)
        self.crop_end = image_size_diff_floor.astype(dtype=int)

        self.cp_grid_shape = [2] + cp_grid_shape.tolist()

        self.num_cp_parameters = np.prod(self.cp_grid_shape)
        self.kernel = self.bspline_kernel().expand(2, *((np.ones(2 + 1, dtype=int) * -1).tolist()))
        self.kernel_size = np.asarray(self.kernel.size())[2:]
        self.padding = ((self.kernel_size - 1) / 2).astype(dtype=int).tolist()

        # Network params
        num_features = torch.prod(((((torch.tensor(input_size) - 4) / 2 - 4) / 2) - 4) / 2)
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=5).to(self.device)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5).to(self.device)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5).to(self.device)
        self.fc = nn.Linear(32 * num_features, self.num_cp_parameters).to(self.device)

    def gen_mesh_grid(self, h, w):
        # move into self to save compute?
        h_s = torch.linspace(-1, 1, h)
        w_s = torch.linspace(-1, 1, w)

        h_s, w_s = torch.meshgrid([h_s, w_s])

        mesh_grid = torch.stack([w_s, h_s])
        return mesh_grid.permute(1, 2, 0).to(self.device)  # h x w x 2

    def bspline_kernel(self, order=3):
        kernel_ones = torch.ones(1, 1, *self.control_point_spacing)
        kernel = kernel_ones

        for i in range(1, order + 1):
            kernel = F.conv2d(kernel, kernel_ones, padding=self.control_point_spacing.tolist()) / self.area

        return kernel.to(dtype=self.dtype, device=self.device)

    def compute_displacement(self, params):
        # compute dense displacement
        displacement = F.conv_transpose2d(params, self.kernel,
                                          padding=self.padding, stride=self.stride, groups=2)

        # crop displacement
        displacement = displacement[:, :,
                       self.control_point_spacing[0] + self.crop_start[0]:-self.control_point_spacing[0] -
                                                                          self.crop_end[0],
                       self.control_point_spacing[1] + self.crop_start[1]:-self.control_point_spacing[1] -
                                                                          self.crop_end[1]]

        return displacement.permute(0, 2, 3, 1)

    def get_theta(self, i):
        return self.control_points[i]

    def forward(self, x):
        b, c, h, w = x.shape
        xs = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        xs = F.avg_pool2d(F.relu(self.conv2(xs)), 2)
        xs = F.avg_pool2d(F.relu(self.conv3(xs)), 2)
        xs = xs.view(xs.size(0), -1)
        # cap the displacement field by (-1,1) this still allows for non-diffeomorphic transformations
        xs = torch.tanh(self.fc(xs)) * 0.2
        xs = xs.view(-1, *self.cp_grid_shape)

        self.displacement_field = self.compute_displacement(xs) + self.gen_mesh_grid(h, w).unsqueeze(0)
        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def warp_image(self, img):
        wrp = F.grid_sample(img, self.displacement_field)

        return wrp


class STN3D(nn.Module):

    def __init__(self, input_size, input_channels, device):
        super(STN3D, self).__init__()
        self.input_size = input_size
        num_features = torch.prod(((((torch.tensor(input_size) - 4) / 2 - 4) / 2) - 4) / 2)
        self.device = device
        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float

        self.conv1 = nn.Conv3d(input_channels, 8, kernel_size=5).to(self.device)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=5).to(self.device)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=5).to(self.device)
        self.fc = nn.Linear(32 * num_features, 32).to(self.device)
        self.theta = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=self.dtype).view(3, 4)

        # Regressor for the 3 * 4 affine matrix
        # self.affine_regressor = nn.Linear(32, 3 * 4)

        # initialize the weights/bias with identity transformation
        # self.affine_regressor.weight.data.zero_()
        # self.affine_regressor.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # Regressor for individual parameters
        self.translation = nn.Linear(32, 3).to(self.device)
        self.rotation = nn.Linear(32, 3).to(self.device)
        self.scaling = nn.Linear(32, 3).to(self.device)
        self.shearing = nn.Linear(32, 3).to(self.device)

        # initialize the weights/bias with identity transformation
        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.shearing.weight.data.zero_()
        self.shearing.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))

    def get_theta(self, i):
        return self.theta[i]

    def forward(self, x):
        xs = F.avg_pool3d(F.relu(self.conv1(x)), 2)
        xs = F.avg_pool3d(F.relu(self.conv2(xs)), 2)
        xs = F.avg_pool3d(F.relu(self.conv3(xs)), 2)
        xs = xs.view(xs.size(0), -1)
        xs = F.relu(self.fc(xs))
        # theta = self.affine_regressor(xs).view(-1, 3, 4)
        self.theta = self.affine_matrix(xs)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def gen_3d_mesh_grid(self, d, h, w):
        # move into self to save compute?
        d_s = torch.linspace(-1, 1, d)
        h_s = torch.linspace(-1, 1, h)
        w_s = torch.linspace(-1, 1, w)

        d_s, h_s, w_s = torch.meshgrid([d_s, h_s, w_s])
        one_s = torch.ones_like(w_s)

        mesh_grid = torch.stack([w_s, h_s, d_s, one_s])
        return mesh_grid  # 4 x d x h x w

    def affine_grid(self, theta, size):
        b, c, d, h, w = size
        mesh_grid = self.gen_3d_mesh_grid(d, h, w)
        mesh_grid = mesh_grid.unsqueeze(0)

        mesh_grid = mesh_grid.repeat(b, 1, 1, 1, 1)  # channel dim = 4
        mesh_grid = mesh_grid.view(b, 4, -1)
        mesh_grid = torch.bmm(theta, mesh_grid)  # channel dim = 3
        mesh_grid = mesh_grid.permute(0, 2, 1)  # move channel to last dim
        return mesh_grid.view(b, d, h, w, 3)

    def warp_image(self, img):
        grid = self.affine_grid(self.theta, img.size()).to(self.device)
        wrp = F.grid_sample(img, grid)

        return wrp

    def affine_matrix(self, x):
        b = x.size(0)

        # self.trans = self.translation(x)
        trans = torch.tanh(self.translation(x)) * 0.1
        translation_matrix = torch.zeros([b, 4, 4], dtype=torch.float)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 2, 2] = 1.0
        translation_matrix[:, 0, 3] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 3] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 3] = trans[:, 2].view(-1)
        translation_matrix[:, 3, 3] = 1.0

        # self.rot = self.rotation(x)
        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
        # rot Z
        angle_1 = rot[:, 0].view(-1)
        rotation_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_1[:, 0, 0] = torch.cos(angle_1)
        rotation_matrix_1[:, 0, 1] = -torch.sin(angle_1)
        rotation_matrix_1[:, 1, 0] = torch.sin(angle_1)
        rotation_matrix_1[:, 1, 1] = torch.cos(angle_1)
        rotation_matrix_1[:, 2, 2] = 1.0
        rotation_matrix_1[:, 3, 3] = 1.0
        # rot X
        angle_2 = rot[:, 1].view(-1)
        rotation_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_2[:, 1, 1] = torch.cos(angle_2)
        rotation_matrix_2[:, 1, 2] = -torch.sin(angle_2)
        rotation_matrix_2[:, 2, 1] = torch.sin(angle_2)
        rotation_matrix_2[:, 2, 2] = torch.cos(angle_2)
        rotation_matrix_2[:, 0, 0] = 1.0
        rotation_matrix_2[:, 3, 3] = 1.0
        # rot Z
        angle_3 = rot[:, 2].view(-1)
        rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
        rotation_matrix_3[:, 0, 1] = -torch.sin(angle_3)
        rotation_matrix_3[:, 1, 0] = torch.sin(angle_3)
        rotation_matrix_3[:, 1, 1] = torch.cos(angle_3)
        rotation_matrix_3[:, 2, 2] = 1.0
        rotation_matrix_3[:, 3, 3] = 1.0

        rotation_matrix = torch.bmm(rotation_matrix_1, rotation_matrix_2)
        rotation_matrix = torch.bmm(rotation_matrix, rotation_matrix_3)

        # self.scale = F.softplus(self.scaling(x), beta=np.log(2.0))
        # self.scale = self.scaling(x)
        scale = torch.tanh(self.scaling(x)) * 0.2
        scaling_matrix = torch.zeros([b, 4, 4], dtype=torch.float)
        # scaling_matrix[:, 0, 0] = self.scale[:, 0].view(-1)
        # scaling_matrix[:, 1, 1] = self.scale[:, 1].view(-1)
        # scaling_matrix[:, 2, 2] = self.scale[:, 2].view(-1)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = torch.exp(scale[:, 2].view(-1))
        scaling_matrix[:, 3, 3] = 1.0

        # self.shear = self.shearing(x)
        shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)

        shear_1 = shear[:, 0].view(-1)
        shearing_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float)
        shearing_matrix_1[:, 1, 1] = torch.cos(shear_1)
        shearing_matrix_1[:, 1, 2] = -torch.sin(shear_1)
        shearing_matrix_1[:, 2, 1] = torch.sin(shear_1)
        shearing_matrix_1[:, 2, 2] = torch.cos(shear_1)
        shearing_matrix_1[:, 0, 0] = 1.0
        shearing_matrix_1[:, 3, 3] = 1.0

        shear_2 = shear[:, 1].view(-1)
        shearing_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float)
        shearing_matrix_2[:, 0, 0] = torch.cos(shear_2)
        shearing_matrix_2[:, 0, 2] = torch.sin(shear_2)
        shearing_matrix_2[:, 2, 0] = -torch.sin(shear_2)
        shearing_matrix_2[:, 2, 2] = torch.cos(shear_2)
        shearing_matrix_2[:, 1, 1] = 1.0
        shearing_matrix_2[:, 3, 3] = 1.0

        shear_3 = shear[:, 2].view(-1)
        shearing_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float)
        shearing_matrix_3[:, 0, 0] = torch.cos(shear_3)
        shearing_matrix_3[:, 0, 1] = -torch.sin(shear_3)
        shearing_matrix_3[:, 1, 0] = torch.sin(shear_3)
        shearing_matrix_3[:, 1, 1] = torch.cos(shear_3)
        shearing_matrix_3[:, 2, 2] = 1.0
        shearing_matrix_3[:, 3, 3] = 1.0

        shearing_matrix = torch.bmm(shearing_matrix_1, shearing_matrix_2)
        shearing_matrix = torch.bmm(shearing_matrix, shearing_matrix_3)

        # Affine transform
        matrix = torch.bmm(shearing_matrix, scaling_matrix)
        matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        matrix = torch.bmm(matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        # matrix = torch.bmm(translation_matrix, rotation_matrix)
        # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        # matrix = torch.bmm(matrix, scaling_matrix)
        # matrix = torch.bmm(matrix, shearing_matrix)

        # No-shear transform
        # matrix = torch.bmm(scaling_matrix, rotation_matrix)
        # matrix = torch.bmm(matrix, translation_matrix)

        # Rigid-body transform
        # matrix = torch.bmm(rotation_matrix, translation_matrix)

        # print('shearing')
        # print(shearing_matrix[0, :, :])
        # print('scaling')
        # print(scaling_matrix[0, :, :])
        # print('rotation')
        # print(rotation_matrix[0, :, :])
        # print('translation')
        # print(translation_matrix[0, :, :])
        # print('affine')
        # print(matrix[0, :, :])

        return matrix[:, 0:3, :]


class BSplineSTN3D(nn.Module):
    """
    B-spline implementation inspired by https://github.com/airlab-unibas/airlab
    """

    def __init__(self, input_size, input_channels, device, control_point_spacing=(10, 10, 10)):
        super(BSplineSTN3D, self).__init__()
        # Cuda params
        self.device = device
        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float

        self.input_size = input_size
        self.control_point_spacing = np.array(control_point_spacing)
        self.stride = self.control_point_spacing.astype(dtype=int).tolist()

        area = self.control_point_spacing[0] * self.control_point_spacing[1] * self.control_point_spacing[2]
        self.area = area.astype(float)
        cp_grid_shape = np.ceil(np.divide(self.input_size, self.control_point_spacing)).astype(dtype=int)

        # new image size after convolution
        self.inner_image_size = np.multiply(self.control_point_spacing, cp_grid_shape) - (
                self.control_point_spacing - 1)

        # add one control point at each side
        cp_grid_shape = cp_grid_shape + 2

        # image size with additional control points
        self.new_image_size = np.multiply(self.control_point_spacing, cp_grid_shape) - (self.control_point_spacing - 1)

        # center image between control points
        image_size_diff = self.inner_image_size - self.input_size
        image_size_diff_floor = np.floor((np.abs(image_size_diff) / 2)) * np.sign(image_size_diff)
        crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2) * np.sign(image_size_diff)
        self.crop_start = crop_start.astype(dtype=int)
        self.crop_end = image_size_diff_floor.astype(dtype=int)

        self.cp_grid_shape = [3] + cp_grid_shape.tolist()

        self.num_control_points = np.prod(self.cp_grid_shape)
        self.kernel = self.bspline_kernel_3d().expand(3, *((np.ones(3 + 1, dtype=int) * -1).tolist()))
        self.kernel_size = np.asarray(self.kernel.size())[2:]
        self.padding = ((self.kernel_size - 1) / 2).astype(dtype=int).tolist()

        # Network params
        num_features = torch.prod(((((torch.tensor(input_size) - 4) / 2 - 4) / 2) - 4) / 2)
        self.conv1 = nn.Conv3d(input_channels, 8, kernel_size=5).to(self.device)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=5).to(self.device)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=5).to(self.device)
        self.fc = nn.Linear(32 * num_features, self.num_control_points).to(self.device)

    def gen_3d_mesh_grid(self, d, h, w):
        # move into self to save compute?
        d_s = torch.linspace(-1, 1, d)
        h_s = torch.linspace(-1, 1, h)
        w_s = torch.linspace(-1, 1, w)

        d_s, h_s, w_s = torch.meshgrid([d_s, h_s, w_s])

        mesh_grid = torch.stack([w_s, h_s, d_s])
        return mesh_grid.permute(1, 2, 3, 0).to(self.device)  # d x h x w x 3

    def bspline_kernel_3d(self, order=3):
        kernel_ones = torch.ones(1, 1, *self.control_point_spacing)
        kernel = kernel_ones

        for i in range(1, order + 1):
            kernel = F.conv3d(kernel, kernel_ones, padding=self.control_point_spacing.tolist()) / self.area

        return kernel.to(dtype=self.dtype, device=self.device)

    def compute_displacement(self, params):
        # compute dense displacement
        displacement = F.conv_transpose3d(params, self.kernel,
                                          padding=self.padding, stride=self.stride, groups=3)

        # crop displacement
        displacement = displacement[:, :,
                       self.control_point_spacing[0] + self.crop_start[0]:-self.control_point_spacing[0] -
                                                                          self.crop_end[0],
                       self.control_point_spacing[1] + self.crop_start[1]:-self.control_point_spacing[1] -
                                                                          self.crop_end[1],
                       self.control_point_spacing[2] + self.crop_start[2]:-self.control_point_spacing[2] -
                                                                          self.crop_end[2]]

        return displacement.permute(0, 2, 3, 4, 1)

    def get_theta(self, i):
        return self.control_points[i]

    def forward(self, x):
        b, c, d, h, w = x.shape
        xs = F.avg_pool3d(F.relu(self.conv1(x)), 2)
        xs = F.avg_pool3d(F.relu(self.conv2(xs)), 2)
        xs = F.avg_pool3d(F.relu(self.conv3(xs)), 2)
        xs = xs.view(xs.size(0), -1)
        self.regularisation_loss = 30.0 * torch.mean(torch.abs(xs))
        # cap the displacement field by (-1,1) this still allows for non-diffeomorphic transformations
        xs = torch.tanh(self.fc(xs)) * 0.2
        xs = xs.view(-1, *self.cp_grid_shape)

        self.displacement_field = self.compute_displacement(xs) + self.gen_3d_mesh_grid(d, h, w).unsqueeze(0)
        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def warp_image(self, img):
        wrp = F.grid_sample(img, self.displacement_field)

        return wrp
