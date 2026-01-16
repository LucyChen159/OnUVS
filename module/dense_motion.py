from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian, to_homogeneous, from_homogeneous
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)
# img = flow_to_image(flow)
# plt.imshow(img)
# plt.show()

class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving 从 kp_source 和 kp_driving 给出的稀疏运动表示预测密集运动的模块
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)  # (8, 10, 64, 64) 10 keypoints
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)  # (8, 10, 64, 64)
        heatmap = gaussian_driving - gaussian_source  # keypoints difference

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())#(4,1,64,64)
        heatmap = torch.cat([zeros, heatmap], dim=1)  # (8, 11, 64, 64)
        heatmap = heatmap.unsqueeze(2)  # (8, 11, 1, 64, 64)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source, bg_param):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())  # (64, 64, 2)
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)  # (4, 5, 64, 64, 2)
        if 'jacobian' in kp_driving:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))  # Eq 5 (4, 5, 2, 2) 
            #torch.inverse:计算逆矩阵 torch.matmul：矩阵相乘
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3) # jacobian.shape torch.Size([4, 5, 1, 1, 2, 2])
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)  # (4, 5, 64, 64, 2, 2)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)  # (4, 5, 64, 64, 2)

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)  # (4, 5, 64, 64, 2)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1) # torch.Size([4, 1, 64, 64, 2])
        # print(f"identity_grid1:{identity_grid}, {identity_grid.shape}")

        # affine background transformation
        if not (bg_param is None):            
            identity_grid = to_homogeneous(identity_grid)
            # print(f"identity_grid to_homogeneous:{identity_grid}, {identity_grid.shape}")
            identity_grid = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid.unsqueeze(-1)).squeeze(-1)
            # print(f"identity_grid matmul:{identity_grid}, {identity_grid.shape}")
            # print(f"bg_param matmul:{bg_param}, {bg_param.shape}")
            identity_grid = from_homogeneous(identity_grid)
            # print(f"identity_grid from_homogeneous:{identity_grid}, {identity_grid.shape}")

        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)  # (4, 6, 64, 64, 2)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1) #torch.Size([4, 6, 1, 1, 64, 64])
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w) #torch.Size([24, 1, 64, 64])
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1)) #torch.Size([24, 64, 64, 2])
        # 可视化光流
        # for i in range(sparse_motions.shape[0]):
        #     a = sparse_motions[i].cpu().detach().numpy()
        #     img = flow_to_image(a)
        #     plt.imshow(img)
        #     plt.savefig(f'/data2/zhouhan/video/a-paper/heart_vit_bg/modules/flow_{i}.png')

        sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=True) #应用双线性插值，把输入的tensor转换为指定大小
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed  # (8, 11, 3, 64, 64)

    def dropout_softmax(self, X, P):
        '''
        Dropout for TPS transformations. Eq(7) and Eq(8) in the paper.
        '''
        drop = (torch.rand(X.shape[0],X.shape[1]) < (1-P)).type(X.type()).to(X.device)
        drop[..., 0] = 1
        drop = drop.repeat(X.shape[2],X.shape[3],1,1).permute(2,3,0,1)

        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:,1:,...] /= (1-P)
        mask_bool =(drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition  

    def forward(self, source_image, kp_driving, kp_source,  bg_param = None, dropout_flag=False, dropout_p = 0):  # kp_driving: value (8, 10, 2), jacobian (8, 10, 2, 2)
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)   # (4, 6, 1, 64, 64)
        # 可视化heatmap
        # for i in range(12):
        #     a = heatmap_representation[int(i/6)][i%6][0]
        #     a = a.cpu().detach().numpy()
        #     fig = sns.heatmap(a, cmap="crest")
        #     heatmap = fig.get_figure()
        #     heatmap.savefig(f"/data2/zhouhan/video/a-paper/heart_vit_bg/modules/{i}.png")
        #     heatmap.clear()
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source, bg_param)  # (4, 6, 64, 64, 2)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)  # (4, 6, 3, 64, 64)
        out_dict['sparse_deformed'] = deformed_source#根据偏移矩阵将原始图像逐像素扭曲warp到目标位置，形成新的图片

        input = torch.cat([heatmap_representation, deformed_source], dim=2)  # (4, 6, 4, 64, 64)
        input = input.view(bs, -1, h, w)   # (4, 12, 64, 64)

        prediction = self.hourglass(input)  # (4, 76, 64, 64)

        mask = self.mask(prediction)  
        
        if(dropout_flag):
            mask = self.dropout_softmax(mask, dropout_p)
        else:
            mask = F.softmax(mask, dim=1)

        out_dict['mask'] = mask
        
        # Combine the K+1 transformations
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation  # (8, 64, 64, 2)

        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map  # (8, 1, 64, 64)

        return out_dict
