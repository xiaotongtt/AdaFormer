import os
import math
import time
import datetime
from functools import reduce

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from pytorch_msssim import ssim
# import pyiqa
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')
        for d in args.data_test:
            os.makedirs(self.dir + '/results/{}'.format(d), exist_ok=True)
        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def draw_features(self, dataset, filename, x, scale):
        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)
        _make_dir(self.dir + '/results/{}/errormap/'.format(dataset.dataset.name))
        save_filename = '{}/results/{}/errormap/{}_x{}_.png'.format(self.dir, dataset.dataset.name, filename, scale)
        img = x[0, 0, :, :]
        pmin = np.min(img) #if np.min(img) == 1.0 else 0.0
        pmax = np.max(img)
        # pmin = 0.0
        # pmax = 1.0
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        cv2.imwrite(save_filename, img)


    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            # filename = self.get_path(
            #     'results/{}'.format(dataset.dataset.name),
            #     '{}_x{}_'.format(filename, scale)
            # )
            filename = '{}/results/{}/{}_x{}_'.format(self.dir, dataset.dataset.name, filename, scale)
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                # misc.imsave('{}{}.png'.format(filename, p), ndarr)
                imageio.imsave('{}{}.png'.format(filename, p), ndarr)

    # def save_results(self, filename, save_list, scale):
    #     filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
    #     postfix = ('SR', 'LR', 'HR')
    #     for v, p in zip(save_list, postfix):
    #         normalized = v[0].data.mul(255 / self.args.rgb_range)
    #         ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
    #         misc.imsave('{}{}.png'.format(filename, p), ndarr)


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)



def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

# ====
# def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
#     """Calculate SSIM (structural similarity).
#     Ref:
#     Image quality assessment: From error visibility to structural similarity
#     The results are the same as that of the official released MATLAB code in
#     https://ece.uwaterloo.ca/~z70wang/research/ssim/.
#     For three-channel images, SSIM is calculated for each channel and then
#     averaged.
#     Args:
#         img1 (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#         crop_border (int): Cropped pixels in each edge of an image. These
#             pixels are not involved in the SSIM calculation.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'.
#             Default: 'HWC'.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
#     Returns:
#         float: ssim result.
#     """
#
#     assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
#     img1 = reorder_image(img1, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#
#     if crop_border != 0:
#         img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
#
#     if test_y_channel:
#         img1 = to_y_channel(img1)
#         img2 = to_y_channel(img2)
#
#     ssims = []
#     for i in range(img1.shape[2]):
#         ssims.append(_ssim(img1[..., i], img2[..., i]))
#     return np.array(ssims).mean()
#
#
# def _ssim(img1, img2):
#     """Calculate SSIM (structural similarity) for one channel images.
#     It is called by func:`calculate_ssim`.
#     Args:
#         img1 (ndarray): Images with range [0, 255] with order 'HWC'.
#         img2 (ndarray): Images with range [0, 255] with order 'HWC'.
#     Returns:
#         float: ssim result.
#     """
#
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
#
# def to_y_channel(img):
#     """Change to Y channel of YCbCr.
#     Args:
#         img (ndarray): Images with range [0, 255].
#     Returns:
#         (ndarray): Images with range [0, 255] (float type) without round.
#     """
#     img = img.astype(np.float32) / 255.
#     if img.ndim == 3 and img.shape[2] == 3:
#         img = bgr2ycbcr(img, y_only=True)
#         img = img[..., None]
#     return img * 255.
#
# def bgr2ycbcr(img, y_only=False):
#     """Convert a BGR image to YCbCr image.
#     The bgr version of rgb2ycbcr.
#     It implements the ITU-R BT.601 conversion for standard-definition
#     television. See more details in
#     https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
#     It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
#     In OpenCV, it implements a JPEG conversion. See more details in
#     https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
#     Args:
#         img (ndarray): The input image. It accepts:
#             1. np.uint8 type with range [0, 255];
#             2. np.float32 type with range [0, 1].
#         y_only (bool): Whether to only return Y channel. Default: False.
#     Returns:
#         ndarray: The converted YCbCr image. The output image has the same type
#             and range as input image.
#     """
#     img_type = img.dtype
#     img = _convert_input_type_range(img)
#     if y_only:
#         out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
#     else:
#         out_img = np.matmul(
#             img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
#     out_img = _convert_output_type_range(out_img, img_type)
#     return
#
# def _convert_output_type_range(img, dst_type):
#     """Convert the type and range of the image according to dst_type.
#     It converts the image to desired type and range. If `dst_type` is np.uint8,
#     images will be converted to np.uint8 type with range [0, 255]. If
#     `dst_type` is np.float32, it converts the image to np.float32 type with
#     range [0, 1].
#     It is mainly used for post-processing images in colorspace convertion
#     functions such as rgb2ycbcr and ycbcr2rgb.
#     Args:
#         img (ndarray): The image to be converted with np.float32 type and
#             range [0, 255].
#         dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
#             converts the image to np.uint8 type with range [0, 255]. If
#             dst_type is np.float32, it converts the image to np.float32 type
#             with range [0, 1].
#     Returns:
#         (ndarray): The converted image with desired type and range.
#     """
#     if dst_type not in (np.uint8, np.float32):
#         raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
#     if dst_type == np.uint8:
#         img = img.round()
#     else:
#         img /= 255.
#     return img.astype(dst_type)
#
# def _convert_input_type_range(img):
#     """Convert the type and range of the input image.
#     It converts the input image to np.float32 type and range of [0, 1].
#     It is mainly used for pre-processing the input image in colorspace
#     convertion functions such as rgb2ycbcr and ycbcr2rgb.
#     Args:
#         img (ndarray): The input image. It accepts:
#             1. np.uint8 type with range [0, 255];
#             2. np.float32 type with range [0, 1].
#     Returns:
#         (ndarray): The converted image with type of np.float32 and range of
#             [0, 1].
#     """
#     img_type = img.dtype
#     img = img.astype(np.float32)
#     if img_type == np.float32:
#         pass
#     elif img_type == np.uint8:
#         img /= 255.
#     else:
#         raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
#     return
# def reorder_image(img, input_order='HWC'):
#     """Reorder images to 'HWC' order.
#     If the input_order is (h, w), return (h, w, 1);
#     If the input_order is (c, h, w), return (h, w, c);
#     If the input_order is (h, w, c), return as it is.
#     Args:
#         img (ndarray): Input image.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'.
#             If the input image shape is (h, w), input_order will not have
#             effects. Default: 'HWC'.
#     Returns:
#         ndarray: reordered image.
#     """
#
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
#     if len(img.shape) == 2:
#         img = img[..., None]
#     if input_order == 'CHW':
#         img = img.transpose(1, 2, 0)
#     return


def calc_ssim(sr, hr, scale, rgb_range):
    # diff = (sr - hr).data.div(rgb_range)
    # sr = sr.data.div(rgb_range)
    # hr = hr.data.div(rgb_range)
    shave = scale
    hr = hr.clamp(0, 255)
    sr = sr.clamp(0, 255)
    hr_ycbcr = rgb_to_ycbcr(hr)
    sr_ycbcr = rgb_to_ycbcr(sr)
    hr = hr_ycbcr[:, 0:1, :, :]
    sr = sr_ycbcr[:, 0:1, :, :]
    sr = sr[:, :, shave:-shave, shave:-shave]
    hr = hr[:, :, shave:-shave, shave:-shave]
    ssim_val = ssim(sr, hr, size_average=True)
    return float(ssim_val)


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.
    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.
    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    image = image / 255. ## image in range (0, 1)
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    cb: torch.Tensor = -37.797 * r + -74.203 * g + 112.0 * b + 128.0
    cr: torch.Tensor = 112.0 * r + -93.786 * g + -18.214 * b + 128.0

    return torch.stack((y, cb, cr), -3)

def pyiqa_psnr(sr, hr, rgb_range=255.):
    iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(sr.device)
    hr = hr.clamp(0, 255) / 255.
    sr = sr.clamp(0, 255) / 255.
    score = iqa_metric(sr, hr)
    return score

def pyiqa_lpips(sr, hr, rgb_range=255.):
    iqa_metric = pyiqa.create_metric('lpips', device=torch.device('cuda'))
    hr = hr.clamp(0, 255) / 255.
    sr = sr.clamp(0, 255) / 255.
    score = iqa_metric(sr, hr)
    return score

