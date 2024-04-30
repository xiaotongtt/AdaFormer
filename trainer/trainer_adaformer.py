import os
import math
# from torchsummaryX import summary
import time
from decimal import Decimal
import utility
import torch.nn as nn
import torch
import torch.nn.utils as utils
from einops import rearrange
from tqdm import tqdm
# import pyiqa
import numpy as np
# from flops import FLOPs, chop_EF_EDSR

def make_trainer(args, loader, my_model, my_loss, ckp):
    return Trainer(args, loader, my_model, my_loss, ckp)


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):

        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.stop_epoch = args.stop_epoch
        # print("??",self.scheduler.get_lr()[0])
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = args.lr
                # print("??", self.scheduler.get_lr()[0])
            for _ in range(len(ckp.log)): self.scheduler.step()
            print(len(ckp.log))
            self.ckp.log = ckp.log[:len(ckp.log)]
            # print("??",self.scheduler.get_lr()[0])

        self.error_last = 1e8

        self.var_weight = 1


    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        if epoch >= self.stop_epoch:
            print("stop var")
            for name, param in self.model.model.body.var.named_parameters():
                # print(name)
                param.requirs_grad = False
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr, halting_mask, _, _ = self.model(lr, 0)
            loss = self.loss(sr, hr, halting_mask=halting_mask, scale=self.scale[0])

            loss.backward()
            # loss.backward(retain_graph=True)
            self.optimizer.step()
            # if loss.item() < self.args.skip_threshold * self.error_last:
            #     loss.backward()
            #     self.optimizer.step()
            # else:
            #     print('Skip this batch {}! (Loss: {})'.format(
            #         batch + 1, loss.item()
            #     ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch  # + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    eval_acc = 0
                    eval_lpips = 0
                    eval_ssim = 0
                    d.dataset.set_scale(idx_scale)
                    tqdm_test = tqdm(d, ncols=80)
                    time_sum = []
                    sparse_global_avg = []
                    sparse_local_avg = []
                    total_flops = 0
                    transformer_flops = 0

                    for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                        # if idx_img > 11: break
                        filename = filename[0]
                        #print("filename:", filename)
                        no_eval = (hr.nelement() == 1)
                        if not no_eval:
                            lr, hr = self.prepare([lr, hr])
                        else:
                            lr = self.prepare([lr])[0]
                        sparse_global_one = []
                        sparse_local_one = []
                        B, _, H, W = lr.shape
                        # print(H // 48, W // 48)
                        torch.cuda.synchronize()
                        time_start = time.time()
                        sr, halting_mask, global_mask_list, local_mask_list = self.model.model(lr)
                        torch.cuda.synchronize()
                        time_end = time.time()
                        # time_sum += (time_end - time_start)
                        time_sum.append(time_end - time_start)

                        sr = utility.quantize(sr, self.args.rgb_range)

                        diff = abs(hr - sr)
                        if diff.size(1) > 1:
                            convert = diff.new(1, 3, 1, 1)
                            convert[0, 0, 0, 0] = 65.738
                            convert[0, 1, 0, 0] = 129.057
                            convert[0, 2, 0, 0] = 25.064
                            # var.mul_(convert).div_(256)
                            # var = var.sum(dim=1, keepdim=True)

                            diff.mul_(convert).div_(256)
                            diff = diff.sum(dim=1, keepdim=True)

                        save_list = [sr]
                        if not no_eval:
                            cur_psnr = utility.calc_psnr(
                                sr, hr, scale, self.args.rgb_range,
                                benchmark=d.dataset.benchmark
                            )

                            # cur_psnr = utility.pyiqa_psnr(sr, hr)
                            cur_ssim = utility.calc_ssim(sr, hr, scale, self.args.rgb_range,)
                            # cur_lpips = utility.pyiqa_lpips(sr, hr)

                            eval_acc += cur_psnr
                            eval_ssim += cur_ssim
                            # eval_lpips += cur_lpips
                            # save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename, save_list, scale)
                            # self.ckp.draw_features(d, filename[0] + '_gt', diff.cpu().numpy(), scale)
                            self.ckp.draw_features(d, filename + '_gt', diff.cpu().numpy(), scale)

                            for i, (var, global_mask, local_mask) in enumerate(zip(halting_mask, global_mask_list, local_mask_list)):
                                # print(var.shape)
                                var = var[:, :, :H, :W]
                                global_mask = global_mask[:, :, :H, :W]
                                local_mask = local_mask[:, :, :H, :W]

                                self.ckp.draw_features(d, filename + '_{}'.format(i), var.float().cpu().numpy(),
                                                       scale)
                                self.ckp.draw_features(d, filename + '_global_mask_{}'.format(i),
                                                       global_mask.float().cpu().numpy(),
                                                       scale)
                                self.ckp.draw_features(d, filename + '_local_mask_{}'.format(i),
                                                       local_mask.float().cpu().numpy(),
                                                       scale)

                        # =======稀疏性计算========
                        for i, (global_mask, local_mask) in enumerate(zip(global_mask_list, local_mask_list)):
                            ws = 8
                            g_h, g_w = global_mask.shape[2:]
                            hg, wg = g_h // ws, g_w // ws
                            local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg,
                                                   hs=ws, w=wg, ws=ws).squeeze(-1)
                            local_mask = local_mask.max(dim=-1).values
                            sparse_global_one.append(global_mask.sum() / (H * W))
                            sparse_local_one.append(local_mask.sum() / (hg * wg))

                        flops_dict = {'global_mask': global_mask_list, 'local_mask': local_mask_list}

                        flops_return_dict = self.model.model.flops((H, W), scale, flops_dict)
                        total_flops += flops_return_dict['flops'] / 1000000000
                        transformer_flops += flops_return_dict['transformer_flops'] / 1000000000
                        sparse_global_avg.append(sparse_global_one)
                        sparse_local_avg.append(sparse_local_one)
                        # print(cur_psnr)

                    sum_time = sum(time_sum) - max(time_sum) - min(time_sum)
                    print(sum_time / (len(d) - 2))
                    # print(sum_time / 10)
                    print("total flops:", total_flops / len(d))
                    print("transformer flops:", transformer_flops / len(d))
                    self.ckp.log[-1, idx_scale] = eval_acc / len(d)
                    eval_ssim = eval_ssim / len(d)
                    # eval_lpips = eval_lpips / len(d)
                    best = self.ckp.log.max(0)
                    sparse_global_avg = torch.tensor(sparse_global_avg)
                    sparse_local_avg = torch.tensor(sparse_local_avg)
                    print("global mask",1 - sparse_global_avg.mean(0), (1 - sparse_global_avg.mean(0)).mean())
                    print("local mask", 1 - sparse_local_avg.mean(0), (1 - sparse_local_avg.mean(0)).mean())
                    # print("eval_lpips", eval_lpips)

                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} \tSSIM: {:.4f}(Best: {:.3f} @epoch {} )'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_scale],
                            eval_ssim,
                            best[0][idx_scale],
                            best[1][idx_scale] + 1
                        )
                    )

                self.ckp.write_log(
                    'Inference time: {:.4f}, Total time: {:.2f}s\n'.format((sum_time / (len(d) - 2))*1000, timer_test.toc()), refresh=True
                )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))


    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
