'''
Name: DLP Lab4
Topic: Conditional VAE for Video Prediction
Author: CHEN, KE-RONG
Date: 2025/07/26
'''
from datetime import datetime
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio
import wandb
import matplotlib.pyplot as plt
from math import log10



def Generate_PSNR(imgs1, imgs2, data_range=1.0):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2)  # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

# 計算 VAE 的 KL 散度損失 (Kullback-Leibler divergence)
def kl_criterion(mu, logvar, batch_size):
    # D_KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD


class kl_annealing:
    def __init__(self, args, current_epoch=0):
        self.args = args
        self.current_epoch = current_epoch

        # 確保參數正確
        assert args.kl_anneal_type in ["Cyclical", "Monotonic", "None"]

        # 根據不同的退火類型，生成對應的 beta 值序列
        if args.kl_anneal_type == "Cyclical":
            # 週期性退火：beta 值在多個週期內從 start 線性增加到 stop
            self.betas = self.frange_cycle_linear(args.num_epoch, n_cycle=args.kl_anneal_cycle, ratio=args.kl_anneal_ratio)
        elif args.kl_anneal_type == "Monotonic":
            # 單調退火：beta 值在單個週期內（通常是整個訓練過程）從 start 線性增加到 stop
            self.betas = self.frange_cycle_linear(args.num_epoch, n_cycle=1, ratio=args.kl_anneal_ratio)
        elif args.kl_anneal_type == "None":
            # 不使用退火：beta 值在所有 epoch 中都保持為 1
            self.betas = np.ones(args.num_epoch)
        else:
            raise NotImplementedError

    def update(self):
        self.current_epoch += 1 # 更新目前的 epoch 計數。

    def get_beta(self):
        return self.betas[self.current_epoch] # 獲取目前 epoch 對應的 beta

    def frange_cycle_linear(self, n_iter, start=0.1, stop=1.0, n_cycle=1, ratio=1):
        # 生成一個週期性線性變化的 beta 值序列。

        betas = np.ones(n_iter) * stop # 初始化所有 beta 值為 stop
        period = n_iter // n_cycle # 計算每個週期的長度

        # 為每個週期生成線性增加的 beta 值
        for i in range(n_cycle):
            start_ = period * i
            end_ = start_ + int(period * ratio)
            betas[start_:end_] = np.linspace(start, stop, int(period * ratio))
        return betas


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)

        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim = optim.AdamW(self.parameters(), lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=args.num_epoch)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size

    def forward(self, img, label):
        pass

    def training_stage(self):
        # 跳過第一張
        for i in range(1, self.args.num_epoch + 1):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            total_loss = 0

            for img, label in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                total_loss += loss.detach().cpu()

                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar(
                        "train [TeacherForcing: ON, {:.1f}], beta: {:.3f}".format(self.tfr, beta),
                        pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar("train [TeacherForcing: OFF, {:.1f}], beta: {:.3f}".format(self.tfr, beta),
                        pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

            if i % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch_{i}.ckpt"))

            if self.args.wandb:
                wandb.log({
                    "train/loss": total_loss / len(train_loader),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/tfr": self.tfr,
                    "train/beta": beta,
                }, step=self.current_epoch)

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        total_loss = 0
        total_psnr = 0
        for img, label in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)
            self.tqdm_bar(
                "val", pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0]
            )
            
            total_loss += loss.detach().cpu()
            total_psnr += psnr.detach().cpu()

        if self.args.wandb:
            wandb.log({
                "val/loss": total_loss / len(val_loader),
                "val/lr": self.scheduler.get_last_lr()[0],
                "val/tfr": self.tfr,
                "val/psnr": total_psnr / len(val_loader),
            }, step=self.current_epoch)

    def training_one_step(self, image_sequence, label_sequence, use_teacher_forcing):
        total_training_loss = 0
        kl_beta_weight = self.kl_annealing.get_beta()

        batch_size, timesteps = image_sequence.shape[:2]
        predicted_frame = image_sequence[:, 0, ...]

        # 從第二個影格開始
        for time_step in range(1, timesteps):
            # 根據 use_teacher_forcing 決定解碼器的輸入是真實前一影格還是模型生成的影格
            if use_teacher_forcing:
                previous_frame_input = image_sequence[:, time_step - 1, ...]  # 使用真實的前一影格 (Teacher Forcing)
            else:
                previous_frame_input = predicted_frame  # 使用模型生成的影格

            current_ground_truth_frame = image_sequence[:, time_step, ...]
            current_ground_truth_label = label_sequence[:, time_step, ...]
            encoded_current_frame = self.frame_transformation(current_ground_truth_frame)
            encoded_current_label = self.label_transformation(current_ground_truth_label)

            # 使用高斯預測器從目前影格和標籤預測潛在變數 z 的分佈
            latent_z, posterior_mu, posterior_logvar = self.Gaussian_Predictor(encoded_current_frame, encoded_current_label)

            # 將前一影格編碼為特徵，並使用 .detach() 避免梯度回傳
            encoded_previous_frame = self.frame_transformation(previous_frame_input).detach()

            # 解碼器融合前一影格特徵、目前標籤特徵和潛在變數 z
            decoded_features = self.Decoder_Fusion(encoded_previous_frame, encoded_current_label, latent_z)

            predicted_frame = self.Generator(decoded_features)
            predicted_frame = nn.functional.sigmoid(predicted_frame)

            # 計算重建損失 (MSE) 和 KL 散度損失
            reconstruction_loss = self.mse_criterion(predicted_frame, current_ground_truth_frame)
            kl_divergence_loss = kl_criterion(posterior_mu, posterior_logvar, batch_size)

            # 結合兩種損失，並使用 beta 加權 KL 損失
            step_loss = reconstruction_loss + kl_beta_weight * kl_divergence_loss
            total_training_loss += step_loss

        # 正規化總損失
        total_training_loss /= batch_size

        # 梯度清零、反向傳播和更新模型參數
        self.optim.zero_grad()
        total_training_loss.backward()
        self.optimizer_step()

        return total_training_loss

    @torch.no_grad()
    def val_one_step(self, image_sequence, label_sequence):
        total_validation_loss = 0
        total_validation_psnr = 0
        batch_size, timesteps = image_sequence.shape[:2]
        # 將 predicted_frame 初始化為影片的第一個影格
        predicted_frame = image_sequence[:, 0, ...]

        kl_beta_weight = self.kl_annealing.get_beta()

        for time_step in range(1, timesteps):
            # 在驗證階段，總是使用模型自己生成的前一影格
            previous_frame_input = predicted_frame

            current_ground_truth_frame = image_sequence[:, time_step, ...]
            current_ground_truth_label = label_sequence[:, time_step, ...]
            encoded_current_frame = self.frame_transformation(current_ground_truth_frame)
            encoded_current_label = self.label_transformation(current_ground_truth_label)

            # 預測潛在變數 z
            latent_z, posterior_mu, posterior_logvar = self.Gaussian_Predictor(encoded_current_frame, encoded_current_label)

            # 編碼前一影格
            encoded_previous_frame = self.frame_transformation(previous_frame_input)
            # 解碼器融合特徵以生成新影格
            decoded_features = self.Decoder_Fusion(encoded_previous_frame, encoded_current_label, latent_z)

            predicted_frame = self.Generator(decoded_features)
            predicted_frame = nn.functional.sigmoid(predicted_frame)

            # 計算重建損失和 KL 損失
            reconstruction_loss = self.mse_criterion(predicted_frame, current_ground_truth_frame)
            kl_divergence_loss = kl_criterion(posterior_mu, posterior_logvar, batch_size)

            # 結合損失
            step_loss = reconstruction_loss + kl_beta_weight * kl_divergence_loss
            total_validation_loss += step_loss
            # 計算並累加 PSNR
            total_validation_psnr += Generate_PSNR(predicted_frame, current_ground_truth_frame)

        # 正規化總損失和總 PSNR
        total_validation_loss /= batch_size
        total_validation_psnr /= batch_size * timesteps

        return total_validation_loss, total_validation_psnr

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(img_name, format="GIF", append_images=new_list, save_all=True, duration=40, loop=0)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode="train", video_len=self.train_vi_len, partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False

        train_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.args.num_workers, drop_last=True, shuffle=False)
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode="val", video_len=self.val_vi_len, partial=1.0)
        val_loader = DataLoader(dataset, batch_size=1, num_workers=self.args.num_workers, drop_last=True, shuffle=False)
        return val_loader

    def teacher_forcing_ratio_update(self):
        # 檢查目前的 epoch 是否已達到開始衰減 Teacher Forcing 比例的 epoch
        if self.current_epoch >= self.args.tfr_sde:
            # 將 Teacher Forcing 比例減去指定的衰減步長
            self.tfr -= self.args.tfr_d_step
            self.tfr = max(self.tfr, 0) # 確保比例不會小於 0

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "optimizer": self.state_dict(),
                "lr": self.scheduler.get_last_lr()[0],
                "tfr": self.tfr,
                "last_epoch": self.current_epoch,
            }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint["state_dict"], strict=True)
            self.args.lr = checkpoint["lr"]
            self.tfr = checkpoint["tfr"]

            self.optim = optim.AdamW(self.parameters(), lr=self.args.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[20, 40], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint["last_epoch"])
            self.current_epoch = checkpoint["last_epoch"]

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()


def main(args):
    if args.wandb:
        wandb.init(project="Lab4_VAE", name=f"VAE_tfr_{args.tfr}_kl-type_{args.kl_anneal_type}_kl-cycle_{args.kl_anneal_cycle}_kl-ratio_{args.kl_anneal_ratio}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", config=vars(args))

    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument('--device', type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--optim", type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument("--DR", type=str, required=True, help="Your Dataset Path")
    parser.add_argument("--save_root", type=str, required=True, help="The path to save your data")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=70, help="number of total epoch")
    parser.add_argument("--per_save", type=int, default=10, help="Save checkpoint every seted epoch")
    parser.add_argument("--partial", type=float, default=1.0, help="Part of the training dataset to be trained",)
    parser.add_argument('--train_vi_len', type=int, default=16, help="Training video length")
    parser.add_argument('--val_vi_len', type=int, default=630, help="valdation video length")
    parser.add_argument('--frame_H', type=int, default=32, help="Height input image to be resize")
    parser.add_argument('--frame_W', type=int, default=64, help="Width input image to be resize")

    # Module parameters setting
    parser.add_argument('--F_dim', type=int, default=128, help="Dimension of feature human frame")
    parser.add_argument('--L_dim', type=int, default=32, help="Dimension of feature label frame")
    parser.add_argument('--N_dim', type=int, default=12, help="Dimension of the Noise")
    parser.add_argument('--D_out_dim', type=int, default=192, help="Dimension of the output in Decoder_Fusion")

    # Teacher Forcing strategy
    parser.add_argument('--tfr', type=float, default=1.0, help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde', type=int, default=10, help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step', type=float, default=0.1, help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path', type=str, default=None, help="The path of your checkpoints")

    # Training Strategy
    parser.add_argument('--fast_train', action='store_true')
    parser.add_argument('--fast_partial', type=float, default=0.4, help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch', type=int, default=5, help="Number of epoch to use fast train mode")

    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type', type=str, default='Cyclical', help="")
    parser.add_argument('--kl_anneal_cycle', type=int, default=10, help="")
    parser.add_argument('--kl_anneal_ratio', type=float, default=1, help="")
    

    args = parser.parse_args()

    main(args)
