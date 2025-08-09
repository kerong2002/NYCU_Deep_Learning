import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid, save_image
import wandb

# [修改] 從 diffusers 導入 DDIMScheduler 和學習率排程器
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup


from model import DiffusionModel
from dataloader import get_dataloader
from evaluator import evaluation_model


def train(args, device):
    """主要的訓練函式，整合了學習率排程和 CFG 訓練。"""
    # --- 1. 初始化設定 ---
    print("開始訓練模式 (DDIM, CFG + LR Scheduler)...")
    run_output_dir = os.path.join(args.output_dir, args.run_name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    sample_dir = os.path.join(run_output_dir, "samples")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    wandb.init(project=args.project_name, name=args.run_name, config=args)

    # --- 2. 載入數據 ---
    train_loader = get_dataloader(
        data_dir=args.data_dir, image_dir=args.image_dir, mode='train',
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # --- 3. 初始化模型、排程器、優化器 ---
    model = DiffusionModel(
        image_size=args.image_size,
        num_classes=args.num_classes
    ).to(device)

    # [修改] 使用 DDIMScheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        clip_sample=False,  # DDIM 通常建議不裁切樣本
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_loader) * args.epochs),
    )

    criterion = nn.MSELoss()
    evaluator = evaluation_model()

    # --- 4. 訓練迴圈 ---
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in train_loader:
            clean_images, labels = batch
            clean_images = clean_images.to(device)
            labels = labels.to(device)

            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],),
                                      device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            optimizer.zero_grad()

            if torch.rand(1).item() < args.cfg_dropout_prob:
                labels = torch.zeros_like(labels)

            noise_pred = model(noisy_images, timesteps, labels)
            loss = criterion(noise_pred, noise)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])
            wandb.log({"step_loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step)
            global_step += 1

        progress_bar.close()

        # --- 5. 定期採樣與評估 (使用 DDIM Scheduler) ---
        if (epoch + 1) % args.save_interval == 0:
            model.eval()
            with torch.no_grad():
                preview_images = torch.randn((args.preview_batch_size, 3, args.image_size, args.image_size),
                                             device=device)
                preview_labels = next(iter(train_loader))[1][:args.preview_batch_size].to(device)

                # [修改] 設定推論步數，此處預覽與訓練步數一致
                noise_scheduler.set_timesteps(args.num_train_timesteps)

                for t in tqdm(noise_scheduler.timesteps, desc="Generating preview samples"):
                    t_for_model = t.to(device).expand(preview_images.shape[0])
                    noise_pred = model(preview_images, t_for_model, preview_labels)
                    # [修改] 使用 DDIM 的 step 函數
                    preview_images = noise_scheduler.step(noise_pred, t, preview_images).prev_sample

                accuracy = evaluator.eval(preview_images, preview_labels)
                gen_images_display = (preview_images.clamp(-1, 1) + 1) / 2
                save_path = os.path.join(sample_dir, f"epoch_{epoch + 1}.png")
                save_image(make_grid(gen_images_display), save_path)

                wandb.log({
                    "epoch": epoch + 1, "epoch_loss": loss.item(),
                    "samples": wandb.Image(save_path), "sample_accuracy": accuracy
                }, step=global_step)

            ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Epoch {epoch + 1} 完成，權重已儲存至 {ckpt_path}")
            model.train()

    wandb.finish()
    print("訓練完成。")


def main():
    parser = argparse.ArgumentParser(description="訓練一個基於 DDIM 的條件式擴散模型")

    # 路徑與專案設定
    parser.add_argument("--data_dir", type=str, default="./src", help="存放 .json 標籤檔的目錄")
    parser.add_argument("--image_dir", type=str, default="./iclevr", help="存放 iclevr 圖像的目錄")
    parser.add_argument("--output_dir", type=str, default="output", help="輸出根目錄")
    parser.add_argument("--run_name", type=str, default="ddim-cfg-v1", help="本次執行的名稱，用於 wandb 和輸出目錄")
    parser.add_argument("--project_name", type=str, default="Lab6-DDIM-Stronger", help="wandb 專案名稱")

    # 模型與擴散參數
    parser.add_argument("--image_size", type=int, default=64, help="圖像尺寸")
    parser.add_argument("--num_classes", type=int, default=24, help="標籤類別總數")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="訓練時間步數")
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2", help="噪聲排程策略")

    # 訓練超參數
    parser.add_argument("--epochs", type=int, default=200, help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="學習率峰值")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="學習率熱身步數")
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.1, help="CFG 訓練時丟棄條件的機率")

    # 其他設定
    parser.add_argument("--save_interval", type=int, default=10, help="儲存權重和預覽的 epoch 間隔")
    parser.add_argument("--preview_batch_size", type=int, default=16, help="預覽樣本的批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 的工作進程數")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()