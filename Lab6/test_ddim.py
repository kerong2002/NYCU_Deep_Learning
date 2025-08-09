import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
# [修改] 從 diffusers 導入 DDIMScheduler
from diffusers import DDIMScheduler


from model import DiffusionModel
from evaluator import evaluation_model


def generate_denoising_visualization(args, model, noise_scheduler, objects_map, output_dir, device):
    """
    為固定的提示詞生成並儲存去噪過程視覺化圖像。
    圖片會儲存到指定的 output_dir。
    """
    print("\n--- 開始生成去噪過程視覺化圖像 (DDIM) ---")

    prompt = ["red sphere", "cyan cylinder", "cyan cube"]
    print(f"使用範例提示詞: {prompt}")

    labels = torch.zeros(1, args.num_classes, device=device)
    for label_text in prompt:
        if label_text in objects_map:
            labels[0, objects_map[label_text]] = 1

    # 使用固定的初始噪聲以確保可重現性
    generator = torch.manual_seed(42)
    image = torch.randn((1, 3, args.image_size, args.image_size), generator=generator, device=device)

    denoise_process_images = [image.clone()]

    num_inference_steps = len(noise_scheduler.timesteps)
    save_indices = np.linspace(0, num_inference_steps - 1, num=min(10, num_inference_steps), dtype=int)

    for i, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Generating denoising process")):
        with torch.no_grad():
            latent_model_input = torch.cat([image] * 2)
            uncond_labels = torch.zeros_like(labels)
            combined_labels = torch.cat([labels, uncond_labels], dim=0)

            t_for_model = t.to(device).expand(latent_model_input.shape[0])
            noise_pred = model(latent_model_input, t_for_model, combined_labels)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            guided_noise = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # [修改] 使用 DDIM 的 step 函數，並傳入 eta
        image = noise_scheduler.step(guided_noise, t, image, eta=args.eta).prev_sample

        if i in save_indices:
            denoise_process_images.append(image.clone())

    denoise_process_images.append(image.clone())
    denoise_process_images = [(img.clamp(-1, 1) + 1) / 2 for img in denoise_process_images]

    grid = make_grid(torch.cat(denoise_process_images), nrow=len(denoise_process_images))
    save_path = os.path.join(output_dir, "denoising_process_ddim.png")
    save_image(grid, save_path)
    print(f"已成功儲存去噪過程圖像至: {save_path}")


def test(args, device):
    """主要的測試函式，使用 DDIM 和 CFG 進行高品質圖像生成。"""
    print("開始測試模式 (DDIM + Classifier-Free Guidance)...")
    run_dir = os.path.join("result", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    model = DiffusionModel(
        image_size=args.image_size,
        num_classes=args.num_classes
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()
    print(f"成功從 {args.ckpt_path} 載入模型權重。")

    # [修改] 使用 DDIMScheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        clip_sample=False  # DDIM 通常建議不裁切樣本，以避免偽影
    )
    # [修改] 為 DDIM 設定推論步數
    noise_scheduler.set_timesteps(args.num_inference_steps)
    print(f"使用 DDIM 排程器，推論步數: {args.num_inference_steps}, Eta: {args.eta}")

    evaluator = evaluation_model()
    with open(os.path.join(args.data_dir, 'objects.json'), 'r') as f:
        objects_map = json.load(f)

    accuracies = {}
    for mode in ['test', 'new_test']:
        print(f"\n--- 開始處理 {mode}.json ---")
        json_path = os.path.join(args.data_dir, f"{mode}.json")
        with open(json_path, 'r') as f:
            test_prompts = json.load(f)

        all_generated_images = []
        all_ground_truth_labels = []

        # [修改] 為確保每次測試結果一致，使用固定的隨機種子
        generator = torch.manual_seed(0)

        for conditional_prompt in tqdm(test_prompts, desc=f"生成圖像 ({mode})"):
            labels = torch.zeros(1, args.num_classes, device=device)
            for label_text in conditional_prompt:
                labels[0, objects_map[label_text]] = 1

            # 使用 generator 產生可重現的初始噪聲
            image = torch.randn((1, 3, args.image_size, args.image_size), generator=generator, device=device)

            for t in noise_scheduler.timesteps:
                with torch.no_grad():
                    latent_model_input = torch.cat([image] * 2)
                    uncond_labels = torch.zeros_like(labels)
                    combined_labels = torch.cat([labels, uncond_labels], dim=0)
                    t_for_model = t.to(device).expand(latent_model_input.shape[0])
                    noise_pred = model(latent_model_input, t_for_model, combined_labels)
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    guided_noise = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # [修改] 使用 DDIM 的 step 函數，並傳入 eta
                image = noise_scheduler.step(guided_noise, t, image, eta=args.eta).prev_sample

            all_generated_images.append(image)
            all_ground_truth_labels.append(labels)

        final_images_batch = torch.cat(all_generated_images, dim=0)
        final_labels_batch = torch.cat(all_ground_truth_labels, dim=0)

        accuracy = evaluator.eval(final_images_batch, final_labels_batch)
        accuracies[mode] = accuracy
        print(f"{mode}.json 的準確率: {accuracy:.4f}")

        image_output_dir = os.path.join(run_dir, "images", mode)
        os.makedirs(image_output_dir, exist_ok=True)
        for i, img_tensor in enumerate(final_images_batch):
            save_image((img_tensor.clamp(-1, 1) + 1) / 2, os.path.join(image_output_dir, f"{i}.png"))

        grid = make_grid((final_images_batch[:32].clamp(-1, 1) + 1) / 2, nrow=8)
        save_image(grid, os.path.join(run_dir, f"{mode}_grid_ddim.png"))
        print(f"已儲存圖像。獨立圖片位於: {image_output_dir}, 網格圖位於: {run_dir}")

    results_path = os.path.join(run_dir, "accuracies_ddim.json")
    with open(results_path, 'w') as f:
        json.dump(accuracies, f, indent=2)
    print(f"\n準確率結果已儲存至: {results_path}")

    generate_denoising_visualization(args, model, noise_scheduler, objects_map, output_dir=run_dir, device=device)


def main():
    parser = argparse.ArgumentParser(description="測試一個基於 DDIM 的條件式擴散模型")

    parser.add_argument('--ckpt_path', type=str, required=True, help='要測試的權重路徑 (.pth)')
    parser.add_argument('--data_dir', type=str, default='./src', help='存放 .json 檔案的目錄')
    parser.add_argument('--run_name', type=str, default="test-run-ddim", help='本次測試的名稱，用於創建輸出子目錄')

    parser.add_argument("--image_size", type=int, default=64, help="圖像尺寸")
    parser.add_argument("--num_classes", type=int, default=24, help="標籤類別總數")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="模型訓練時所用的時間步數")
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2", help="噪聲排程策略")

    # [修改] 減少預設推論步數，因為 DDIM 更高效
    parser.add_argument("--num_inference_steps", type=int, default=50, help="DDIM 推斷時的去噪步數 (可大幅減少)")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='CFG 引導的強度')
    # [新增] DDIM 特有的 eta 參數
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta 參數。0.0 為確定性 DDIM，1.0 為 DDPM。')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test(args, device)


if __name__ == '__main__':
    main()