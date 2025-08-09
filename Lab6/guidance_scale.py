import os
import subprocess
import json
import argparse
import wandb
from datetime import datetime
import glob


def run_test_with_guidance_scale(ckpt_path, guidance_scale, run_name_base, data_dir="./code-v5"):
    """
    執行測試腳本，讀取準確度結果，並找出生成的範例圖片路徑。
    """
    run_name = f"{run_name_base}_guidance_{guidance_scale}"
    results_dir = os.path.join("result", run_name)
    images_dir = os.path.join(results_dir, "images")  # 假設圖片會存在這裡

    # 構建命令
    cmd = [
        "python", os.path.join(data_dir, "test.py"),
        "--ckpt_path", ckpt_path,
        "--guidance_scale", str(guidance_scale),
        "--run_name", run_name,
        "--data_dir", data_dir
    ]

    print(f"\n--- 執行 Guidance Scale {guidance_scale} 測試 ---")
    print(f"命令: {' '.join(cmd)}")

    try:
        # 執行命令
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Guidance Scale {guidance_scale} 測試完成")

        # 從 JSON 檔案讀取準確度結果
        results_path = os.path.join(results_dir, "accuracies.json")
        test_accuracy = None
        new_test_accuracy = None
        success = False

        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                accuracies = json.load(f)
            test_accuracy = accuracies.get("test")
            new_test_accuracy = accuracies.get("new_test")

            if test_accuracy is not None and new_test_accuracy is not None:
                print(f"  成功從 {results_path} 讀取結果。")
                print(f"  Test 準確率: {test_accuracy:.4f}")
                print(f"  New Test 準確率: {new_test_accuracy:.4f}")
                success = True
            else:
                print(f"  ❌ 錯誤: 結果檔案 {results_path} 內容不完整。")
        else:
            print(f"  ❌ 錯誤: 找不到結果檔案 {results_path}")

        # **【新增】** 尋找生成的範例圖片
        sample_images = []
        if os.path.exists(images_dir):
            # 尋找所有 .png 或 .jpg 檔案，最多找5張
            sample_images = glob.glob(os.path.join(images_dir, "*.png"))
            sample_images.extend(glob.glob(os.path.join(images_dir, "*.jpg")))
            sample_images = sample_images[:5]
            if sample_images:
                print(f"  🎨 找到 {len(sample_images)} 張範例圖片。")

        return {
            "guidance_scale": guidance_scale,
            "test_accuracy": test_accuracy,
            "new_test_accuracy": new_test_accuracy,
            "run_name": run_name,
            "success": success,
            "sample_images": sample_images,  # 回傳圖片路徑
            "output": "Results read from JSON file."
        }

    except subprocess.CalledProcessError as e:
        print(f"❌ Guidance Scale {guidance_scale} 測試失敗")
        print(f"錯誤: {e.stderr}")
        return {
            "guidance_scale": guidance_scale,
            "test_accuracy": None,
            "new_test_accuracy": None,
            "run_name": run_name,
            "success": False,
            "sample_images": [],
            "error": e.stderr
        }


def main():
    parser = argparse.ArgumentParser(description="測試不同 Guidance Scale 對 DDPM 生成品質的影響")
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型檢查點路徑")
    parser.add_argument("--data_dir", type=str, default="./src", help="測試腳本和數據所在目錄")
    parser.add_argument("--project_name", type=str, default="DDPM-Guidance-Scale-Study", help="wandb 專案名稱")
    parser.add_argument("--run_base_name", type=str, default="guidance_experiment", help="實驗基礎名稱")

    # **【修改】** 產生新的 guidance scale 預設列表 (1.0, 1.5, ..., 10.0)
    new_default_scales = [round(i * 0.5, 1) for i in range(2, 21)]
    parser.add_argument("--guidance_scales", type=float, nargs='+',
                        default=new_default_scales,
                        help="要測試的 guidance scale 值列表 (預設為 1.0 到 10.0，間隔 0.5)")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path) or not os.path.exists(os.path.join(args.data_dir, "test.py")):
        print("❌ 找不到模型或測試腳本，程式中止。")
        return

    print("🚀 開始 Guidance Scale 實驗")
    print(f"模型檢查點: {args.ckpt_path}")
    print(f"測試的 Guidance Scale 值: {args.guidance_scales}")

    # 初始化 wandb
    experiment_name = f"{args.run_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.project_name,
        name=experiment_name,
        config=vars(args)
    )

    # **【新增】** 定義 wandb 的 x 軸，讓圖表直接使用 guidance_scale
    wandb.define_metric("test_accuracy", step_metric="guidance_scale")
    wandb.define_metric("new_test_accuracy", step_metric="guidance_scale")
    wandb.define_metric("average_accuracy", step_metric="guidance_scale")

    all_results = []

    # 執行不同 guidance scale 的測試
    for guidance_scale in args.guidance_scales:
        result = run_test_with_guidance_scale(
            ckpt_path=args.ckpt_path,
            guidance_scale=guidance_scale,
            run_name_base=args.run_base_name,
            data_dir=args.data_dir
        )
        all_results.append(result)

        if result["success"]:
            log_data = {
                "guidance_scale": result["guidance_scale"],
                "test_accuracy": result["test_accuracy"],
                "new_test_accuracy": result["new_test_accuracy"],
                "average_accuracy": (result["test_accuracy"] + result["new_test_accuracy"]) / 2
            }

            if result["sample_images"]:
                log_data["examples"] = [wandb.Image(img_path, caption=f"Guidance: {guidance_scale}") for img_path in
                                        result["sample_images"]]

            wandb.log(log_data)
        else:
            wandb.log({"guidance_scale": result["guidance_scale"], "failed": True})

    print("\n" + "=" * 60)
    print("📊 實驗結果總結")
    print("=" * 60)

    successful_results = [r for r in all_results if r["success"]]
    if successful_results:
        print(f"{'Guidance Scale':<15} {'Test Acc':<12} {'New Test Acc':<15} {'Average':<12}")
        print("-" * 60)
        for result in successful_results:
            test_acc = result.get("test_accuracy")
            new_test_acc = result.get("new_test_accuracy")
            test_acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
            new_test_acc_str = f"{new_test_acc:.4f}" if new_test_acc is not None else "N/A"
            if test_acc is not None and new_test_acc is not None:
                avg_acc_str = f"{(test_acc + new_test_acc) / 2:.4f}"
            else:
                avg_acc_str = "N/A"
            print(f"{result['guidance_scale']:<15.1f} {test_acc_str:<12} {new_test_acc_str:<15} {avg_acc_str:<12}")

    print("\n" + "=" * 60)
    print("📋 正在上傳總結報告至 W&B...")
    print("=" * 60)

    columns = ["Guidance Scale", "Test Acc", "New Test Acc", "Average Acc", "Example Image"]
    summary_table = wandb.Table(columns=columns)

    for result in successful_results:
        test_acc = result.get("test_accuracy")
        new_test_acc = result.get("new_test_accuracy")
        if test_acc is not None and new_test_acc is not None:
            avg_acc = (test_acc + new_test_acc) / 2
            example_image = wandb.Image(result["sample_images"][0]) if result["sample_images"] else None
            summary_table.add_data(
                result["guidance_scale"], f"{test_acc:.4f}", f"{new_test_acc:.4f}", f"{avg_acc:.4f}", example_image
            )

    wandb.log({"guidance_scale_summary": summary_table})
    print("✅ 總結報告已上傳！")

    valid_results = [r for r in successful_results if
                     r.get("test_accuracy") is not None and r.get("new_test_accuracy") is not None]
    if valid_results:
        best_result = max(valid_results, key=lambda x: (x["test_accuracy"] + x["new_test_accuracy"]) / 2)
        best_avg_acc = (best_result['test_accuracy'] + best_result['new_test_accuracy']) / 2
        print(f"\n🏆 最佳 Guidance Scale: {best_result['guidance_scale']}")
        print(f"   平均準確率: {best_avg_acc:.4f}")
        wandb.summary["best_guidance_scale"] = best_result['guidance_scale']
        wandb.summary["best_average_accuracy"] = best_avg_acc

    results_file = f"guidance_scale_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📁 詳細結果已保存至: {results_file}")

    wandb.finish()


if __name__ == "__main__":
    main()