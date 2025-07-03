# DLP Lab1: Back-propagation

This project implements a simple neural network with two hidden layers from scratch using only NumPy. The goal is to understand and implement the forward pass and back-propagation algorithm. The project is highly modular and configurable via command-line arguments.

## 專案結構

```
Lab1/
├── My_package/
│   ├── __init__.py
│   ├── layers.py       # 定義神經網路層 (Linear, Sigmoid, ReLU)
│   ├── loss.py         # 定義損失函數 (MSE)
│   ├── model.py        # 定義 Sequential 模型
│   ├── optimizer.py    # 定義優化器 (SGD with Momentum)
│   └── trainer.py      # 定義訓練器
├── dataset.py          # 產生 linear 和 XOR 資料集
├── main.py             # 主執行檔，負責解析參數、建構與訓練模型
├── show_result.py      # 視覺化訓練結果與損失曲線
└── README.md           # 本說明文件
```

## 如何執行

您可以透過執行 `main.py` 來啟動訓練。所有實驗參數都可以透過命令列參數進行設定。

基礎執行指令：
```bash
python Lab1/main.py
```

### 命令列參數說明

您可以使用 `-h` 或 `--help` 來查看所有可用的參數：
```bash
python Lab1/main.py --help
```

以下是主要的參數列表：

| 參數            | 預設值             | 選項                    | 說明                           | 確認 |
|-----------------|--------------------|-----------------------|--------------------------------|----|
| `--dataset`     | `xor`              | `linear`, `xor`       | 選擇要使用的資料集。           | ✅️  |
| `--epochs`      | `20000`            | 任意正整數                 | 設定訓練的週期總數。           | ✅️  |
| `--lr`          | `0.1`              | 任意正浮點數                | 設定學習率。                   |✅️|
| `--hidden-dims` | `10 10`            | 多個正整數（空白分隔）           | 設定每個隱藏層的神經元數量。   ||
| `--activation`  | `sigmoid`          | `sigmoid`, `relu`     | 選擇隱藏層的活化函數。         ||
| `--optimizer`   | `sgd`              | `sgd`, `gd`, `adam`, `adagrad` | 選擇優化器（目前僅支援 SGD）。 |✅️|
| `--loss`        | `bce`              | `bce`, `mse`, `cross` | 選擇損失函數（建議分類問題用 bce）。|✅️|
| `--momentum`    | `0.0`              | 0.0~1.0               | 設定 SGD 優化器的動量值。      |✅️|
| `--seed`        | `1`                | 任意正整數                 | 設定隨機數種子，確保實驗可重複。|✅️|
| `--log-interval`| `1000`             | 任意正整數                 | 每隔多少週期輸出一次訓練日誌。  |✅️|
### 執行範例

1.  **使用預設參數執行 (XOR 資料集, Sigmoid, 20000 epochs):**
    ```bash
    python Lab1/main.py
    ```

2.  **訓練線性資料集，並使用 ReLU 活化函數：**
    ```bash
    python Lab1/main.py --dataset linear --activation relu
    ```

3.  **降低學習率，並增加訓練週期：**
    ```bash
    python Lab1/main.py --lr 0.05 --epochs 30000
    ```

4.  **嘗試不同的網路結構（例如：三層隱藏層）：**
    ```bash
    python Lab1/main.py --hidden-dims 10 20 10
    ```

5.  **使用帶有動量的 SGD 優化器：**
    ```bash
    python Lab1/main.py --momentum 0.9

6.  **進階組合範例 (一次指定多個參數):**
    ```bash
    python Lab_hw/Lab1/main.py --dataset linear --activation relu --lr 0.01 --momentum 0.9 --epochs 15000 --hidden-dims 20 20 --loss bce
    python Lab_hw/Lab1/main.py --dataset linear --activation relu --lr 0.01 --momentum 0.9 --epochs 1000 --hidden-dims 20 20 --loss mse --optimizer adam
    ```