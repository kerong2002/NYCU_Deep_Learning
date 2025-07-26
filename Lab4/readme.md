'''
Name: DLP Lab4
Topic: Conditional VAE for Video Prediction
Author: CHEN, KE-RONG
Date: 2025/07/22
'''

# Lab 4 - Conditional VAE for Video Prediction

## 1. Introduction

在本實驗中，我們實現了一個基於條件變分自動編碼器（Conditional Variational Autoencoder, CVAE）的模型，用於影片預測。模型的目標是根據前一個影格和一系列的姿態（pose）標籤，生成後續的影片影格。這個任務在許多領域都有應用，例如影片壓縮、影片補幀和人機互動。我們的模型結構主要參考了 [1] 和 [2] 的論文，並結合了 VAE 和 LSTM 的思想，以 RNN 的方式生成影片序列。

## 2. Implementation detail

### 2.1 How do you write your training/testing protocol

#### 訓練流程 (Training Protocol)

訓練流程的核心在 `Trainer.py` 中的 `training_stage` 和 `training_one_step` 函數。

1.  **`training_stage`**:
    *   這個函數負責整個訓練過程的迭代。
    *   在每個 epoch 開始時，會根據當前的 teacher forcing ratio (`self.tfr`) 決定是否使用 teacher forcing。
    *   它會遍歷 `train_dataloader` 提供的所有 batch。
    *   對每個 batch，調用 `training_one_step` 進行單步訓練。
    *   在每個 epoch 結束後，會更新 learning rate (`self.scheduler.step()`)、teacher forcing ratio (`self.teacher_forcing_ratio_update()`) 和 KL annealing 的 beta 值 (`self.kl_annealing.update()`)。
    *   每隔一定的 epoch（`per_save`），會儲存一次模型權重。

2.  **`training_one_step`**:
    *   這個函數處理單個 batch 的訓練。
    *   它會迭代一個影片序列中的所有影格（除了第一個）。
    *   對於序列中的每個時間點 `t`，模型會接收前一個影格（`prev_frame`）和當前的姿態標籤（`current_label`）作為輸入，並試圖重建當前的真實影格（`current_frame`）。
    *   損失函數包含兩部分：
        *   **重建損失 (MSE Loss)**: 計算生成影格和真實影格之間的均方誤差。
        *   **KL 散度 (KL Divergence)**: 作為 VAE 的正則化項，衡量後驗分佈和先驗分佈之間的差異。
    *   `beta` 值（來自 KL annealing）被用來加權 KL 散度項。
    *   根據是否啟用 teacher forcing，下一個時間點的 `prev_frame` 會是當前的真實影格（Teacher Forcing ON）或是模型自己生成的影格（Teacher Forcing OFF）。
    *   最後，計算總損失，並執行反向傳播和優化器步驟。

#### 測試流程 (Testing Protocol)

測試流程在 `Tester.py` 中的 `eval` 和 `val_one_step` 函數中實現。

1.  **`eval`**:
    *   這個函數遍歷 `val_dataloader` 提供的所有測試序列。
    *   對每個序列，調用 `val_one_step` 生成預測的影格序列。
    *   將所有生成的序列整合成一個提交文件 `submission.csv`。

2.  **`val_one_step`**:
    *   這個函數負責生成一個完整的影片序列。
    *   它以測試集提供的第一個影格作為初始影格。
    *   在每個時間點，模型以前一個生成的影格和當前的姿態標籤作為輸入，生成下一個影格。
    *   在測試階段，潛在變量 `z` 是直接從先驗分佈（標準常態分佈）中採樣的，而不是由 `Gaussian_Predictor` 預測。
    *   這個過程會一直持續，直到生成所有 629 個影格。
    *   最後，生成的影格序列會被儲存為 GIF 動畫，以便進行視覺化評估。

### 2.2 How do you implement reparameterization tricks

重參數化技巧在 `modules/modules.py` 的 `Gaussian_Predictor` 類別中的 `reparameterize` 函數實現。這是 VAE 能夠進行端到端訓練的關鍵。

```python
# modules/modules.py

class Gaussian_Predictor(nn.Sequential):
    # ... (其他部分)
    def reparameterize(self, mu, logvar):
        # 重參數化技巧
        # mu: 平均值
        # logvar: 對數變異數
        std = torch.exp(0.5*logvar)  # 計算標準差
        eps = torch.randn_like(std)   # 從標準常態分佈中採樣噪聲
        return mu + eps*std           # 返回採樣的潛在變量
```

**解釋**:

1.  `Gaussian_Predictor` 的前向傳播會輸出來自編碼器的 `mu` (平均值) 和 `logvar` (對數變異數)。
2.  我們不能直接對一個隨機節點（從 `N(mu, var)` 採樣）進行反向傳播。
3.  重參數化技巧將採樣過程分離出來：`z = mu + sigma * epsilon`，其中 `epsilon` 是從一個固定的標準常態分佈 `N(0, I)` 中採樣的噪聲。
4.  這樣，梯度可以通過 `mu` 和 `sigma` 回傳到編碼器，而採樣的隨機性則被移到了噪聲 `epsilon` 上，使其不影響梯度計算。
5.  程式碼中，`std = torch.exp(0.5*logvar)` 是從對數變異數計算標準差 `sigma`。

### 2.3 How do you set your teacher forcing strategy

Teacher forcing 策略在 `Trainer.py` 中實現，主要涉及 `training_stage` 和 `teacher_forcing_ratio_update` 兩個函數。

```python
# Trainer.py

class VAE_Model(nn.Module):
    # ...
    def training_stage(self):
        for i in range(self.args.num_epoch):
            # ...
            # 根據 tfr 決定是否使用 teacher forcing
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                # ...
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                # ...
            # ...
            self.teacher_forcing_ratio_update() # 更新 tfr
            # ...

    def training_one_step(self, img, label, adapt_TeacherForcing):
        # ...
        for i in range(1, self.train_vi_len):
            # ...
            if adapt_TeacherForcing:
                # Teacher forcing: 使用真實的上一影格作為輸入
                prev_frame = current_frame
            else:
                # Curriculum learning: 使用模型生成的上一影格作為輸入
                prev_frame = recon_frame.detach()
        # ...

    def teacher_forcing_ratio_update(self):
        # 更新 teacher forcing ratio
        if self.current_epoch >= self.tfr_sde: # tfr_sde: teacher forcing start decay epoch
            self.tfr = max(0, self.tfr - self.tfr_d_step) # tfr_d_step: decay step
```

**解釋**:

1.  **`tfr` (Teacher Forcing Ratio)**: 這是一個介於 0 和 1 之間的機率值。在每個 epoch 開始時，我們會生成一個隨機數，如果這個隨機數小於 `tfr`，則該 epoch 的所有訓練步驟都會使用 teacher forcing。
2.  **策略**: 在訓練初期，`tfr` 設置為 1.0，意味著模型總是使用真實的上一影格作為輸入。這有助於模型快速學習生成單個影格。
3.  **衰減 (Decay)**: 隨著訓練的進行（當 `current_epoch` 達到 `tfr_sde`），`tfr` 會線性衰減。這迫使模型逐漸學會依賴自己先前生成的影格，從而提高在生成長序列時的穩定性，解決 exposure bias 的問題。

### 2.4 How do you set your kl annealing ratio

KL Annealing 策略在 `Trainer.py` 的 `kl_annealing` 類別中實現。這個策略的目的是在訓練初期讓模型專注於重建任務，然後逐漸引入 KL 散度作為正則化項。

```python
# Trainer.py

class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.type = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch
        # 計算週期性退火的 beta 值列表
        self.betas = self.frange_cycle_linear(args.num_epoch, start=0.0, stop=1.0, n_cycle=self.cycle, ratio=self.ratio)
        self.epoch_per_cycle = args.num_epoch / self.cycle
        self.update()

    def update(self):
        # 更新 KL annealing 的 beta 值
        if self.type == 'Cyclical':
            # 週期性退火
            try:
                self.beta = self.betas[self.current_epoch]
            except:
                self.beta = 1.0
        elif self.type == 'Monotonic':
            # 單調性退火
            self.beta = min(1.0, self.current_epoch / (self.args.num_epoch * self.ratio))
        else: # None
            self.beta = 1.0
        self.current_epoch += 1
    
    def get_beta(self):
        # 獲取當前的 beta 值
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # 計算週期性線性變化的 beta 值
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L
```

**解釋**:

我們實現了三種 KL Annealing 策略，可以通過 `--kl_anneal_type` 參數選擇：

1.  **`Monotonic` (單調性)**: `beta` 值從 0 開始，在訓練過程中線性增加，直到達到 1.0，然後保持不變。這是一種溫和的引入 KL 散度的方法。
2.  **`Cyclical` (週期性)**: `beta` 值在多個週期內從 0 線性增加到 1.0。這種方法允許模型在訓練過程中多次重新學習潛在空間的表示，有助於避免 KL Vanishing 問題並找到更好的局部最優解。
3.  **`None` (無退火)**: `beta` 值始終為 1.0，即從一開始就將 KL 散度完全納入損失計算。

## 3. Analysis & Discussion

### 3.1 Plot Teacher forcing ratio

*(這部分需要等訓練完成後，根據 log 繪製圖表並進行分析)*

**預期分析**:
*   繪製 Teacher Forcing Ratio (TFR) 隨 epoch 變化的曲線圖。它應該在 `tfr_sde` 之前保持為 1.0，然後線性下降。
*   繪製訓練損失曲線圖 (Total Loss, MSE Loss, KL Loss)。
*   比較 TFR 曲線和損失曲線。預期在 TFR 開始下降時，模型的訓練損失可能會出現波動或輕微上升，因為模型開始面對自己生成的不完美輸入，這增加了學習的難度。然而，這有助於提高模型在推論時的泛化能力。

### 3.2 Plot the loss curve while training with different setting.

*(這部分需要分別使用三種 KL annealing 策略進行訓練，並繪製對應的損失曲線)*

**預期分析**:

*   **With KL annealing (Monotonic)**:
    *   預期 KL loss 會隨著 `beta` 的增加而逐漸上升，而 MSE loss 可能會略有增加，因為模型需要在重建和正則化之間找到平衡。
*   **With KL annealing (Cyclical)**:
    *   預期 KL loss 會呈現週期性波動，每次 `beta` 重置為 0 時，KL loss 會下降，然後再次上升。這種週期性的 "放鬆" 可能會讓模型在某些階段更專注於降低 MSE loss，從而可能達到更低的整體損失。
*   **Without KL annealing**:
    *   預期 KL loss 從一開始就在損失函數中佔有較大權重。這可能導致 "posterior collapse" (KL loss 很快趨近於 0)，使得 VAE 退化為一個普通的自編碼器，生成的樣本缺乏多樣性。或者，模型可能難以同時優化 MSE 和 KL loss，導致整體收斂效果不佳。

### 3.3 Plot the PSNR-per-frame diagram in the validation dataset and analyze it

*(這部分需要在驗證集上生成預測序列，計算每個影格的 PSNR，並繪製圖表)*

**預期分析**:
*   繪製 PSNR 隨影格序號（1 到 629）變化的曲線圖。
*   預期 PSNR 會隨著時間的推移而逐漸下降。這是因為預測誤差會不斷累積，越往後的影格，其輸入（即模型前一步的輸出）的不確定性越大，導致生成質量下降。
*   圖表中可能會出現一些突然的 PSNR 下降或上升，這可能對應影片中姿態變化劇烈或靜止的片段。分析這些點可以幫助理解模型的優點和缺點。

### 3.4 Other training strategy analysis (Bonus 10%)

*(這部分可以探討一些其他的訓練策略)*

**可行的分析方向**:
1.  **不同的優化器**: 比較 Adam, AdamW, RMSprop 等不同優化器對收斂速度和最終性能的影響。
2.  **學習率排程 (Learning Rate Scheduler)**: 實驗不同的學習率排程，如 Cosine Annealing, StepLR, ExponentialLR，並分析其效果。
3.  **模型架構調整**: 嘗試增加或減少 Generator 或 Encoder 的層數/通道數，分析其對模型容量和性能的影響。
4.  **噪聲注入 (Noise Injection)**: 在生成器的輸入或中間層加入額外的噪聲，觀察其對生成樣本多樣性的影響。

---

## 訓練與測試指令

### 訓練模型

```bash
python Lab_hw/Lab4/Trainer.py --DR [你的資料集路徑] --save_root Lab_hw/Lab4/checkpoints --fast_train --kl_anneal_type [Cyclical/Monotonic/None]

python Trainer.py --DR ../LAB4_Dataset/LAB4_Dataset --save_root checkpoints --use_wandb --num_epoch 300 --per_save 1 --num_workers 6 --batch_size 8
python Trainer.py --DR ../LAB4_Dataset --save_root checkpoints --num_epoch 300 --per_save 1 --num_workers 6 --batch_size 8 --use_wandb
```
*   `--DR`: 請替換為你的 `LAB4_Dataset` 資料夾的路徑。
*   `--save_root`: 模型權重和日誌的儲存路徑。
*   `--fast_train`: 使用較少的數據進行快速訓練，方便調試。
*   `--kl_anneal_type`: 選擇 KL Annealing 的策略。


## server use
```bash
python Trainer.py --DR ../LAB4_Dataset/LAB4_Dataset --save_root checkpoints --use_wandb --num_epoch 1000 --per_save 1 --num_workers 10 --batch_size 8

python Trainer.py --DR ../LAB4_Dataset/LAB4_Dataset --save_root checkpoints --use_wandb --num_epoch 1000 --per_save 1 --num_workers 6 --batch_size 8


python Trainer.py --DR ../LAB4_Dataset --save_root checkpoints --use_wandb --num_epoch 70 --per_save 1 --num_workers 6 --batch_size 8

python Trainer.py --DR ../LAB4_Dataset --save_root checkpoints --num_epoch 500 --per_save 1 --num_workers 6 --batch_size 8 --kl_anneal_ratio 0.5 --tfr_sde 20 --tfr_d_step 0.05
```
### 測試模型並生成提交文件

```bash
python Tester.py --DR ../LAB4_Dataset --save_root ./results --ckpt_path checkpoints
```

```bash
python LAB4_template\Tester.py --DR ./LAB4_Dataset --save_root ./results --ckpt_path LAB4_template/checkpoints
```
*   `--DR`: 請替換為你的 `LAB4_Dataset` 資料夾的路徑。
*   `--save_root`: 預測結果（GIF 和 submission.csv）的儲存路徑。
*   `--ckpt_path`: 指定訓練好的模型權重檔案（`.ckpt`）的路徑。