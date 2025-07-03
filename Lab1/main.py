'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''
import argparse
import numpy as np

from dataset import generate_linear, generate_XOR_easy, plot_data
from My_package.model import Sequential
from My_package.layers import Linear, Sigmoid, ReLU
from My_package.loss import MSE, BCELoss, CrossEntropyLoss
from My_package.optimizer import SGD
from My_package.trainer import Trainer
from show_result import show_result, plot_loss_curve

def main():
    """
    主函式，負責解析命令列參數、建構模型並執行訓練。
    """
    # --- 命令列參數解析 ---
    parser = argparse.ArgumentParser(description='Lab1: Back-propagation')
    parser.add_argument('--dataset', type=str, default='xor', choices=['linear', 'xor'],
                        help='dataset to use (default: xor)')
    parser.add_argument('--epochs', type=int, default=20000, metavar='N',
                        help='number of epochs to train (default: 20000)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[10, 10],
                        help='dimensions of hidden layers (default: 10 10)')
    parser.add_argument('--activation', type=str, default='sigmoid', choices=['sigmoid', 'relu'],
                        help='activation function to use (default: sigmoid)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd'],
                        help='optimizer to use (default: sgd)')
    parser.add_argument('--loss', type=str, default='bce', choices=['mse', 'bce', 'cross'],
                        help='loss function to use (default: bce)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many epochs to wait before logging training status')
    
    args = parser.parse_args()
    np.random.seed(args.seed)

    # --- 資料準備 ---
    print(f"使用資料集: {args.dataset.upper()}")
    if args.dataset == 'linear':
        X, y = generate_linear(n=100)
    else:
        X, y = generate_XOR_easy()
    plot_data(X, y, title=f"Original {args.dataset.upper()} Data")

    # --- 模型建構 ---
    print("\n開始建構循序模型...")
    activation_layer = {'sigmoid': Sigmoid, 'relu': ReLU}[args.activation]
    
    layers = []
    input_dim = 2
    for hidden_dim in args.hidden_dims:
        layers.append(Linear(input_dim, hidden_dim))
        layers.append(activation_layer())
        input_dim = hidden_dim
    layers.append(Linear(input_dim, 1))
    layers.append(Sigmoid()) # 輸出層通常使用 Sigmoid 進行二元分類

    model = Sequential(layers)
    print("模型結構:")
    for layer in model.layers:
        print(f"- {layer.__class__.__name__}")


    # --- 訓練設定 ---
    loss_fn = {'mse': MSE, 'bce': BCELoss, 'cross': CrossEntropyLoss}[args.loss]()

    if args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.lr, momentum=args.momentum)
    else:
        # 未來可以擴充其他優化器
        raise ValueError(f"不支援的優化器: {args.optimizer}")

    trainer = Trainer(model, loss_fn, optimizer)

    # --- 模型訓練 ---
    print(f"\n開始訓練... (Epochs: {args.epochs}, LR: {args.lr}, Activation: {args.activation}, Momentum: {args.momentum})")
    loss_history = trainer.train(X, y, args.epochs, args.log_interval)

    # --- 結果顯示 ---
    print("\n訓練完成！顯示結果...")
    plot_loss_curve(loss_history)
    show_result(model, X, y)

if __name__ == '__main__':
    main()