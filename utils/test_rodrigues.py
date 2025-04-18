import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def _hat(v: torch.Tensor) -> torch.Tensor:
    """Skew‑symmetric matrix (hat operator) for a 3‑vector."""
    h = torch.zeros((3, 3), dtype=v.dtype, device=v.device)
    h[0, 1], h[0, 2] = -v[2],  v[1]
    h[1, 0], h[1, 2] =  v[2], -v[0]
    h[2, 0], h[2, 1] = -v[1],  v[0]
    return h

def rodrigues_torch_impl1(rvec):
    """First implementation of Rodrigues formula with small-angle safeguard"""
    # Convert numpy array to torch tensor if needed
    if isinstance(rvec, np.ndarray):
        rvec = torch.from_numpy(rvec).float()
    
    theta = torch.linalg.norm(rvec)
    if theta < 1.0e-6:  # 1st‑order Taylor
        R = torch.eye(3, device=rvec.device) + _hat(rvec)
    else:
        r_axis = rvec / theta
        K = _hat(r_axis)
        R = (
            torch.eye(3, device=rvec.device)
            + torch.sin(theta) * K
            + (1.0 - torch.cos(theta)) * (K @ K)
        )
    
    return R.numpy() if isinstance(rvec, np.ndarray) else R

def rodrigues_torch_impl2(rvec):
    """Second implementation of Rodrigues formula (OpenCV-like)"""
    # Convert numpy array to torch tensor if needed
    if isinstance(rvec, np.ndarray):
        rvec = torch.from_numpy(rvec).float()
    
    # ノルム(回転角)
    theta = torch.clamp(torch.norm(rvec), min=1e-12)
    # 単位方向
    r_axis = rvec / theta

    # 外積行列K
    K = torch.zeros((3,3), dtype=torch.float32, device=rvec.device)
    K[0,1] = -r_axis[2]
    K[0,2] =  r_axis[1]
    K[1,0] =  r_axis[2]
    K[1,2] = -r_axis[0]
    K[2,0] = -r_axis[1]
    K[2,1] =  r_axis[0]

    # Rodrigues formula
    I = torch.eye(3, dtype=torch.float32, device=rvec.device)
    R = I + torch.sin(theta)*K + (1.0 - torch.cos(theta))*(K @ K)
    return R.numpy() if isinstance(rvec, np.ndarray) else R

def compare_rodrigues_implementations():
    # 基本的なテストケース
    basic_test_cases = [
        np.array([0.1, 0.2, 0.3]),  # 小さな回転
        np.array([1.0, 0.0, 0.0]),  # X軸周りの回転
        np.array([0.0, 1.0, 0.0]),  # Y軸周りの回転
        np.array([0.0, 0.0, 1.0]),  # Z軸周りの回転
        np.array([1.0, 1.0, 1.0]),  # 対角線方向の回転
        np.array([0.001, 0.001, 0.001]),  # 非常に小さな回転
        np.array([3.14, 0.0, 0.0]),  # ほぼπのX軸回転
        np.array([0.0, 0.0, 0.0]),  # ゼロ回転（特殊ケース）
    ]
    
    # より厳しいテストケース
    intensive_test_cases = [
        np.array([1e-8, 1e-8, 1e-8]),  # 極めて小さな回転（small-angle近似の境界付近）
        np.array([1e-6, 1e-6, 1e-6]),  # small-angle近似の境界
        np.array([1e-5, 1e-5, 1e-5]),  # small-angle近似の境界をわずかに超える
        np.array([np.pi, 0.0, 0.0]),   # ちょうどπのX軸回転
        np.array([0.0, np.pi, 0.0]),   # ちょうどπのY軸回転
        np.array([0.0, 0.0, np.pi]),   # ちょうどπのZ軸回転
        np.array([2*np.pi, 0.0, 0.0]), # 2πのX軸回転（実質的に0回転）
        np.array([100.0, 0.0, 0.0]),   # 非常に大きな回転
        np.array([1e-10, 1e10, 1e-10]), # 極端に不均一なスケール
        np.array([np.pi/2, np.pi/3, np.pi/4]), # 複合的な回転
        np.random.randn(3) * 5,         # ランダムな大きな回転
        np.random.randn(3) * 0.0001,    # ランダムな非常に小さな回転
    ]
    
    # 全テストケースを結合
    test_cases = basic_test_cases + intensive_test_cases
    
    results = []
    
    print("Numerical Comparison Test:")
    print("=" * 80)
    
    for i, rvec in enumerate(test_cases):
        # OpenCVのRodrigues
        R_cv, _ = cv2.Rodrigues(rvec)
        
        # PyTorch実装1のRodrigues
        R_torch_impl1 = rodrigues_torch_impl1(rvec)
        
        # PyTorch実装2のRodrigues
        R_torch_impl2 = rodrigues_torch_impl2(rvec)
        
        # PyTorchの結果をNumPy配列に変換
        if torch.is_tensor(R_torch_impl1):
            R_torch_impl1 = R_torch_impl1.numpy()
        if torch.is_tensor(R_torch_impl2):
            R_torch_impl2 = R_torch_impl2.numpy()
        
        # 差分を計算
        diff_cv_impl1 = np.abs(R_cv - R_torch_impl1).max()
        diff_cv_impl2 = np.abs(R_cv - R_torch_impl2).max()
        diff_impl1_impl2 = np.abs(R_torch_impl1 - R_torch_impl2).max()
        
        results.append({
            'case': i+1,
            'rvec': rvec,
            'R_cv': R_cv,
            'R_torch_impl1': R_torch_impl1,
            'R_torch_impl2': R_torch_impl2,
            'diff_cv_impl1': diff_cv_impl1,
            'diff_cv_impl2': diff_cv_impl2,
            'diff_impl1_impl2': diff_impl1_impl2
        })
        
        print(f"Case {i+1}: rvec = {rvec}")
        print(f"OpenCV Rodrigues:\n{R_cv}")
        print(f"PyTorch Rodrigues Impl1 (with small-angle safeguard):\n{R_torch_impl1}")
        print(f"PyTorch Rodrigues Impl2 (OpenCV-like):\n{R_torch_impl2}")
        print(f"Maximum absolute difference (OpenCV vs Impl1): {diff_cv_impl1}")
        print(f"Maximum absolute difference (OpenCV vs Impl2): {diff_cv_impl2}")
        print(f"Maximum absolute difference (Impl1 vs Impl2): {diff_impl1_impl2}")
        print("-" * 80)
    
    # 差分の可視化
    plt.figure(figsize=(15, 8))
    
    # X軸のラベル設定
    x = np.arange(len(test_cases))
    width = 0.25
    
    # 棒グラフの描画
    plt.bar(x - width, [r['diff_cv_impl1'] for r in results], width, label='OpenCV vs Impl1')
    plt.bar(x, [r['diff_cv_impl2'] for r in results], width, label='OpenCV vs Impl2')
    plt.bar(x + width, [r['diff_impl1_impl2'] for r in results], width, label='Impl1 vs Impl2')
    
    plt.xlabel('Test Case')
    plt.ylabel('Maximum Absolute Difference')
    plt.title('Differences between Rodrigues Implementations')
    plt.xticks(x, [f"{i+1}" for i in range(len(test_cases))], rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')  # 対数スケールで表示（差分が小さい場合に見やすくするため）
    plt.legend()
    plt.tight_layout()
    plt.savefig('rodrigues_numerical_comparison.png')
    
    # 全体の最大差分を表示
    max_diff_cv_impl1 = max([r['diff_cv_impl1'] for r in results])
    max_diff_cv_impl2 = max([r['diff_cv_impl2'] for r in results])
    max_diff_impl1_impl2 = max([r['diff_impl1_impl2'] for r in results])
    
    print("\nOverall maximum differences across all test cases:")
    print(f"OpenCV vs Impl1: {max_diff_cv_impl1}")
    print(f"OpenCV vs Impl2: {max_diff_cv_impl2}")
    print(f"Impl1 vs Impl2: {max_diff_impl1_impl2}")
    
    # 特に差分が大きいケースがあれば詳細表示
    threshold = 1e-5
    print(f"\nCases with difference larger than {threshold}:")
    
    for r in results:
        if (r['diff_cv_impl1'] > threshold or 
            r['diff_cv_impl2'] > threshold or 
            r['diff_impl1_impl2'] > threshold):
            
            print(f"Case {r['case']}: rvec = {r['rvec']}")
            if r['diff_cv_impl1'] > threshold:
                print(f"  OpenCV vs Impl1: {r['diff_cv_impl1']}")
            if r['diff_cv_impl2'] > threshold:
                print(f"  OpenCV vs Impl2: {r['diff_cv_impl2']}")
            if r['diff_impl1_impl2'] > threshold:
                print(f"  Impl1 vs Impl2: {r['diff_impl1_impl2']}")
    
    return results

def test_gradient_flow():
    """Test if gradients flow properly through both implementations"""
    print("\nGradient Flow Test:")
    print("=" * 80)
    
    # 勾配テスト用の回転ベクトル
    gradient_test_cases = [
        [0.1, 0.2, 0.3],       # 小さな回転
        [1.0, 0.0, 0.0],       # X軸周りの回転
        [0.001, 0.001, 0.001], # 非常に小さな回転 (small-angle近似が使われる)
        [np.pi, 0.0, 0.0],     # πのX軸回転
        [1e-7, 1e-7, 1e-7],    # 極めて小さな回転 (実装間の分岐点近辺)
    ]
    
    # グラフ用のデータ
    gradient_results = []
    
    for i, rvec_vals in enumerate(gradient_test_cases):
        print(f"Gradient Test Case {i+1}: rvec = {rvec_vals}")
        
        # テスト用のターゲット回転行列（単位行列からわずかに回転）
        target_rotation = torch.tensor([
            [0.9, -0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        # 実装1の勾配テスト
        rvec1 = torch.tensor(rvec_vals, dtype=torch.float32, requires_grad=True)
        R1 = rodrigues_torch_impl1(rvec1)
        loss1 = torch.nn.functional.mse_loss(R1, target_rotation)
        loss1.backward()
        grad1 = rvec1.grad.clone()
        
        # 実装2の勾配テスト
        rvec2 = torch.tensor(rvec_vals, dtype=torch.float32, requires_grad=True)
        R2 = rodrigues_torch_impl2(rvec2)
        loss2 = torch.nn.functional.mse_loss(R2, target_rotation)
        loss2.backward()
        grad2 = rvec2.grad.clone()
        
        # 勾配の差分
        grad_diff = torch.abs(grad1 - grad2).max().item()
        grad_ratio = torch.norm(grad1) / torch.norm(grad2)
        
        gradient_results.append({
            'rvec': rvec_vals,
            'grad1': grad1.tolist(),
            'grad2': grad2.tolist(),
            'grad_diff': grad_diff,
            'grad_ratio': grad_ratio.item(),
            'has_grad1': grad1.abs().sum().item() > 0,
            'has_grad2': grad2.abs().sum().item() > 0
        })
        
        print(f"Implementation 1 gradient: {grad1}")
        print(f"Implementation 2 gradient: {grad2}")
        print(f"Maximum gradient difference: {grad_diff}")
        print(f"Gradient norm ratio (Impl1/Impl2): {grad_ratio.item()}")
        print(f"Implementation 1 has gradient: {'Yes' if grad1.abs().sum().item() > 0 else 'No'}")
        print(f"Implementation 2 has gradient: {'Yes' if grad2.abs().sum().item() > 0 else 'No'}")
        print("-" * 80)
    
    # 勾配差分の可視化
    plt.figure(figsize=(12, 6))
    x = np.arange(len(gradient_test_cases))
    
    plt.subplot(1, 2, 1)
    plt.bar(x, [r['grad_diff'] for r in gradient_results])
    plt.xlabel('Test Case')
    plt.ylabel('Maximum Gradient Difference')
    plt.title('Gradient Difference Between Implementations')
    plt.xticks(x, [f"{i+1}" for i in range(len(gradient_test_cases))])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.bar(x, [r['grad_ratio'] for r in gradient_results])
    plt.xlabel('Test Case')
    plt.ylabel('Gradient Norm Ratio (Impl1/Impl2)')
    plt.title('Gradient Norm Ratio')
    plt.xticks(x, [f"{i+1}" for i in range(len(gradient_test_cases))])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='-')
    
    plt.tight_layout()
    plt.savefig('rodrigues_gradient_comparison.png')
    
    # 勾配テスト結果のサマリー
    print("\nGradient Test Summary:")
    all_have_grad1 = all(r['has_grad1'] for r in gradient_results)
    all_have_grad2 = all(r['has_grad2'] for r in gradient_results)
    
    print(f"Implementation 1 propagates gradients for all test cases: {'Yes' if all_have_grad1 else 'No'}")
    print(f"Implementation 2 propagates gradients for all test cases: {'Yes' if all_have_grad2 else 'No'}")
    
    if not all_have_grad1:
        failed_cases = [i+1 for i, r in enumerate(gradient_results) if not r['has_grad1']]
        print(f"Implementation 1 failed to propagate gradients for cases: {failed_cases}")
    
    if not all_have_grad2:
        failed_cases = [i+1 for i, r in enumerate(gradient_results) if not r['has_grad2']]
        print(f"Implementation 2 failed to propagate gradients for cases: {failed_cases}")
    
    # 勾配比率の評価
    large_ratio_diff = [i+1 for i, r in enumerate(gradient_results) 
                        if r['grad_ratio'] > 1.5 or r['grad_ratio'] < 0.67]
    
    if large_ratio_diff:
        print(f"Cases with large gradient ratio differences: {large_ratio_diff}")
        print("These cases may indicate different gradient behavior between implementations.")
    else:
        print("Gradient ratios are within reasonable range for all test cases.")
    
    return gradient_results

if __name__ == "__main__":
    print("Running numerical comparison tests...")
    results = compare_rodrigues_implementations()
    
    print("\nRunning gradient flow tests...")
    gradient_results = test_gradient_flow()
    
    plt.show()