import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def rodrigues_torch(rvec):
    """PyTorch implementation of Rodrigues formula"""
    # Convert numpy array to torch tensor if needed
    if isinstance(rvec, np.ndarray):
        rvec = torch.from_numpy(rvec).float()
    
    # ノルム(回転角)
    theta = torch.norm(rvec) + 1e-12
    # 単位方向
    r_axis = rvec / theta

    # 外積行列K
    K = torch.zeros((3,3), dtype=torch.float32)
    K[0,1] = -r_axis[2]
    K[0,2] =  r_axis[1]
    K[1,0] =  r_axis[2]
    K[1,2] = -r_axis[0]
    K[2,0] = -r_axis[1]
    K[2,1] =  r_axis[0]

    # Rodrigues formula
    I = torch.eye(3, dtype=torch.float32)
    R = I + torch.sin(theta)*K + (1.0 - torch.cos(theta))*(K @ K)
    return R.numpy() if isinstance(rvec, np.ndarray) else R

def compare_rodrigues_implementations():
    # テスト用の回転ベクトルを生成
    test_cases = [
        np.array([0.1, 0.2, 0.3]),  # 小さな回転
        np.array([1.0, 0.0, 0.0]),  # X軸周りの回転
        np.array([0.0, 1.0, 0.0]),  # Y軸周りの回転
        np.array([0.0, 0.0, 1.0]),  # Z軸周りの回転
        np.array([1.0, 1.0, 1.0]),  # 対角線方向の回転
        np.array([0.001, 0.001, 0.001]),  # 非常に小さな回転
        np.array([3.14, 0.0, 0.0]),  # ほぼπのX軸回転
        np.array([0.0, 0.0, 0.0]),  # ゼロ回転（特殊ケース）
    ]
    
    results = []
    
    for i, rvec in enumerate(test_cases):
        # OpenCVのRodrigues
        R_cv, _ = cv2.Rodrigues(rvec)
        
        # PyTorchのRodrigues
        R_torch = rodrigues_torch(rvec)
        
        # PyTorchの結果をNumPy配列に変換
        if torch.is_tensor(R_torch):
            R_torch = R_torch.numpy()
        
        # 差分を計算
        diff = np.abs(R_cv - R_torch).max()
        
        results.append({
            'case': i+1,
            'rvec': rvec,
            'R_cv': R_cv,
            'R_torch': R_torch,
            'max_diff': diff
        })
        
        print(f"Case {i+1}: rvec = {rvec}")
        print(f"OpenCV Rodrigues:\n{R_cv}")
        print(f"PyTorch Rodrigues:\n{R_torch}")
        print(f"Maximum absolute difference: {diff}")
        print("-" * 50)
    
    # 差分の可視化
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(results)+1), [r['max_diff'] for r in results])
    plt.xlabel('Test Case')
    plt.ylabel('Maximum Absolute Difference')
    plt.title('Difference between OpenCV and PyTorch Rodrigues Implementations')
    plt.xticks(range(1, len(results)+1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')  # 対数スケールで表示（差分が小さい場合に見やすくするため）
    plt.tight_layout()
    plt.savefig('rodrigues_comparison.png')
    plt.show()
    
    return results

if __name__ == "__main__":
    results = compare_rodrigues_implementations()
    
    # 全体の最大差分を表示
    max_overall_diff = max([r['max_diff'] for r in results])
    print(f"\nOverall maximum difference across all test cases: {max_overall_diff}")
    
    # 特に差分が大きいケースがあれば詳細表示
    threshold = 1e-5
    large_diff_cases = [r for r in results if r['max_diff'] > threshold]
    if large_diff_cases:
        print(f"\nCases with difference larger than {threshold}:")
        for case in large_diff_cases:
            print(f"Case {case['case']}: rvec = {case['rvec']}, max_diff = {case['max_diff']}")