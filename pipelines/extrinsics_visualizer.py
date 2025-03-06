import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

##############################################################################
# CameraPoseVisualizer
##############################################################################
class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color_map='red', focal_len_scaled=5, aspect_ratio=0.3, plotly_viz=False, legend_group='g', name='n', show_legend=True):
        """
        カメラの外部パラメータ(4x4行列)を受け取り、カメラフラスタム(ピラミッド)を3Dで描画します。
        """
        # 標準化されたカメラフラスタムの頂点（カメラ原点含む）
        vertex_std = np.array([
            [0, 0, 0, 1],  # カメラ原点
            [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
            [focal_len_scaled * aspect_ratio,  focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
            [-focal_len_scaled * aspect_ratio,  focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
            [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        ])
        # 世界座標系に変換
        vertex_transformed = vertex_std @ extrinsic.T
        
        # フラスタムを構成するポリゴン面
        # meshes は、各面を構成する頂点の配列。
        meshes = [
            [vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]
        ]

        # matplotlib上での色指定
        if isinstance(color_map, str):
            color = color_map
        else:
            color = plt.cm.rainbow(color_map)

        # Matplotlib で描画
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35)
        )

        # Plotly での描画用データを作成
        if plotly_viz:
            # Plotly用のカラースケールから色を取得
            color_plotly = sample_colorscale('rainbow', color_map)[0] if not isinstance(color_map, str) else color_map
            self.plotly_data = self.draw_interactive(
                meshes=meshes,
                color=color_plotly,
                legend_group=legend_group,
                name=name,
                show_legend=show_legend
            )

    def draw_interactive(self, meshes, color, legend_group, name, show_legend):
        """
        Plotly用の3D Meshデータを生成
        """
        # meshes の座標を取り出す
        x = []
        y = []
        z = []
        for face in meshes:
            for vertex in face:
                x.append(vertex[0])
                y.append(vertex[1])
                z.append(vertex[2])

        # カメラ中心（最初の面の最初の頂点を利用。0番目=カメラ原点を共有している）
        camera_pos = meshes[0][0]

        data = go.Mesh3d(
            x=x, 
            y=y, 
            z=z, 
            opacity=0.7,
            color=color, 
            showlegend=show_legend, 
            legendgroup=legend_group, 
            name=name,
            hoverinfo='text',
            hovertext=f"{name}<br>Position: ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})",
            hoverlabel=dict(bgcolor='white', font_size=12)
        )

        return data

    def customize_legend(self, list_label):
        """
        Matplotlib用のレジェンドをカスタマイズします（未使用の例）
        """
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        """
        カメラのフレーム番号をカラースケールにする場合の目盛り
        """
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        """
        Matplotlibによる表示
        """
        plt.title('Extrinsic Parameters')
        plt.show()


##############################################################################
# COLMAPファイル読み込み用関数
##############################################################################
def read_images_txt(images_txt_path):
    """
    COLMAPのimages.txtからカメラ姿勢を読み込む
    (テキスト形式)
    """
    cameras = []
    with open(images_txt_path, 'r') as f:
        is_camera_line = True
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if is_camera_line:
                # フォーマット: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                elements = line.split()
                image_id = int(elements[0])
                qw, qx, qy, qz = map(float, elements[1:5])
                tx, ty, tz = map(float, elements[5:8])
                camera_id = int(elements[8])
                image_name = elements[9]
                
                # クォータニオンを正規化
                q = np.array([qw, qx, qy, qz], dtype=float)
                q = q / np.linalg.norm(q)
                
                # world->camera の回転行列をクォータニオンから計算
                R = np.zeros((3, 3))
                R[0, 0] = 1 - 2 * (q[2]**2 + q[3]**2)
                R[0, 1] = 2 * (q[1] * q[2] - q[3] * q[0])
                R[0, 2] = 2 * (q[1] * q[3] + q[2] * q[0])
                R[1, 0] = 2 * (q[1] * q[2] + q[3] * q[0])
                R[1, 1] = 1 - 2 * (q[1]**2 + q[3]**2)
                R[1, 2] = 2 * (q[2] * q[3] - q[1] * q[0])
                R[2, 0] = 2 * (q[1] * q[3] - q[2] * q[0])
                R[2, 1] = 2 * (q[2] * q[3] + q[1] * q[0])
                R[2, 2] = 1 - 2 * (q[1]**2 + q[2]**2)
                
                # world->camera の並進ベクトル
                t = np.array([tx, ty, tz], dtype=float).reshape(3, 1)
                
                # カメラ中心 C(ワールド座標系) = - R^T * t
                # カメラ->ワールドの変換行列（回転）= R^T
                C = -R.T @ t

                # extrinsic(4x4)の作成: カメラ座標系->ワールド座標系
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = R.T
                extrinsic[:3, 3] = C.flatten()
                
                cameras.append({
                    'image_id': image_id,
                    'camera_id': camera_id,
                    'name': image_name,
                    'extrinsic': extrinsic,
                    'quaternion': q,
                    'position': C.flatten(),
                    'R': R,
                    't': t.flatten()
                })
            
            # images.txt では2行ごとに画像情報が入っており、2行目は2D点のリストなのでスキップ
            is_camera_line = not is_camera_line
    
    return cameras

def read_images_bin(images_bin_path):
    """
    COLMAPのimages.binからカメラ姿勢を読み込む
    (バイナリ形式)
    参考: https://colmap.github.io/format.html
    """
    import struct
    cameras = []
    with open(images_bin_path, 'rb') as f:
        # 最初に画像数（uint64）を読み込む
        num_images = struct.unpack('Q', f.read(8))[0]
        
        for _ in range(num_images):
            # 1つのイメージ情報を読み込む
            # フォーマット:
            #   IMAGE_ID (uint32) | QW, QX, QY, QZ (double*4) | TX, TY, TZ (double*3) | CAMERA_ID (uint32)
            #   その後に NAME (\0終端文字列) と 2D点群 が続く
            
            # 4 + (4*8) + (3*8) + 4 = 4 + 32 + 24 + 4 = 64 バイト
            data_header = struct.unpack('<I 4d 3d I', f.read(64))
            image_id = data_header[0]
            qw, qx, qy, qz = data_header[1:5]
            tx, ty, tz = data_header[5:8]
            camera_id = data_header[8]
            
            # 次に NAME (\0終端) を読み込む
            name_bytes = []
            while True:
                c = f.read(1)
                if c == b'\x00' or c == b'':
                    break
                name_bytes.append(c)
            image_name = b''.join(name_bytes).decode('utf-8')
            
            # 次に POINTS2D の数 (#points2D) をuint64で読み込む
            num_points2D = struct.unpack('Q', f.read(8))[0]
            # さらに各2D点のデータ (x, y, point3D_id) をスキップ
            #   x, y: double (8バイト*2 =16バイト)
            #   point3D_id: int64 (8バイト)
            # → 合計24バイト/点
            f.seek(num_points2D * 24, os.SEEK_CUR)
            
            # クォータニオンを正規化
            q = np.array([qw, qx, qy, qz], dtype=float)
            q = q / np.linalg.norm(q)
            
            # world->camera の回転行列をクォータニオンから計算
            R = np.zeros((3, 3))
            w, x, y, z = q
            R[0, 0] = 1 - 2 * (y**2 + z**2)
            R[0, 1] = 2 * (x*y - z*w)
            R[0, 2] = 2 * (x*z + y*w)
            R[1, 0] = 2 * (x*y + z*w)
            R[1, 1] = 1 - 2 * (x**2 + z**2)
            R[1, 2] = 2 * (y*z - x*w)
            R[2, 0] = 2 * (x*z - y*w)
            R[2, 1] = 2 * (y*z + x*w)
            R[2, 2] = 1 - 2 * (x**2 + y**2)

            # 並進ベクトル (world->camera)
            t = np.array([tx, ty, tz], dtype=float).reshape(3, 1)
            
            # カメラ中心 C(ワールド座標系) = - R^T * t
            C = -R.T @ t
            
            # camera->world の4x4行列
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R.T
            extrinsic[:3, 3] = C.flatten()
            
            cameras.append({
                'image_id': image_id,
                'camera_id': camera_id,
                'name': image_name,
                'extrinsic': extrinsic,
                'quaternion': q,
                'position': C.flatten(),
                'R': R,
                't': t.flatten()
            })
            
    return cameras

def read_images(images_path):
    """
    images.txt or images.bin に応じて読み込み関数を切り替える
    """
    if images_path.endswith('.txt'):
        return read_images_txt(images_path)
    elif images_path.endswith('.bin'):
        return read_images_bin(images_path)
    else:
        raise ValueError("images file must have .txt or .bin extension")

def read_points3D_txt(points3D_txt_path):
    """
    COLMAPのpoints3D.txtから3D点群を読み込む (テキスト形式)
    """
    points = []
    colors = []
    with open(points3D_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elements = line.split()
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK...
            x, y, z = map(float, elements[1:4])
            r, g, b = map(int, elements[4:7])
            points.append([x, y, z])
            colors.append([r, g, b])
    return np.array(points), np.array(colors)

def read_points3D_bin(points3D_bin_path):
    """
    COLMAPのpoints3D.binから3D点群を読み込む (バイナリ形式)
    """
    import struct
    points = []
    colors = []
    with open(points3D_bin_path, 'rb') as f:
        # 最初に3D点の数を読み込む (uint64)
        num_points = struct.unpack('Q', f.read(8))[0]
        
        # points3D.binの1点あたりのヘッダフォーマット
        # <Q 3d 3B d Q> = 
        #   point_id (uint64),
        #   X, Y, Z (double*3),
        #   R, G, B (uint8*3),
        #   ERROR (double),
        #   TRACK_LENGTH (uint64),
        # その後に (IMAGE_ID(uint32), POINT2D_IDX(uint32)) x TRACK_LENGTH が続く
        point_data_struct = struct.Struct('<Q 3d 3B d Q')
        
        for _ in range(num_points):
            data = point_data_struct.unpack(f.read(point_data_struct.size))
            # data[0] = point_id
            x, y, z = data[1:4]
            r, g, b = data[4:7]
            # data[7] = error
            track_len = data[8]
            
            points.append([x, y, z])
            colors.append([r, g, b])
            
            # TRACK_LENGTH 個の (IMAGE_ID, POINT2D_IDX) をスキップ
            #   (uint32, uint32) = 4+4 = 8 バイト
            f.seek(track_len * 8, os.SEEK_CUR)
    
    return np.array(points), np.array(colors)

def read_points3D(points3D_path):
    """
    points3D.txt か points3D.bin かで読み込み関数を切り替え
    """
    if points3D_path.endswith('.txt'):
        return read_points3D_txt(points3D_path)
    elif points3D_path.endswith('.bin'):
        return read_points3D_bin(points3D_path)
    else:
        raise ValueError("Points3D file must have .txt or .bin extension")


##############################################################################
# 可視化用関数
##############################################################################
def visualize_cameras_and_points(
    cameras,
    points=None,
    colors=None,
    scene_bounds=None,
    use_plotly=True,
    max_points=50000,
    focal_len=5,
    aspect_ratio=0.3
):
    """
    カメラの姿勢と3D点群を可視化する。

    Args:
        cameras: read_images_txt / read_images_bin で得られるカメラ情報のリスト
        points: (N, 3) の3D点群座標
        colors: (N, 3) のRGB色 (0-255)
        scene_bounds: ( (x_min, x_max), (y_min, y_max), (z_min, z_max) ) のタプル
                      None の場合は自動計算
        use_plotly: Trueの場合はPlotlyによるインタラクティブ可視化
        max_points: 表示する最大点数（多すぎる場合はサンプリング）
        focal_len: カメラフラスタムの奥行きスケール
        aspect_ratio: カメラフラスタムの画角スケール
    """
    # カメラ中心座標をまとめる
    camera_positions = np.array([cam['position'] for cam in cameras])
    
    # scene_bounds が未指定なら自動推定する
    if scene_bounds is None:
        # まず全カメラのフラスタム頂点を集めて、そこから境界を推定する
        all_vertices = []
        for cam in cameras:
            vertex_std = np.array([
                [0, 0, 0, 1],
                [focal_len * aspect_ratio, -focal_len * aspect_ratio, focal_len, 1],
                [focal_len * aspect_ratio,  focal_len * aspect_ratio, focal_len, 1],
                [-focal_len * aspect_ratio,  focal_len * aspect_ratio, focal_len, 1],
                [-focal_len * aspect_ratio, -focal_len * aspect_ratio, focal_len, 1]
            ])
            vertex_transformed = vertex_std @ cam['extrinsic'].T
            all_vertices.extend([v[:-1] for v in vertex_transformed])
        
        all_vertices = np.array(all_vertices)
        
        # カメラ中心とフラスタム頂点を合わせた点群
        all_points_for_bound = np.vstack([camera_positions, all_vertices])
        
        # 中心と最大半径を計算
        center = all_points_for_bound.mean(axis=0)
        distances = np.linalg.norm(all_points_for_bound - center, axis=1)
        max_dist = np.max(distances)
        
        scene_bounds = (
            [center[0] - max_dist*1.2, center[0] + max_dist*1.2],
            [center[1] - max_dist*1.2, center[1] + max_dist*1.2],
            [center[2] - max_dist*1.2, center[2] + max_dist*1.2]
        )
    
    # Matplotlib 用ビジュアライザを作成
    visualizer = CameraPoseVisualizer(
        scene_bounds[0], scene_bounds[1], scene_bounds[2]
    )
    
    # Plotly の場合は Figure をあらかじめ作成
    if use_plotly:
        final_layout = go.Figure()
        final_layout.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=scene_bounds[0]),
                yaxis=dict(nticks=4, range=scene_bounds[1]),
                zaxis=dict(nticks=4, range=scene_bounds[2]),
                aspectmode='data'
            ),
            title='Camera and Point Cloud Visualization',
            legend=dict(x=0.8, y=0.5, font=dict(color='black', size=12))
        )
    
    # 各カメラを追加描画
    for i, cam in enumerate(cameras):
        color_val = i / max(1, len(cameras) - 1)  # カラースケール用(0-1)
        camera_name = f"Camera {i+1}: {cam['name']}"
        
        # Matplotlib用のフラスタム描画
        visualizer.extrinsic2pyramid(
            cam['extrinsic'], 
            color_map=color_val,
            focal_len_scaled=focal_len,
            aspect_ratio=aspect_ratio,
            plotly_viz=use_plotly,
            legend_group="Cameras",
            name=camera_name,
            show_legend=True
        )
        
        # Plotly用データが更新されるので、それを Figure に追加
        if use_plotly:
            final_layout.add_trace(visualizer.plotly_data)
    
    # 3D点群を追加
    if points is not None and len(points) > 0:
        # 点数が多い場合はサンプリング
        if len(points) > max_points:
            print(f"Subsampling point cloud from {len(points)} to {max_points} points")
            indices = np.random.choice(len(points), max_points, replace=False)
            points_subset = points[indices]
            colors_subset = colors[indices] if colors is not None else None
        else:
            points_subset = points
            colors_subset = colors
        
        # Plotly の場合
        if use_plotly:
            if colors_subset is None:
                colors_subset = np.ones((len(points_subset), 3)) * np.array([255, 0, 0])  # デフォルト赤
            
            # (R,G,B)を "rgb(r,g,b)" の文字列に変換
            point_colors_hex = [f'rgb({r},{g},{b})' for r, g, b in colors_subset]
            
            final_layout.add_trace(
                go.Scatter3d(
                    x=points_subset[:, 0],
                    y=points_subset[:, 1],
                    z=points_subset[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1.5,
                        color=point_colors_hex,
                        opacity=0.7
                    ),
                    name='3D Points',
                    showlegend=True
                )
            )
        else:
            # Matplotlib の場合
            if colors_subset is None:
                point_colors_mpl = np.ones((len(points_subset), 3)) * np.array([1.0, 0.0, 0.0])
            else:
                point_colors_mpl = colors_subset.astype(float) / 255.0
            
            visualizer.ax.scatter(
                points_subset[:, 0], points_subset[:, 1], points_subset[:, 2],
                c=point_colors_mpl, s=0.5, alpha=0.5
            )
    
    # 可視化を実行
    if use_plotly:
        # カメラ選択用のボタンを作成
        camera_buttons = []
        for i, cam in enumerate(cameras):
            # 各カメラ以外を非表示にする設定
            #   トレースの並び:
            #   0         -> Camera1
            #   1         -> Camera2
            #   ...
            #   len(cameras)-1 -> CameraN
            #   次が3D点(あるなら1トレース)
            
            # カメラ部分: j == i のみ True, それ以外 False
            visibility_list = [(j == i) for j in range(len(cameras))]
            # 3D点がある場合は末尾のトレースを常に表示する ( True )
            if points is not None and len(points) > 0:
                visibility_list.append(True)
            
            camera_buttons.append(
                dict(
                    method="update",
                    args=[
                        {"visible": visibility_list},
                        {"title": f"Camera {i+1}: {cam['name']}"}
                    ],
                    label=f"Camera {i+1}"
                )
            )
        
        # "Show All" ボタン
        show_all_visibility = [True]*len(cameras)
        if points is not None and len(points) > 0:
            show_all_visibility.append(True)
        
        camera_buttons.append(
            dict(
                method="update",
                args=[
                    {"visible": show_all_visibility},
                    {"title": "All Cameras"}
                ],
                label="Show All"
            )
        )
        
        # ボタンを配置
        final_layout.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    buttons=camera_buttons,
                    showactive=True,
                    x=0.1,
                    y=1.15,
                    xanchor="left",
                    yanchor="top"
                )
            ]
        )
        final_layout.show()
    else:
        # Matplotlib の場合のカラーバー設定
        visualizer.colorbar(len(cameras))
        visualizer.show()


##############################################################################
# メイン処理
##############################################################################
def main():
    parser = argparse.ArgumentParser(description='Visualize COLMAP camera poses and 3D points')
    parser.add_argument('--images', type=str, required=True,
                        help='Path to COLMAP images.txt or images.bin file')
    parser.add_argument('--points', type=str, default=None,
                        help='Path to COLMAP points3D.txt or points3D.bin file')
    parser.add_argument('--plotly', action='store_true', default=True,
                        help='Use Plotly for interactive visualization')
    parser.add_argument('--max_points', type=int, default=50000,
                        help='Maximum number of points to display')
    parser.add_argument('--focal_len', type=float, default=2,
                        help='Focal length for camera visualization')
    parser.add_argument('--aspect_ratio', type=float, default=0.1,
                        help='Aspect ratio for camera visualization')
    
    args = parser.parse_args()
    
    # カメラ情報を読み込み
    cameras = read_images(args.images)
    print(f"Loaded {len(cameras)} cameras from {args.images}")
    
    points = None
    colors = None
    
    # 3D点群を読み込み（オプション）
    if args.points:
        try:
            points, colors = read_points3D(args.points)
            print(f"Loaded {len(points)} points from {args.points}")
        except Exception as e:
            print(f"Error loading points: {e}")
            import traceback
            traceback.print_exc()
    
    # カメラ・点群を可視化
    visualize_cameras_and_points(
        cameras,
        points=points,
        colors=colors,
        use_plotly=args.plotly,
        max_points=args.max_points,
        focal_len=args.focal_len,
        aspect_ratio=args.aspect_ratio
    )

if __name__ == "__main__":
    main()

#python extrinsics_visualizer.py --images /Users/kohsukeide/dev/perspective-n-gaussian/pipelines/results/final/colmap/images.txt --points /Users/kohsukeide/dev/perspective-n-gaussian/pipelines/results/final/colmap/points3d.txt

#python extrinsics_visualizer.py --images /Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/sparse/0/images_correct.txt --points /Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/sparse/0/points3D.txt

#python extrinsics_visualizer.py --images /Users/kohsukeide/dev/perspective-n-gaussian/data/nerf_synthetic/textureless/sparse/0/images.bin --points /Users/kohsukeide/dev/perspective-n-gaussian/data/nerf_synthetic/textureless/sparse/0/points3D.bin
