# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
import numpy as np  
from scipy.spatial.distance import cdist  


class TrajectoryViewerWithFormation:
    def __init__(self, root):
        self.root = root
        self.root.title("Trajectory Viewer with Formation Detection")
        self.file_path = ""
        self.right_court_image_path = "../img/right_court.png"
        self.left_court_image_path = "../img/left_court.png"
        self.right_court_image = None
        self.left_court_image = None

        tk.Button(root, text="CSVを選択", command=self.load_csv).pack(pady=5)

        self.start_entry = None
        self.end_entry = None
        self.figure = None
        self.canvas = None

        # Define offsets as constants
        self.RED_X_OFFSET = 0.1
        self.RED_Y_OFFSET = -0.1
        self.WHITE_X_OFFSET = 0.0
        self.WHITE_Y_OFFSET = 0.0

        # フレームを進めるボタンを追加
        frame_control_frame = tk.Frame(self.root)
        frame_control_frame.pack()

        prev_button = tk.Button(frame_control_frame, text="前のフレーム", command=self.prev_frame)
        prev_button.pack(side="left", padx=5)

        next_button = tk.Button(frame_control_frame, text="次のフレーム", command=self.next_frame)
        next_button.pack(side="left", padx=5)

        # Add UI for start frame and trajectory range
        tk.Label(self.root, text="開始フレーム").pack()
        self.start_frame_entry = tk.Entry(self.root)
        self.start_frame_entry.pack()

        tk.Label(self.root, text="描画範囲 (フレーム数)").pack()
        self.trajectory_range_entry = tk.Entry(self.root)
        self.trajectory_range_entry.pack()

        # Add "開始" button to update the plot
        start_button = tk.Button(self.root, text="開始", command=self.update_plot)
        start_button.pack(pady=5)

    def prev_frame(self):
        if hasattr(self, 'start_frame_entry'):
            try:
                current_frame = int(self.start_frame_entry.get())
                if current_frame > self.min_frame:
                    self.start_frame_entry.delete(0, tk.END)
                    self.start_frame_entry.insert(0, str(current_frame - 1))
                    self.update_plot()
            except ValueError:
                print("フレーム番号が不正です")

    def next_frame(self):
        if hasattr(self, 'start_frame_entry'):
            try:
                current_frame = int(self.start_frame_entry.get())
                if current_frame < self.max_frame:
                    self.start_frame_entry.delete(0, tk.END)
                    self.start_frame_entry.insert(0, str(current_frame + 1))
                    self.update_plot()
            except ValueError:
                print("フレーム番号が不正です")

    def load_csv(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            self.df = pd.read_csv(self.file_path)
            print("CSV読み込み完了")

            self.min_frame = self.df["frame_num"].min()
            self.max_frame = self.df["frame_num"].max()

            # Update start frame and trajectory range UI
            self.start_frame_entry.delete(0, tk.END)
            self.start_frame_entry.insert(0, str(self.min_frame))

            self.trajectory_range_entry.delete(0, tk.END)
            self.trajectory_range_entry.insert(0, "1")

            self.load_court_images()
            self.update_plot()

    def load_court_images(self):
        try:
            self.right_court_image = mpimg.imread(self.right_court_image_path)
            self.left_court_image = mpimg.imread(self.left_court_image_path)
            print("コート画像読み込み完了")
        except Exception as e:
            print(f"コート画像の読み込みに失敗しました: {e}")

    def detect_formation(self):
        if not hasattr(self, 'df'):
            print("データが読み込まれていません")
            return

        try:
            start_frame = int(self.start_frame_entry.get())
            trajectory_range = int(self.trajectory_range_entry.get())
        except ValueError:
            print("フレーム番号または描画範囲が不正です")
            return

        end_frame = start_frame + trajectory_range

        # フレーム範囲のデータを取得
        df_filtered = self.df[(self.df["frame_num"] >= start_frame) & (self.df["frame_num"] < end_frame)]
        if df_filtered.empty:
            print("指定されたフレーム範囲にデータがありません")
            return

    def update_plot(self, event=None):
        if not hasattr(self, 'df'):
            return

        try:
            start_frame = int(self.start_frame_entry.get())
            trajectory_range = int(self.trajectory_range_entry.get())
        except ValueError:
            print("フレーム番号または描画範囲が不正です")
            return

        end_frame = start_frame + trajectory_range

        # フレーム範囲のデータを取得
        df_filtered = self.df[(self.df["frame_num"] >= start_frame) & (self.df["frame_num"] < end_frame)]
        if df_filtered.empty:
            print("指定されたフレーム範囲にデータがありません")
            return

        if self.figure is None:
            self.figure = plt.Figure(figsize=(10, 6))
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.ax.clear()

        directions = df_filtered["direction"].unique()

        if len(directions) == 1:
            direction = directions[0]
            if direction == "right" and self.right_court_image is not None:
                self.ax.imshow(self.right_court_image, extent=[0, 1, 1, 0], aspect='auto')
            elif direction == "left" and self.left_court_image is not None:
                self.ax.imshow(self.left_court_image, extent=[0, 1, 1, 0], aspect='auto')

            # Use constants for offsets
            x_offset = self.RED_X_OFFSET
            y_offset = self.RED_Y_OFFSET
            white_x_offset = self.WHITE_X_OFFSET
            white_y_offset = self.WHITE_Y_OFFSET

            positions = []

            #検出した選手のうち誤検出と思われる選手を除外する
            # フィルタリング条件を定義
            #ゴールラインよりも内側の選手を除外
            #ゴールラインはrightの場合は(1.0,0.4)を基準として半径0.4の円と,(1.0,0.6)を基準として半径0.4の円と，(0.7,0.4)(0.7,0.6)(1.0,0.6)(1.0,0.4)を4点とする四角形の範囲内全てとする
            #leftの場合は(0.0,0.4)を基準として半径0.4の円と,(0.0,0.6)を基準として半径0.4の円と，(0.3,0.4)(0.3,0.6)(0.0,0.6)(0.0,0.4)を4点とする四角形の範囲内全てとする
            if direction == "right":
                # ゴールラインよりも内側の選手を除外
                df_filtered = df_filtered[
                    ~(
                        # 半径0.4の円 (1.0, 0.4) を基準
                        ((df_filtered["x"] - 1.0) ** 2 + (df_filtered["y"] - 0.4) ** 2 <= 0.4 ** 2) |
                        # 半径0.4の円 (1.0, 0.6) を基準
                        ((df_filtered["x"] - 1.0) ** 2 + (df_filtered["y"] - 0.6) ** 2 <= 0.4 ** 2) |
                        # 四角形範囲 (0.7, 0.4), (0.7, 0.6), (1.0, 0.6), (1.0, 0.4)
                        ((df_filtered["x"] >= 0.7) & (df_filtered["x"] <= 1.0) & (df_filtered["y"] >= 0.4) & (df_filtered["y"] <= 0.6)) |
                        # x座標が0.4より小さい選手
                        (df_filtered["x"] < 0.4)
                    )
                ]
            elif direction == "left":
                # ゴールラインよりも内側の選手を除外
                df_filtered = df_filtered[
                    ~((df_filtered["x"] <= 0.4) & (df_filtered["y"] <= 0.4) & (df_filtered["y"] >= 0.6)) &
                    ~((df_filtered["x"] <= 0.3) & (df_filtered["y"] <= 0.4) & (df_filtered["y"] >= 0.6))
                ]
            # フィルタリング後のデータを取得
            df_filtered = df_filtered[(df_filtered["frame_num"] >= start_frame) & (df_filtered["frame_num"] < end_frame)]

            # フレームごとに選手の位置をプロット
            for frame in range(start_frame, end_frame):
                frame_data = df_filtered[df_filtered["frame_num"] == frame]
                for _, person_data in frame_data.iterrows():
                    x, y = person_data["x"], person_data["y"]
                    team_color = person_data["team_color"]

                    # Redチームの場合、座標にオフセットを適用
                    if team_color == "red":
                        x += x_offset
                        y += y_offset

                    # Whiteチームの場合、座標にオフセットを適用
                    if team_color == "white":
                        x += white_x_offset
                        y += white_y_offset

                    if (direction == "left" and team_color != "white") or (direction == "right" and team_color != "red"):
                        continue

                    # 最後のフレームのみ黒色、それ以外は半透明な灰色でプロット
                    color = "black" if frame == end_frame - 1 else "gray"
                    alpha = 1.0 if frame == end_frame - 1 else 0.5
                    self.ax.scatter(x, y, color=color, alpha=alpha, s=50)

                    # Collect positions for 0-6 formation detection
                    positions.append((x, y))

            ###########################9mと6mラインの描画ここから#############################

            # # 9mラインを描画(directionがleft)
            # if direction == "left":
            #     origins = [(0, 0.4), (0, 0.6)]  # 起点の2パターン
            #     distance = 0.45  # 距離
            #     angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)  # 60点を均等に配置

            #     for origin_x, origin_y in origins:
            #         for angle in angles:
            #             fixed_x = origin_x + distance * np.cos(angle)
            #             fixed_y = origin_y + distance * np.sin(angle)

            #             # 条件に基づいて描画
            #             if origin_y == 0.4 and fixed_y <= 0.4:  # y座標が0.4より小さい位置
            #                 self.ax.scatter(fixed_x, fixed_y, color="blue", s=50)
            #             elif origin_y == 0.6 and fixed_y >= 0.6:  # y座標が0.6より大きい位置
            #                 self.ax.scatter(fixed_x, fixed_y, color="blue", s=50)

            #     # 起点を結ぶ点を描画
            #     connecting_angles = np.linspace(0, 1, 5)  # 起点間を5点で補間
            #     for t in connecting_angles:
            #         connecting_x = 0.45  # x座標を固定
            #         connecting_y = 0.4 + t * (0.6 - 0.4)  # y座標を線形補間
            #         self.ax.scatter(connecting_x, connecting_y, color="blue", s=50)

            #     # 6mラインを描画
            #     distance = 0.3  # 距離を0.3に設定
            #     for origin_x, origin_y in origins:
            #         for angle in angles:
            #             fixed_x = origin_x + distance * np.cos(angle)
            #             fixed_y = origin_y + distance * np.sin(angle)

            #             # 条件に基づいて描画
            #             if origin_y == 0.4 and fixed_y <= 0.4:  # y座標が0.4より小さい位置
            #                 self.ax.scatter(fixed_x, fixed_y, color="blue", s=50)
            #             elif origin_y == 0.6 and fixed_y >= 0.6:  # y座標が0.6より大きい位置
            #                 self.ax.scatter(fixed_x, fixed_y, color="blue", s=50)

            #     # 起点を結ぶ点を描画
            #     connecting_angles = np.linspace(0, 1, 5)  # 起点間を5点で補間
            #     for t in connecting_angles:
            #         connecting_x = 0.3  # x座標を固定
            #         connecting_y = 0.4 + t * (0.6 - 0.4)  # y座標を線形補間
            #         self.ax.scatter(connecting_x, connecting_y, color="blue", s=50)

            # # 9mラインを描画(directionがright)
            # if direction == "right":
            #     origins = [(1, 0.4), (1, 0.6)]  # 起点の2パターン
            #     distance = 0.45  # 距離
            #     angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)  # 60点を均等に配置

            #     for origin_x, origin_y in origins:
            #         for angle in angles:
            #             fixed_x = origin_x - distance * np.cos(angle)  # x座標を左方向に調整
            #             fixed_y = origin_y + distance * np.sin(angle)

            #             # 条件に基づいて描画
            #             if origin_y == 0.4 and fixed_y <= 0.4:  # y座標が0.4より小さい位置
            #                 self.ax.scatter(fixed_x, fixed_y, color="red", s=50)
            #             elif origin_y == 0.6 and fixed_y >= 0.6:  # y座標が0.6より大きい位置
            #                 self.ax.scatter(fixed_x, fixed_y, color="red", s=50)

            #     # 起点を結ぶ点を描画
            #     connecting_angles = np.linspace(0, 1, 5)  # 起点間を5点で補間
            #     for t in connecting_angles:
            #         connecting_x = 0.55  # x座標を固定
            #         connecting_y = 0.4 + t * (0.6 - 0.4)  # y座標を線形補間
            #         self.ax.scatter(connecting_x, connecting_y, color="red", s=50)

            #     # 6mラインを描画
            #     distance = 0.3  # 距離を0.3に設定
            #     for origin_x, origin_y in origins:
            #         for angle in angles:
            #             fixed_x = origin_x - distance * np.cos(angle)  # x座標を左方向に調整
            #             fixed_y = origin_y + distance * np.sin(angle)

            #             # 条件に基づいて描画
            #             if origin_y == 0.4 and fixed_y <= 0.4:  # y座標が0.4より小さい位置
            #                 self.ax.scatter(fixed_x, fixed_y, color="red", s=50)
            #             elif origin_y == 0.6 and fixed_y >= 0.6:  # y座標が0.6より大きい位置
            #                 self.ax.scatter(fixed_x, fixed_y, color="red", s=50)

            #     # 起点を結ぶ点を描画
            #     connecting_angles = np.linspace(0, 1, 5)  # 起点間を5点で補間
            #     for t in connecting_angles:
            #         connecting_x = 0.7  # x座標を固定
            #         connecting_y = 0.4 + t * (0.6 - 0.4)  # y座標を線形補間
            #         self.ax.scatter(connecting_x, connecting_y, color="red", s=50)

            ###########################9mと6mラインの描画ここまで#############################

            # フォーメーションの検出  
            # コートの方向(left/right)ごとの，フォーメーション別の理想位置のリストを作成しそのどれに一番近いかを判定する
            # フォーメーションの理想位置を定義
            formation_positions = {
                "0-6_right": [(0.8, 0.175), (0.7, 0.3), (0.65, 0.4), (0.65, 0.6), (0.7, 0.7),(0.8, 0.825)],
                "0-6_left": [(0.775, 0.3), (0.4, 0.5), (0.6, 0.5), (0.5, 0.4), (0.5, 0.6)],
                "1-5_right": [(0.8, 0.175), (0.7, 0.3), (0.65, 0.5), (0.5, 0.5), (0.7, 0.7), (0.8, 0.825)],
                "1-5_left": [(0.5, 0.5), (0.4, 0.5), (0.6, 0.5), (0.5, 0.4), (0.5, 0.6)],
            }

            # 現在の選手の位置とフォーメーションの理想位置の距離を計算（平均値バージョン）
            closest_formation = None
            min_avg_distance = float('inf')

            for formation, ideal_positions in formation_positions.items():
                distances = cdist(positions, ideal_positions, metric='euclidean')
                min_dists = np.min(distances, axis=1)  # 各選手にとって最も近い理想位置との距離
                avg_distance = np.mean(min_dists)      # 平均距離に変更

                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    closest_formation = formation

            # 一番近しいフォーメーションの位置を緑の十字形で描画
            if closest_formation:
                for ideal_position in formation_positions[closest_formation]:
                    self.ax.scatter(ideal_position[0], ideal_position[1], color="green", marker="x", s=100, label=closest_formation)
            
            # フォーメーションを画面に表示
            if closest_formation:
                self.ax.set_title(f"Closest Formation: {closest_formation} (Dist: {min_avg_distance:.2f})")
            else:
                self.ax.set_title("No matching formation found")
            
        # タイトルとプロット設定
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(1, 0)
        self.ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()


# 起動
if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryViewerWithFormation(root)
    root.mainloop()

