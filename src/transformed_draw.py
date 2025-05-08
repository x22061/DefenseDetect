"""
選手の移動軌跡を可視化し、攻撃方向に応じた防御フォーメーション(0-6,1-5)を自動判定するアプリケーションです。
テンプレの座標を使用し，選手との距離の平均値が一番近いフォーメーションを自動判定します。
"""
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

        # 開始フレームを指定する枠を追加
        tk.Label(self.root, text="開始フレーム").pack()
        self.start_frame_entry = tk.Entry(self.root)
        self.start_frame_entry.pack()

        # 現在のフレームを指定する枠を追加
        tk.Label(self.root, text="現在のフレーム").pack()
        self.current_frame_entry = tk.Entry(self.root)
        self.current_frame_entry.pack()

        #描画を開始するボタンを追加
        start_button = tk.Button(self.root, text="開始", command=self.update_plot)
        start_button.pack(pady=5)

        self.attack_formations = []  # Store formations for each attack phase
        self.current_direction = None  # Track the current direction
        self.phase_formations = []  # Track formations within the current phase
        self.direction_change_frames = []  # Store frames where direction changes occur

    #前のフレームに戻る
    def prev_frame(self):
        if hasattr(self, 'current_frame_entry'):
            try:
                current_frame = int(self.current_frame_entry.get())
                if current_frame > self.min_frame:
                    self.current_frame_entry.delete(0, tk.END)
                    self.current_frame_entry.insert(0, str(current_frame - 1))
                    self.update_plot()
            except ValueError:
                print("フレーム番号が不正です")
    #次のフレームに進む
    def next_frame(self):
        if hasattr(self, 'current_frame_entry'):
            try:
                current_frame = int(self.current_frame_entry.get())
                if current_frame < self.max_frame:
                    self.current_frame_entry.delete(0, tk.END)
                    self.current_frame_entry.insert(0, str(current_frame + 1))
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

            # Update start and current frame UI
            self.start_frame_entry.delete(0, tk.END)
            self.start_frame_entry.insert(0, str(self.min_frame))

            self.current_frame_entry.delete(0, tk.END)
            self.current_frame_entry.insert(0, str(self.min_frame))

            self.load_court_images()
            self.compute_direction_changes()
            self.update_plot()

    # --- 攻撃方向関連 ---
    def compute_direction_changes(self):
        """攻撃方向が変わったフレームを検出する"""
        if not hasattr(self, 'df'):
            return

        self.direction_change_frames = []
        current_direction = None

        for frame, direction in zip(self.df["frame_num"], self.df["direction"]):
            if current_direction is None:
                current_direction = direction
            elif current_direction != direction:
                # 攻撃方向が変化したら記録
                self.direction_change_frames.append(frame)
                current_direction = direction

    def load_court_images(self):
        try:
            self.right_court_image = mpimg.imread(self.right_court_image_path)
            self.left_court_image = mpimg.imread(self.left_court_image_path)
            print("コート画像読み込み完了")
        except Exception as e:
            print(f"コート画像の読み込みに失敗しました: {e}")

    # --- フォーメーション検出・描画関連 ---
    def detect_formation(self):
        if not hasattr(self, 'df'):
            print("データが読み込まれていません")
            return

        try:
            current_frame = int(self.current_frame_entry.get())
        except ValueError:
            print("フレーム番号が不正です")
            return

        # フレーム範囲のデータを取得
        df_filtered = self.df[(self.df["frame_num"] >= self.min_frame) & (self.df["frame_num"] <= current_frame)]
        if df_filtered.empty:
            print("指定されたフレーム範囲にデータがありません")
            return
    
    def update_plot(self, event=None):
        """現在のフレーム範囲に応じてコートと選手軌跡を描画"""
        if not hasattr(self, 'df'):
            return

        try:
            start_frame = int(self.start_frame_entry.get())
            current_frame = int(self.current_frame_entry.get())
        except ValueError:
            print("フレーム番号が不正です")
            return

        if start_frame > current_frame:
            print("開始フレームは現在のフレーム以下でなければなりません")
            return

        # フレーム範囲のデータを取得 (指定した開始フレームから現在のフレームまで)
        df_filtered = self.df[(self.df["frame_num"] >= start_frame) & (self.df["frame_num"] <= current_frame)]
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

            # Detect transition between "right" and "left"
            if self.current_direction is None:
                self.current_direction = direction
            elif self.current_direction != direction:
                # Transition detected, finalize the current phase
                if self.phase_formations:
                    most_common_formation = max(set(self.phase_formations), key=self.phase_formations.count)
                    self.attack_formations.append((self.current_direction, most_common_formation))
                    print(f"攻撃方向: {self.current_direction}, フォーメーション: {most_common_formation}")
                self.current_direction = direction
                self.phase_formations = []  # Reset for the new phase

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

            # フィルタリング条件を適用
            if direction == "right":
                df_filtered = df_filtered[
                    ~(
                        ((df_filtered["x"] - 1.0) ** 2 + (df_filtered["y"] - 0.4) ** 2 <= 0.4 ** 2) |
                        ((df_filtered["x"] - 1.0) ** 2 + (df_filtered["y"] - 0.6) ** 2 <= 0.4 ** 2) |
                        ((df_filtered["x"] >= 0.7) & (df_filtered["x"] <= 1.0) & (df_filtered["y"] >= 0.4) & (df_filtered["y"] <= 0.6)) |
                        (df_filtered["x"] < 0.2)  # 修正: 論理演算子を追加
                    )
                ]
            elif direction == "left":
                df_filtered = df_filtered[
                    ~((df_filtered["x"] <= 0.4) & (df_filtered["y"] <= 0.4) & (df_filtered["y"] >= 0.6)) &
                    ~((df_filtered["x"] <= 0.3) & (df_filtered["y"] <= 0.4) & (df_filtered["y"] >= 0.6)) &
                    ~(df_filtered["x"] > 0.8)
                ]

            # フレームごとに選手の位置をプロット
            for frame in range(start_frame, current_frame + 1):
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

                    # フレームが離れるほど透明度を上げる
                    alpha = max(0.1, 1.0 - (current_frame - frame) / (current_frame - start_frame + 1))
                    color = "black" if frame == current_frame else "gray"
                    
                    # 選手の位置をプロット (コメント化)
                    self.ax.scatter(x, y, color=color, alpha=alpha, s=50)

                    # Collect positions for formation detection
                    positions.append((x, y))

            # フォーメーション検出と描画
            formation_positions = {
                "0-6_right": [(0.8, 0.175), (0.7, 0.3), (0.65, 0.4), (0.65, 0.6), (0.7, 0.7),(0.8, 0.825)],
                "0-6_left": [(0.2, 0.175), (0.3, 0.3), (0.35, 0.4), (0.35, 0.6), (0.3, 0.7),(0.2,0.825)],
                "1-5_right": [(0.8, 0.175), (0.7, 0.3), (0.65, 0.5), (0.5, 0.5), (0.7, 0.7), (0.8, 0.825)],
                "1-5_left": [(0.2, 0.175), (0.3, 0.3), (0.35, 0.5), (0.5, 0.5), (0.3, 0.7), (0.2, 0.825)],
                "1-2-3_right": [(0.75, 0.225), (0.575, 0.35), (0.675, 0.5), (0.5, 0.5), (0.575, 0.65), (0.75, 0.775)],
                "1-2-3_left": [(0.25, 0.225), (0.425, 0.35), (0.325, 0.5), (0.5, 0.5), (0.425, 0.65), (0.25, 0.775)],
                "3-3_right": [(0.5, 0.3), (0.7, 0.3), (0.675, 0.5), (0.5, 0.5), (0.5, 0.7), (0.7, 0.7)],
                "3-3_left": [(0.5, 0.3), (0.3, 0.3), (0.325, 0.5), (0.5, 0.5), (0.5, 0.7), (0.3, 0.7)],
                "2-4_right": [(0.7,0.3), (0.5,0.4), (0.675,0.4), (0.5,0.6), (0.675,0.6), (0.7,0.7)],
                "2-4_left": [(0.3,0.3), (0.5,0.4), (0.325,0.4), (0.5,0.6), (0.325,0.6), (0.3,0.7)],
            }

            closest_formation = None
            min_avg_distance = float('inf')

            for formation, ideal_positions in formation_positions.items():
                distances = cdist(positions, ideal_positions, metric='euclidean')
                min_dists = np.min(distances, axis=1)
                avg_distance = np.mean(min_dists)

                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    closest_formation = formation

            if closest_formation:
                self.phase_formations.append(closest_formation)  # Track formation for the current phase
                
                # フォーメーションの理想位置を緑の丸で描画
                for ideal_position in formation_positions[closest_formation]:
                    self.ax.scatter(ideal_position[0], ideal_position[1], color="green", s=100, alpha=0.7)

            if closest_formation:
                self.ax.set_title(f"Formation: {closest_formation} (Dist: {min_avg_distance:.2f})")
            else:
                self.ax.set_title("No matching formation found")
            
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(1, 0)
        self.ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()

    def print_attack_formations(self):
        """Print all detected attack formations."""
        for direction, formation in self.attack_formations:
            print(f"攻撃方向: {direction}, フォーメーション: {formation}")

    def print_total_frames_between_changes(self):
        """Calculate and print the total frames between direction changes."""
        if not hasattr(self, 'direction_change_frames') or not self.direction_change_frames:
            print("方向転換データがありません")
            return

        total_frames = 0
        for i in range(1, len(self.direction_change_frames)):
            total_frames += self.direction_change_frames[i] - self.direction_change_frames[i - 1]

        print(f"方向転換間のフレーム合計: {total_frames}")


# 起動
if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryViewerWithFormation(root)
    root.mainloop()

