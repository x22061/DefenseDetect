"""
選手の移動軌跡を可視化するGUIアプリケーション
攻撃方向から防御選手のみを描画する
"""
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
import numpy as np  


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
        self.WHITE_X_OFFSET = 0.05
        self.WHITE_Y_OFFSET = 0.0

        # self.RED_X_OFFSET = 0
        # self.RED_Y_OFFSET = 0
        # self.WHITE_X_OFFSET = 0
        # self.WHITE_Y_OFFSET = 0

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

        # フォーメーション表示ラベル
        self.formation_label = tk.Label(self.root, text="フォーメーション: 未推定", font=("Arial", 14), fg="blue")
        self.formation_label.pack(pady=5)

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
            self.update_plot()

    def load_court_images(self):
        try:
            self.right_court_image = mpimg.imread(self.right_court_image_path)
            self.left_court_image = mpimg.imread(self.left_court_image_path)
            print("コート画像読み込み完了")
        except Exception as e:
            print(f"コート画像の読み込みに失敗しました: {e}")

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

            if direction == "right" and self.right_court_image is not None:
                self.ax.imshow(self.right_court_image, extent=[0, 1, 1, 0], aspect='auto')
            elif direction == "left" and self.left_court_image is not None:
                self.ax.imshow(self.left_court_image, extent=[0, 1, 1, 0], aspect='auto')

            # Use constants for offsets
            x_offset = self.RED_X_OFFSET
            y_offset = self.RED_Y_OFFSET
            white_x_offset = self.WHITE_X_OFFSET
            white_y_offset = self.WHITE_Y_OFFSET

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
                    
                    # 防御選手のみを描画
                    if (direction == "left" and team_color != "white") or (direction == "right" and team_color != "red"):
                        continue

                    # フレームが離れるほど透明度を上げる
                    alpha = max(0.1, 1.0 - (current_frame - frame) / (current_frame - start_frame + 1))
                    color = "black" if frame == current_frame else "gray"

                    # 選手の位置をプロット
                    self.ax.scatter(x, y, color=color, alpha=alpha, s=50)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(1, 0)
        self.ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()

        # === フォーメーション推定 ===
        formation = "Unknown"
        frame_data = df_filtered[df_filtered["frame_num"] == current_frame]
        defender_team = "red" if direction == "right" else "white"

        # オフセット適用 + 防御側選手だけ抽出
        defenders = []
        for _, row in frame_data.iterrows():
            if row["team_color"] != defender_team:
                continue
            x, y = row["x"], row["y"]
            if defender_team == "red":
                x += self.RED_X_OFFSET
                y += self.RED_Y_OFFSET
            else:
                x += self.WHITE_X_OFFSET
                y += self.WHITE_Y_OFFSET
            defenders.append((x, y))

        # 防御選手が6人以上いない場合はフォーメーション推定を行わない
        if len(defenders) < 6:
            self.formation_label.config(text="フォーメーション: 推定不可")
            return

        # 中央ゾーンにいる防御選手のカウント
        red_count = 0
        for x, y in defenders:
            if direction == "right" and (0.4 < x < 0.55) and (0.2 < y < 0.8):
                red_count += 1
            elif direction == "left" and (0.45 < x < 0.6) and (0.2 < y < 0.8):
                red_count += 1

        if red_count == 0:
            formation = "0-6"
        elif red_count == 1:
            formation = "1-5"
        elif red_count == 2:
            formation = "2-4"
        elif red_count == 3:
            formation = "3-3"

        # ラベル更新
        self.formation_label.config(text=f"フォーメーション: {formation}")


# 起動
if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryViewerWithFormation(root)
    root.mainloop()

