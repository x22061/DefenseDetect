# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
import numpy as np  # ← 追加

class TrajectoryViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Trajectory Viewer")
        self.file_path = ""
        self.right_court_image_path = "../img/right_court.png"
        self.left_court_image_path = "../img/left_court.png"
        self.right_court_image = None
        self.left_court_image = None

        tk.Button(root, text="CSVを選択", command=self.load_csv).pack(pady=5)

        self.start_entry = None
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

    def prev_frame(self):
        if hasattr(self, 'start_entry'):
            try:
                current_frame = int(self.start_entry.get())
                if current_frame > self.min_frame:
                    self.start_entry.delete(0, tk.END)
                    self.start_entry.insert(0, str(current_frame - 1))
                    self.update_plot()
            except ValueError:
                print("フレーム番号が不正です")

    def next_frame(self):
        if hasattr(self, 'start_entry'):
            try:
                current_frame = int(self.start_entry.get())
                if current_frame < self.max_frame:
                    self.start_entry.delete(0, tk.END)
                    self.start_entry.insert(0, str(current_frame + 1))
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

            if self.start_entry:
                self.start_entry.destroy()

            tk.Label(self.root, text="フレーム番号").pack()
            self.start_entry = tk.Entry(self.root)
            self.start_entry.insert(0, str(self.min_frame))
            self.start_entry.pack()
            self.start_entry.bind("<Return>", self.update_plot)

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
        if not hasattr(self, 'df'):
            return

        try:
            frame_num = int(self.start_entry.get())
        except ValueError:
            print("フレーム番号が不正です")
            return

        #フレームごとに選手の位置を取得
        df_filtered = self.df[self.df["frame_num"] == frame_num]
        if df_filtered.empty:
            print("指定フレームにデータがありません")
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

            positions = []

            # Use constants for offsets
            x_offset = self.RED_X_OFFSET
            y_offset = self.RED_Y_OFFSET
            white_x_offset = self.WHITE_X_OFFSET
            white_y_offset = self.WHITE_Y_OFFSET

            #選手の位置とチームを取得
            for _, person_data in df_filtered.iterrows():
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
                
                # 位置を保存し、プロット
                positions.append((x, y))
                self.ax.scatter(x, y, color="black", s=50)

            # プロットしている選手の量を表示
            print(f"プロットしている選手の数: {len(positions)}")

        ####################################コートラインの定義と描画ここから#########################################

            # 9mラインを描画(directionがleft)
            if direction == "left":
                origins = [(0, 0.4), (0, 0.6)]  # 起点の2パターン
                distance = 0.45  # 距離
                angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)  # 60点を均等に配置

                for origin_x, origin_y in origins:
                    for angle in angles:
                        fixed_x = origin_x + distance * np.cos(angle)
                        fixed_y = origin_y + distance * np.sin(angle)

                        # 条件に基づいて描画
                        if origin_y == 0.4 and fixed_y <= 0.4:  # y座標が0.4より小さい位置
                            self.ax.scatter(fixed_x, fixed_y, color="blue", s=50)
                        elif origin_y == 0.6 and fixed_y >= 0.6:  # y座標が0.6より大きい位置
                            self.ax.scatter(fixed_x, fixed_y, color="blue", s=50)

                # 起点を結ぶ点を描画
                connecting_angles = np.linspace(0, 1, 5)  # 起点間を5点で補間
                for t in connecting_angles:
                    connecting_x = 0.45  # x座標を固定
                    connecting_y = 0.4 + t * (0.6 - 0.4)  # y座標を線形補間
                    self.ax.scatter(connecting_x, connecting_y, color="blue", s=50)

                # 6mラインを描画
                distance = 0.3  # 距離を0.3に設定
                for origin_x, origin_y in origins:
                    for angle in angles:
                        fixed_x = origin_x + distance * np.cos(angle)
                        fixed_y = origin_y + distance * np.sin(angle)

                        # 条件に基づいて描画
                        if origin_y == 0.4 and fixed_y <= 0.4:  # y座標が0.4より小さい位置
                            self.ax.scatter(fixed_x, fixed_y, color="blue", s=50)
                        elif origin_y == 0.6 and fixed_y >= 0.6:  # y座標が0.6より大きい位置
                            self.ax.scatter(fixed_x, fixed_y, color="blue", s=50)

                # 起点を結ぶ点を描画
                connecting_angles = np.linspace(0, 1, 5)  # 起点間を5点で補間
                for t in connecting_angles:
                    connecting_x = 0.3  # x座標を固定
                    connecting_y = 0.4 + t * (0.6 - 0.4)  # y座標を線形補間
                    self.ax.scatter(connecting_x, connecting_y, color="blue", s=50)


            # 9mラインを描画(directionがright)
            if direction == "right":
                origins = [(1, 0.4), (1, 0.6)]  # 起点の2パターン
                distance = 0.45  # 距離
                angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)  # 60点を均等に配置

                for origin_x, origin_y in origins:
                    for angle in angles:
                        fixed_x = origin_x - distance * np.cos(angle)  # x座標を左方向に調整
                        fixed_y = origin_y + distance * np.sin(angle)

                        # 条件に基づいて描画
                        if origin_y == 0.4 and fixed_y <= 0.4:  # y座標が0.4より小さい位置
                            self.ax.scatter(fixed_x, fixed_y, color="red", s=50)
                        elif origin_y == 0.6 and fixed_y >= 0.6:  # y座標が0.6より大きい位置
                            self.ax.scatter(fixed_x, fixed_y, color="red", s=50)

                # 起点を結ぶ点を描画
                connecting_angles = np.linspace(0, 1, 5)  # 起点間を5点で補間
                for t in connecting_angles:
                    connecting_x = 0.55  # x座標を固定
                    connecting_y = 0.4 + t * (0.6 - 0.4)  # y座標を線形補間
                    self.ax.scatter(connecting_x, connecting_y, color="red", s=50)

                # 6mラインを描画
                distance = 0.3  # 距離を0.3に設定
                for origin_x, origin_y in origins:
                    for angle in angles:
                        fixed_x = origin_x - distance * np.cos(angle)  # x座標を左方向に調整
                        fixed_y = origin_y + distance * np.sin(angle)

                        # 条件に基づいて描画
                        if origin_y == 0.4 and fixed_y <= 0.4:  # y座標が0.4より小さい位置
                            self.ax.scatter(fixed_x, fixed_y, color="red", s=50)
                        elif origin_y == 0.6 and fixed_y >= 0.6:  # y座標が0.6より大きい位置
                            self.ax.scatter(fixed_x, fixed_y, color="red", s=50)

                # 起点を結ぶ点を描画
                connecting_angles = np.linspace(0, 1, 5)  # 起点間を5点で補間
                for t in connecting_angles:
                    connecting_x = 0.7  # x座標を固定
                    connecting_y = 0.4 + t * (0.6 - 0.4)  # y座標を線形補間
                    self.ax.scatter(connecting_x, connecting_y, color="red", s=50)

        ####################################コートラインの定義と描画ここまで#########################################

            # 9mラインよりもコート中央にいる選手がいないかを判定
            inside_9m = []
            for x, y in positions:
                if direction == "left":
                    in_range = (0.425 > x)
                elif direction == "right":
                    in_range = (0.575 < x)
                else:
                    in_range = False

                inside_9m.append(in_range)

            # 判定結果を出力し、UIに表示
            if all(inside_9m):
                print("全ての選手が9mラインの内側にいます。")
                self.ax.text(0.5, 0.1, "0-6formation", color="green", fontsize=16, ha="center", transform=self.ax.transAxes)
                self.ax.set_title("0-6formation", fontsize=16, color="green")
            else:
                print("9mラインの外にいる選手がいます。")

        # タイトルとプロット設定
        self.ax.set_title(f"transform:frame {frame_num}")
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
    app = TrajectoryViewer(root)
    root.mainloop()
