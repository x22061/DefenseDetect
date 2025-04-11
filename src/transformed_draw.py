import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.cm import get_cmap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class TrajectoryViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Trajectory Viewer")
        self.file_path = ""

        # UI Elements
        tk.Button(root, text="CSVを選択", command=self.load_csv).pack(pady=5)

        self.start_entry = None
        self.end_scale = None
        self.figure = None
        self.canvas = None

    def load_csv(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            self.df = pd.read_csv(self.file_path)
            print("CSV読み込み完了")

            self.min_frame = self.df["frame_num"].min()
            self.max_frame = self.df["frame_num"].max()

            # 既存のUIを破棄
            if self.start_entry:
                self.start_entry.destroy()
            if self.end_scale:
                self.end_scale.destroy()

            # 開始フレーム（数値入力）
            tk.Label(self.root, text="開始フレーム").pack()
            self.start_entry = tk.Entry(self.root)
            self.start_entry.insert(0, str(self.min_frame))
            self.start_entry.pack()
            self.start_entry.bind("<Return>", self.on_start_change)

            # 終了フレーム（スライダー）
            self.end_scale = tk.Scale(self.root, from_=self.min_frame + 1, to=self.min_frame + 100,
                                      orient="horizontal", label="終了フレーム", command=self.update_plot)
            self.end_scale.set(min(self.min_frame + 100, self.max_frame))
            self.end_scale.pack(fill="x", padx=10)

            self.update_plot(None)

    def on_start_change(self, event=None):
        try:
            start_frame = int(self.start_entry.get())
        except ValueError:
            print("開始フレームは整数で入力してください")
            return

        if start_frame < self.min_frame or start_frame >= self.max_frame:
            print("開始フレームが範囲外です")
            return

        max_end = min(start_frame + 100, self.max_frame)
        self.end_scale.config(from_=start_frame + 1, to=max_end)

        if self.end_scale.get() > max_end:
            self.end_scale.set(max_end)

        self.update_plot()

    def update_plot(self, event=None):
        if not hasattr(self, 'df'):
            return

        try:
            start_frame = int(self.start_entry.get())
        except ValueError:
            return

        end_frame = self.end_scale.get()
        if start_frame >= end_frame:
            return

        df_filtered = self.df[(self.df["frame_num"] >= start_frame) & (self.df["frame_num"] <= end_frame)]

        if self.figure is None:
            self.figure = plt.Figure(figsize=(10, 6))
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.ax.clear()

        cmap = get_cmap("tab10")
        id_list = df_filtered["id"].unique()
        color_map = {id_: cmap(i % 10) for i, id_ in enumerate(id_list)}

        for id_ in id_list:
            person_data = df_filtered[df_filtered["id"] == id_]

            directions = person_data["direction"].unique()
            team_colors = person_data["team_color"].unique()

            if len(directions) != 1 or len(team_colors) != 1:
                continue

            direction = directions[0]
            team_color = team_colors[0]

            if (direction == "left" and team_color != "white") or (direction == "right" and team_color != "red"):
                continue

            self.ax.plot(person_data["x"], person_data["y"],
                         label=f"ID {id_} ({team_color})", color=color_map[id_])

        self.ax.set_title(f"移動軌跡（{start_frame} ～ {end_frame} フレーム）")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(1, 0)
        self.ax.grid(True)
        self.ax.legend(loc="upper left")
        self.figure.tight_layout()
        self.canvas.draw()


# 起動
if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryViewer(root)
    root.mainloop()
