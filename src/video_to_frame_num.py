import cv2
import tkinter as tk
from tkinter import filedialog

def select_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    if path:
        video_path.set(path)

def calculate_frame():
    path = video_path.get()
    if not path:
        result_label.config(text="動画ファイルを選択してください")
        return

    try:
        minutes = int(min_entry.get())
        seconds = int(sec_entry.get())
    except ValueError:
        result_label.config(text="時間を正しく入力してください（数字）")
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        result_label.config(text="動画を開けませんでした")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_time_sec = minutes * 60 + seconds
    frame_number = int(fps * total_time_sec)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_number >= total_frames:
        result_label.config(text=f"指定時間は動画の長さを超えています（最大 {total_frames-1} フレーム）")
    else:
        result_label.config(text=f"{minutes}分{seconds}秒 → フレーム番号: {frame_number}")

# GUIの作成
root = tk.Tk()
root.title("時間からフレーム番号を計算")

video_path = tk.StringVar()

tk.Button(root, text="動画ファイルを選択", command=select_video).pack(pady=5)
tk.Entry(root, textvariable=video_path, width=50).pack(pady=5)

time_frame = tk.Frame(root)
time_frame.pack()

tk.Label(time_frame, text="分:").pack(side="left")
min_entry = tk.Entry(time_frame, width=5)
min_entry.pack(side="left")

tk.Label(time_frame, text="秒:").pack(side="left")
sec_entry = tk.Entry(time_frame, width=5)
sec_entry.pack(side="left")

tk.Button(root, text="フレームを計算", command=calculate_frame).pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=5)

root.mainloop()
