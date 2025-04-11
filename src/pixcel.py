
# 画像内の任意の4点をクリックして座標を取得するスクリプト

import cv2

# クリックした座標を保存するリスト
points = []

# マウスイベントのコールバック関数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        # クリックした点を画像に表示
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", img)

# 画像読み込み
img = cv2.imread("sample_1-5fom.png")
cv2.imshow("Image", img)

# コールバック関数の設定
cv2.setMouseCallback("Image", mouse_callback)

print("画像をクリックして4点選んでください。順番は左上→右上→右下→左下がオススメです。")
cv2.waitKey(0)
cv2.destroyAllWindows()

# 座標を NumPy 配列に変換
import numpy as np
src_points = np.float32(points)
print("取得した座標（src_points）:")
print(src_points)
