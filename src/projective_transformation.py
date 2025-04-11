
# 射影変換し、上面図を作成するスクリプト
# 4点は手動で選択します

import cv2
import numpy as np

# sample.png
# [1557, 806],  # 左上
# [3387, 1069],  # 右上
# [2389, 1716], # 右下
# [387, 1036]   # 左下

# サイドビュー画像における、変換したい四角形の4点（左上→右上→右下→左下）
src_points = np.float32([
    [1126, 447],  # 左上
    [2618, 690], # 右上
    [1796, 1213],  # 右下
    [112, 630] # 左下
])

# 出力画像の幅・高さ
width, height = 3854, 2116

# 上記のsrc_pointsに対応する、俯瞰画像上の理想的な矩形の4点
dst_points = np.float32([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
])

# 射影変換行列を計算
M = cv2.getPerspectiveTransform(src_points, dst_points)

# 画像読み込み
img = cv2.imread("sample_1-5fom.png")

# 射影変換を適用
warped = cv2.warpPerspective(img, M, (width, height))

# 結果を表示
cv2.imshow("Top View", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
