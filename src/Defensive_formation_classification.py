"""
攻撃方向に応じた防御フォーメーション(0-6,1-5)を自動判定するコード
テンプレの座標を使用し，一回の攻撃フェーズ中の選手との距離の平均値が一番近いフォーメーションを自動判定します。
"""
# -*- coding: utf-8 -*-
import pandas as pd
import csv
import collections
from scipy.spatial.distance import cdist
import numpy as np

class FormationClassifier:
    def __init__(self, input_csv_path, output_csv_path):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.data = None
        self.direction_change_frames = []
        self.min_frame = None
        self.max_frame = None
        self.attack_formations = []

    def load_csv(self):
        """CSVファイルを読み込む"""
        try:
            self.data = pd.read_csv(self.input_csv_path)
            print(f"CSVファイルを読み込みました: {self.input_csv_path}")

            self.min_frame = self.data["frame_num"].min()
            self.max_frame = self.data["frame_num"].max()
            self.compute_direction_changes()
        except Exception as e:
            print(f"CSV読み込み中にエラーが発生しました: {e}")

    def compute_direction_changes(self):
        """攻撃方向が変わったフレームを検出する"""
        current_direction = None

        for frame, direction in zip(self.data["frame_num"], self.data["direction"]):
            if current_direction is None:
                current_direction = direction
            elif current_direction != direction:
                self.direction_change_frames.append(frame)
                current_direction = direction

    def classify_formations(self):
        """攻撃方向に基づいてフォーメーションを分類"""
        formation_positions = {
            "0-6_right": [(0.8, 0.175), (0.7, 0.3), (0.65, 0.4), (0.65, 0.6), (0.7, 0.7), (0.8, 0.825)],
            "0-6_left": [(0.2, 0.175), (0.3, 0.3), (0.35, 0.4), (0.35, 0.6), (0.3, 0.7), (0.2, 0.825)],
            "1-5_right": [(0.8, 0.175), (0.7, 0.3), (0.65, 0.5), (0.5, 0.5), (0.7, 0.7), (0.8, 0.825)],
            "1-5_left": [(0.2, 0.175), (0.3, 0.3), (0.35, 0.5), (0.5, 0.5), (0.3, 0.7), (0.2, 0.825)],
        }

        for frame_num, group in self.data.groupby("frame_num"):
            positions = list(zip(group["x"], group["y"]))
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
                direction = group["direction"].iloc[0]
                self.attack_formations.append((frame_num, direction, closest_formation))

    def get_dominant_formations(self):
        """一回の攻撃フェーズ中に最も多く検出されたフォーメーションを出力"""
        dominant_formations = []

        # フェーズを結合するための一時変数
        merged_start_frame = None
        merged_end_frame = None
        merged_formations = []
        previous_direction = None

        for i in range(len(self.direction_change_frames) + 1):
            start_frame = self.direction_change_frames[i - 1] if i > 0 else self.min_frame
            end_frame = self.direction_change_frames[i] if i < len(self.direction_change_frames) else self.max_frame

            # 現在のフェーズの方向を取得
            current_direction = None
            for frame_num, direction, _ in self.attack_formations:
                if start_frame <= frame_num < end_frame:
                    current_direction = direction
                    break

            # フレーム数が50未満の場合、前後のフェーズと結合
            if end_frame - start_frame < 50:
                continue

            # フェーズが交互になっていない場合、結合
            if previous_direction is not None and current_direction == previous_direction:
                merged_end_frame = end_frame
                merged_formations.extend([formation for frame_num, direction, formation in self.attack_formations
                                          if start_frame <= frame_num < end_frame])
                continue

            # 新しいフェーズを開始
            if merged_start_frame is not None:
                # 結合されたフェーズの最頻フォーメーションを計算
                most_common_formation = collections.Counter(merged_formations).most_common(1)[0][0]
                dominant_formations.append((merged_start_frame, merged_end_frame, most_common_formation))

            merged_start_frame = start_frame
            merged_end_frame = end_frame
            merged_formations = [formation for frame_num, direction, formation in self.attack_formations
                                 if start_frame <= frame_num < end_frame]
            previous_direction = current_direction

        # 最後のフェーズを追加
        if merged_start_frame is not None:
            most_common_formation = collections.Counter(merged_formations).most_common(1)[0][0]
            dominant_formations.append((merged_start_frame, merged_end_frame, most_common_formation))

        return dominant_formations

    def save_dominant_formations_to_csv(self):
        """一回の攻撃フェーズ中の最も多く検出されたフォーメーションをCSVに保存"""
        dominant_formations = self.get_dominant_formations()

        try:
            with open(self.output_csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["開始フレーム", "終了フレーム", "フォーメーション"])

                for start_frame, end_frame, formation in dominant_formations:
                    writer.writerow([start_frame, end_frame, formation])

            print(f"支配的なフォーメーションデータを{self.output_csv_path}に保存しました")
        except Exception as e:
            print(f"CSV保存中にエラーが発生しました: {e}")

# Example usage
if __name__ == "__main__":
    input_csv_path = "../data/transform/transformed_player_points.csv"  # 読み込むCSVファイルのパス
    output_csv_path = "formations_output.csv"  # 保存するCSVファイルのパス

    classifier = FormationClassifier(input_csv_path, output_csv_path)
    classifier.load_csv()
    classifier.classify_formations()
    classifier.save_dominant_formations_to_csv()

