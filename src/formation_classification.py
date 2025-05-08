"""
フォーメーションの自動分類を行うコードです．
CSVファイルから選手の位置情報を読み込み，6人の選手の位置を基にフォーメーションを判別します．
y座標が小さい順に6人を選び，その座標を理想的なフォーメーション座標と比較して，最も平均距離が小さいフォーメーションを選択します．
フォーメーションはあらかじめ定義された理想的な座標と比較して決定されます．
"""

import csv
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np

# 理想的なフォーメーション座標
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

class FormationClassifier:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.attack_formations = self.load_csv()

    def load_csv(self):
        """CSVファイルを読み込み、フレームごとに選手位置をまとめる"""
        frames = defaultdict(list)
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # ← ヘッダーをスキップ！
            for row in reader:
                frame_num = int(row[0])
                x = float(row[3])
                y = float(row[4])
                direction = row[5]
                frames[(frame_num, direction)].append((x, y))
        return frames

    def classify_formations(self):
        """フレームごとに6人まとめてフォーメーション判別（ソートベース）"""
        classified_formations = []
        for (frame_num, direction), positions in self.attack_formations.items():
            if len(positions) < 6:
                continue
            
            # 6人以上でもy座標の小さい順の6人だけを対象にする
            positions = positions[:6]

            # y座標でソート（上から下へ）
            positions_sorted = sorted(positions, key=lambda p: p[1])

            closest_formation = None
            min_total_distance = float('inf')

            for formation, ideal_positions in formation_positions.items():
                # 理想フォーメーションもy座標でソート
                ideal_positions_sorted = sorted(ideal_positions, key=lambda p: p[1])

                # 対応する順番で平均距離を計算
                total_distance = sum(
                    np.linalg.norm(np.array(pos) - np.array(ideal_pos))
                    for pos, ideal_pos in zip(positions_sorted, ideal_positions_sorted)
                ) / len(positions_sorted)

                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    closest_formation = formation

            # 信頼度を計算（距離が小さいほど信頼度が高い）
            confidence = 1 / (1 + min_total_distance)  # 距離が大きいほど信頼度が低くなる

            classified_formations.append((frame_num, direction, closest_formation, confidence))

        return classified_formations

    def get_dominant_formations(self, classified_formations, min_length=50):
        """
        directionごとにフェーズを区切り、100フレーム未満のフェーズは前後と結合できなければ除外、
        フェーズ単位で多数決により支配的なフォーメーションを決定する
        """
        # directionごとにフェーズをまず分割
        phases = []
        current_phase = []
        current_direction = None

        for frame_num, direction, formation, confidence in classified_formations:
            if current_direction is None:
                current_direction = direction
                current_phase.append((frame_num, direction, formation, confidence))
            elif direction == current_direction:
                current_phase.append((frame_num, direction, formation, confidence))
            else:
                phases.append(current_phase)
                current_phase = [(frame_num, direction, formation, confidence)]
                current_direction = direction

        if current_phase:
            phases.append(current_phase)

        # フェーズ結合処理
        def combine_phases(phases):
            combined_phases = []
            i = 0
            while i < len(phases):
                phase = phases[i]
                frame_nums = [p[0] for p in phase]
                start_frame = frame_nums[0]
                end_frame = frame_nums[-1]
                phase_length = end_frame - start_frame

                if phase_length < min_length:
                    merged = False
                    # 前フェーズと結合
                    if i > 0 and combined_phases and combined_phases[-1][-1][1] == phase[0][1]:  # directionが同じ
                        combined_phases[-1].extend(phase)
                        merged = True
                    # 次フェーズと結合
                    elif i + 1 < len(phases) and phases[i+1][0][1] == phase[-1][1]:  # directionが同じ
                        phases[i+1] = phase + phases[i+1]
                        merged = True

                    if not merged:
                        # どちらにも結合できなければ、このフェーズは無視（除外）
                        i += 1
                        continue
                else:
                    combined_phases.append(phase)
                i += 1

            return combined_phases

        # 再帰的に結合
        phases = combine_phases(phases)

        # 最後に、フェーズごとにフォーメーションを多数決
        dominant_formations = []
        for phase in phases:
            frame_nums = [p[0] for p in phase]
            formations = [p[2] for p in phase]
            direction = phase[0][1]

            start_frame = frame_nums[0]
            end_frame = frame_nums[-1]
            phase_length = end_frame - start_frame

            if phase_length < min_length:
                # 最後に念のため、まだ短いフェーズがあったら除外する
                continue

            if len(formations) == 0:
                continue

            most_common_formation = Counter(formations).most_common(1)[0][0]
            dominant_formations.append((start_frame, end_frame, most_common_formation, direction))

        return dominant_formations



    def save_dominant_formations(self, dominant_formations, classified_formations, output_file):
        """CSVに保存"""
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["開始フレーム", "終了フレーム", "フォーメーション", "方向", "信頼度"])
            
            for start_frame, end_frame, formation, direction in dominant_formations:
                # フェーズ内の信頼度を抽出
                confidences = [
                    confidence for frame_num, dir_, form, confidence in classified_formations
                    if start_frame <= frame_num <= end_frame and dir_ == direction and form == formation
                ]
                # 平均信頼度を計算
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                avg_confidence = round(avg_confidence, 2)  # 信頼度を少数第2位までに丸める
                writer.writerow([start_frame, end_frame, formation, direction, avg_confidence])

# 使用例
if __name__ == "__main__":
    csv_file = "../data/transform/transformed_player_points.csv"
    output_file = "formations_output_tmp3.csv"

    classifier = FormationClassifier(csv_file)
    classified_formations = classifier.classify_formations()
    dominant_formations = classifier.get_dominant_formations(classified_formations)
    classifier.save_dominant_formations(dominant_formations, classified_formations, output_file)
