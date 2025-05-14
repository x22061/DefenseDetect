"""
フォーメーションの自動分類を行うコードです．
CSVファイルから選手の位置情報を読み込み，6人の選手の位置を基にフォーメーションを判別します．
フォーメーションはあらかじめ定義された理想的な座標との距離で比較して決定されます．
9mラインの外側にいる選手の数に基づいてフォーメーションを分類します．
"""

import csv
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from itertools import combinations
from tqdm import tqdm
import time  # Add import for time module


class FormationClassifier:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.attack_formations = self.load_csv()

    def load_csv(self):
        """CSVファイルを読み込み、フレームごとに選手位置をまとめる"""
        frames = defaultdict(list)
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                frame_num = int(row[0])
                x = float(row[3])
                y = float(row[4])
                direction = row[5]
                frames[(frame_num, direction)].append((x, y))
        return frames

##################フォーメーションの判別を行うメソッドここから##################

    def classify_formations(self):
        """フレームごとに9mラインの外側にいる選手の数でフォーメーションを判別"""
        classified_formations = []

        for (frame_num, direction), positions in self.attack_formations.items():
            red_count = 0

            for x, y in positions:
                if direction == "right" and (0.4 < x < 0.55) and (0.2 < y < 0.8):
                    red_count += 1
                elif direction == "left" and (0.45 < x < 0.6) and (0.2 < y < 0.8):
                    red_count += 1

            # フォーメーションを判別
            if red_count == 1:
                closest_formation = "1-5 Formation"
            elif red_count == 0:
                closest_formation = "0-6 Formation"
            elif red_count == 2:
                closest_formation = "2-4 Formation"
            elif red_count == 3:
                closest_formation = "3-3 Formation"
            else:
                closest_formation = "Unknown Formation"

            # 信頼度は仮に1.0とする
            confidence = 1.0
            classified_formations.append((frame_num, direction, closest_formation, confidence))

        return classified_formations
##################フォーメーションの判別を行うメソッドここまで##################

    def get_dominant_formations(self, classified_formations, min_length=50):
        """
        directionごとにフェーズを区切り、100フレーム未満のフェーズは前後と結合できなければ除外、
        フェーズ単位で信頼度に基づいて防御フォーメーションを決定する
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

        # 最後に、フェーズごとに信頼度加重による支配的なフォーメーションを決定
        dominant_formations = []
        for phase in phases:
            frame_nums = [p[0] for p in phase]
            formations = [p[2] for p in phase]
            confidences = [p[3] for p in phase]
            direction = phase[0][1]

            start_frame = frame_nums[0]
            end_frame = frame_nums[-1]
            phase_length = end_frame - start_frame

            if phase_length < min_length:
                # 最後に念のため、まだ短いフェーズがあったら除外する
                continue

            if len(formations) == 0:
                continue

            # 各フォーメーションの信頼度加重平均を計算
            formation_confidence = defaultdict(float)
            formation_count = defaultdict(int)

            for formation, confidence in zip(formations, confidences):
                formation_confidence[formation] += confidence
                formation_count[formation] += 1

            # 信頼度が最も高いフォーメーションを選定
            best_formation = max(formation_confidence, key=lambda f: formation_confidence[f] / formation_count[f])

            dominant_formations.append((start_frame, end_frame, best_formation, direction))

        return dominant_formations

    
##################防御フェーズとフォーメーションを決定するメソッドここまで##################

    def save_dominant_formations(self, dominant_formations, classified_formations, output_file):
        """CSVに保存し、全体の出現数も別ファイルに出力"""
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["開始フレーム", "終了フレーム", "フォーメーション", "方向", "信頼度"])

            for start_frame, end_frame, dominant_formation, direction in dominant_formations:
                # 該当フェーズのフレームから該当フォーメーションを抽出
                relevant = [
                    (form, confidence) for frame_num, dir_, form, confidence in classified_formations
                    if start_frame <= frame_num <= end_frame and dir_ == direction
                ]
                # 信頼度の平均
                dominant_confidences = [conf for form, conf in relevant if form == dominant_formation]
                avg_confidence = sum(dominant_confidences) / len(dominant_confidences) if dominant_confidences else 0
                avg_confidence = round(avg_confidence, 2)

                # 内訳カウント
                counts = Counter([form for form, _ in relevant])
                breakdown_str = ', '.join(f"{k}: {v}" for k, v in counts.items())

                writer.writerow([start_frame, end_frame, dominant_formation, direction, avg_confidence])

        # 全体のフォーメーション出現数も別ファイルに保存
        all_formations = [form for _, _, form, _ in classified_formations]
        formation_counts = Counter(all_formations)
        count_output_file = output_file.replace(".csv", "_counts.csv")

        with open(count_output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["フォーメーション", "出現数"])
            for formation, count in formation_counts.items():
                writer.writerow([formation, count])


# 使用例
if __name__ == "__main__":
    csv_file = "../data/transform/transformed_player_points.csv"
    output_file = "../data/output/formations_output.csv"

    print("Processing...")  # Indicate processing has started
    start_time = time.time()  # Start timing

    classifier = FormationClassifier(csv_file)
    classified_formations = classifier.classify_formations()

    dominant_formations = classifier.get_dominant_formations(classified_formations)
    classifier.save_dominant_formations(dominant_formations, classified_formations, output_file)

    end_time = time.time()  # End timing
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")  # Output processing time



