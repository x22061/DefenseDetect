"""
フォーメーションの自動分類を行うコードです．
CSVファイルから選手の位置情報を読み込み，6人の選手の位置を基にフォーメーションを判別します．
フォーメーションはあらかじめ定義された理想的な座標との距離で比較して決定されます．
全選手から6人を選び、最も一致度の高い組を使ってフォーメーション分類を行います．
信頼度で求めます
"""

import csv
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from itertools import combinations
from tqdm import tqdm
import time  # Add import for time module

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
        """全選手から6人を選び、最も一致度の高い組を使ってフォーメーション分類"""
        classified_formations = []

        all_items = list(self.attack_formations.items())

        for (frame_num, direction), positions in tqdm(all_items, desc="Processing frames"):
            if len(positions) < 6:
                continue

            best_formation = None
            best_confidence = 0
            min_total_distance = float('inf')

            # すべての6人組み合わせを評価
            for six_positions in combinations(positions, 6):
                six_positions = np.array(six_positions)

                for formation, ideal_positions in formation_positions.items():
                    if len(ideal_positions) != 6:
                        continue
                    ideal_positions_arr = np.array(ideal_positions)

                    # 距離行列を計算
                    cost_matrix = cdist(six_positions, ideal_positions_arr)
                    # ハンガリアン法で最適対応を取得
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    total_distance = cost_matrix[row_ind, col_ind].sum() / 6

                    if total_distance < min_total_distance:
                        min_total_distance = total_distance
                        best_formation = formation
                        best_confidence = 1 / (1 + total_distance)

            if best_formation:
                classified_formations.append((frame_num, direction, best_formation, best_confidence))

        return classified_formations

    
##################フォーメーションの判別を行うメソッドここまで##################

##################防御フェーズとフォーメーションを決定するメソッドここから##################

    # def get_dominant_formations(self, classified_formations, min_length=50):
    #     """
    #     directionごとにフェーズを区切り、100フレーム未満のフェーズは前後と結合できなければ除外、
    #     フェーズ単位で多数決により支配的なフォーメーションを決定する
    #     """
    #     # directionごとにフェーズをまず分割
    #     phases = []
    #     current_phase = []
    #     current_direction = None

    #     for frame_num, direction, formation, confidence in classified_formations:
    #         if current_direction is None:
    #             current_direction = direction
    #             current_phase.append((frame_num, direction, formation, confidence))
    #         elif direction == current_direction:
    #             current_phase.append((frame_num, direction, formation, confidence))
    #         else:
    #             phases.append(current_phase)
    #             current_phase = [(frame_num, direction, formation, confidence)]
    #             current_direction = direction

    #     if current_phase:
    #         phases.append(current_phase)

    #     # フェーズ結合処理
    #     def combine_phases(phases):
    #         combined_phases = []
    #         i = 0
    #         while i < len(phases):
    #             phase = phases[i]
    #             frame_nums = [p[0] for p in phase]
    #             start_frame = frame_nums[0]
    #             end_frame = frame_nums[-1]
    #             phase_length = end_frame - start_frame

    #             if phase_length < min_length:
    #                 merged = False
    #                 # 前フェーズと結合
    #                 if i > 0 and combined_phases and combined_phases[-1][-1][1] == phase[0][1]:  # directionが同じ
    #                     combined_phases[-1].extend(phase)
    #                     merged = True
    #                 # 次フェーズと結合
    #                 elif i + 1 < len(phases) and phases[i+1][0][1] == phase[-1][1]:  # directionが同じ
    #                     phases[i+1] = phase + phases[i+1]
    #                     merged = True

    #                 if not merged:
    #                     # どちらにも結合できなければ、このフェーズは無視（除外）
    #                     i += 1
    #                     continue
    #             else:
    #                 combined_phases.append(phase)
    #             i += 1

    #         return combined_phases

    #     # 再帰的に結合
    #     phases = combine_phases(phases)

    #     # 最後に、フェーズごとにフォーメーションを多数決
    #     dominant_formations = []
    #     for phase in phases:
    #         frame_nums = [p[0] for p in phase]
    #         formations = [p[2] for p in phase]
    #         direction = phase[0][1]

    #         start_frame = frame_nums[0]
    #         end_frame = frame_nums[-1]
    #         phase_length = end_frame - start_frame

    #         if phase_length < min_length:
    #             # 最後に念のため、まだ短いフェーズがあったら除外する
    #             continue

    #         if len(formations) == 0:
    #             continue

    #         most_common_formation = Counter(formations).most_common(1)[0][0]
    #         dominant_formations.append((start_frame, end_frame, most_common_formation, direction))

    #     return dominant_formations

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

    # def save_dominant_formations(self, dominant_formations, classified_formations, output_file):
    #     """CSVに保存"""
    #     with open(output_file, 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["開始フレーム", "終了フレーム", "フォーメーション", "方向", "信頼度"])
            
    #         for start_frame, end_frame, formation, direction in dominant_formations:
    #             # フェーズ内の信頼度を抽出
    #             confidences = [
    #                 confidence for frame_num, dir_, form, confidence in classified_formations
    #                 if start_frame <= frame_num <= end_frame and dir_ == direction and form == formation
    #             ]
    #             # 平均信頼度を計算
    #             avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    #             avg_confidence = round(avg_confidence, 2)  # 信頼度を少数第2位までに丸める
    #             writer.writerow([start_frame, end_frame, formation, direction, avg_confidence])


    def save_dominant_formations(self, dominant_formations, classified_formations, output_file):
        """CSVに保存し、全体の出現数も別ファイルに出力"""
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["開始フレーム", "終了フレーム", "フォーメーション", "方向", "信頼度", "フォーメーション内訳"])

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

                writer.writerow([start_frame, end_frame, dominant_formation, direction, avg_confidence, breakdown_str])

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
    output_file = "formations_output_ver02_conf.csv"

    print("Processing...")  # Indicate processing has started
    start_time = time.time()  # Start timing

    classifier = FormationClassifier(csv_file)
    classified_formations = classifier.classify_formations()

    dominant_formations = classifier.get_dominant_formations(classified_formations)
    classifier.save_dominant_formations(dominant_formations, classified_formations, output_file)

    end_time = time.time()  # End timing
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")  # Output processing time

    

