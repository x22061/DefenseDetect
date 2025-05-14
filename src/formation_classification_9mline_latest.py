"""
フォーメーションの自動分類を行うコードです．
CSVファイルから選手の位置情報を読み込み，6人の選手の位置を基にフォーメーションを判別します．
フォーメーションはあらかじめ定義された理想的な座標との距離で比較して決定されます．
ゴール側に近い選手との距離を基に，最も近いフォーメーションを選択します．
防御フェーズはゴールの方向が変わってから9mラインの外側にいる選手が戻るまでとします
"""

import csv
from collections import defaultdict, Counter
from typing import List, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from itertools import combinations
from tqdm import tqdm
import time

# 理想的なフォーメーション座標
# formation_positions = {
#     "0-6_right": [(0.8, 0.175), (0.7, 0.3), (0.65, 0.4), (0.65, 0.6), (0.7, 0.7),(0.8, 0.825)],
#     "0-6_left": [(0.2, 0.175), (0.3, 0.3), (0.35, 0.4), (0.35, 0.6), (0.3, 0.7),(0.2,0.825)],
#     "1-5_right": [(0.8, 0.175), (0.7, 0.3), (0.65, 0.5), (0.5, 0.5), (0.7, 0.7), (0.8, 0.825)],
#     "1-5_left": [(0.2, 0.175), (0.3, 0.3), (0.35, 0.5), (0.5, 0.5), (0.3, 0.7), (0.2, 0.825)],
#     "1-2-3_right": [(0.75, 0.225), (0.575, 0.35), (0.675, 0.5), (0.5, 0.5), (0.575, 0.65), (0.75, 0.775)],
#     "1-2-3_left": [(0.25, 0.225), (0.425, 0.35), (0.325, 0.5), (0.5, 0.5), (0.425, 0.65), (0.25, 0.775)],
#     "3-3_right": [(0.5, 0.3), (0.7, 0.3), (0.675, 0.5), (0.5, 0.5), (0.5, 0.7), (0.7, 0.7)],
#     "3-3_left": [(0.5, 0.3), (0.3, 0.3), (0.325, 0.5), (0.5, 0.5), (0.5, 0.7), (0.3, 0.7)],
#     "2-4_right": [(0.7,0.3), (0.5,0.4), (0.675,0.4), (0.5,0.6), (0.675,0.6), (0.7,0.7)],
#     "2-4_left": [(0.3,0.3), (0.5,0.4), (0.325,0.4), (0.5,0.6), (0.325,0.6), (0.3,0.7)],
# }

class FormationClassifier:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.attack_formations = self.load_csv()

    def load_csv(self):
        frames = defaultdict(list)
        # チームごとのオフセット
        self.RED_X_OFFSET = 0.1
        self.RED_Y_OFFSET = -0.1
        self.WHITE_X_OFFSET = 0.05
        self.WHITE_Y_OFFSET = 0.0

        # CSVファイルを読み込み、フレームごとにオフセットを適用した選手位置をまとめる
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            idx = {name: i for i, name in enumerate(header)}
            for row in reader:
                frame_num = int(row[idx['frame_num']])
                x = float(row[idx['x']])
                y = float(row[idx['y']])
                direction = row[idx['direction']]
                team_color = row[idx['team_color']]
                player_id = row[idx['id']]

                # オフセットの適用
                if team_color == "red":
                    x += self.RED_X_OFFSET
                    y += self.RED_Y_OFFSET
                elif team_color == "white":
                    x += self.WHITE_X_OFFSET
                    y += self.WHITE_Y_OFFSET

                frames[(frame_num, direction)].append((x, y, team_color, player_id))
        return frames

    def classify_formations(self):
        """フレームごとに9mラインの外側にいる選手の数でフォーメーションを判別"""
        classified = []
        # 防御選手のみの位置を取得
        for (frame_num, direction), positions in self.attack_formations.items():
            defender_team = 'red' if direction == 'right' else 'white'
            defenders = [(x, y) for x, y, team_color, _ in positions if team_color == defender_team]

            if len(defenders) < 6:
                continue 

            red_count = 0
            for x, y in defenders:
                if direction == "right" and (0.4 < x < 0.55) and (0.2 < y < 0.8):
                    red_count += 1
                elif direction == "left" and (0.45 < x < 0.6) and (0.2 < y < 0.8):
                    red_count += 1

            if red_count == 1:
                formation = "1-5 Formation"
            elif red_count == 0:
                formation = "0-6 Formation"
            elif red_count == 2:
                formation = "2-4 Formation"
            elif red_count == 3:
                formation = "3-3 Formation"
            else:
                formation = "Unknown Formation"

            confidence = 1.0
            classified.append((frame_num, direction, formation, confidence))
        return classified
    

    def detect_defense_phases(self, min_phase_length=50):
        """
        フェーズを検出する。方向が変わるまでの間に、6人以上の選手が9mラインの外側にいる場合をフェーズとする。
        フェーズの長さが min_phase_length より短い場合は、前後のフェーズと結合する。
        """
        all_frames = sorted(self.attack_formations.items())
        phases = []

        current_direction = None
        i = 0
        while i < len(all_frames):
            (frame_num, direction), players = all_frames[i]
            #　方向が変わった瞬間を探す
            if direction != current_direction:
                current_direction = direction
                defender_team = 'red' if direction == 'right' else 'white'

                # フェーズ開始探し
                while i < len(all_frames):
                    (start_frame, dir_check), players = all_frames[i]
                    # フェーズの方向が変わったら終了
                    if dir_check != direction:
                        break
                    # 6人以上の防御選手が9mラインの外側にいるか確認
                    defenders = [(x, y, pid) for x, y, team, pid in players if team == defender_team]
                    if len(defenders) >= 6:
                        break
                    i += 1

                if i >= len(all_frames):
                    break

                start_frame = all_frames[i][0][0]
                defenders = [(x, y, pid) for x, y, team, pid in players if team == defender_team]
                # 9mラインの外側にいる選手を取得
                outer_defenders = [
                    pid for x, y, pid in defenders
                    if ((direction == 'right' and 0.4 < x < 0.55) or
                        (direction == 'left' and 0.45 < x < 0.6)) and (0.2 < y < 0.8)
                ]

                # フェーズ終了探し
                j = i + 1
                end_frame = start_frame
                end_reason = None

                while j < len(all_frames):
                    (next_frame, next_direction), next_players = all_frames[j]
                    # フェーズの方向が変わったら終了
                    if next_direction != direction:
                        end_reason = 'direction_change'
                        break
                    
                    next_defenders = {
                        (pid, x, y) for x, y, team, pid in next_players if team == defender_team
                    }

                    someone_returned = False
                    # 9mラインの外側にいる選手を取得
                    for pid in outer_defenders:
                        for p, x, y in next_defenders:
                            # 9mラインの外側にいる選手が戻ったか確認
                            if p == pid:
                                if direction == 'right' and not (x < 0.55):
                                    someone_returned = True
                                elif direction == 'left' and not (0.45 < x):
                                    someone_returned = True
                    if someone_returned:
                        end_reason = 'outer_return'
                        break

                    end_frame = next_frame
                    j += 1

                # 保存（end_reason 付きで）
                phases.append((start_frame, end_frame, direction, end_reason))
                i = j
            else:
                i += 1

        return self._merge_short_phases(phases, min_phase_length)
    
    def get_dominant_formations_by_defense_phase(self, classified_formations, defense_phases, min_length=0):
        """
        各守備フェーズ内で最多推定フォーメーションを代表とする。
        """
        direction_indexed = defaultdict(list)
        for frame_num, direction, formation, _ in classified_formations:
            direction_indexed[direction].append((frame_num, formation))

        dominant_formations = []
        for start_frame, end_frame, direction in defense_phases:
            # フェーズ内の推定フォーメーションを集計
            relevant_forms = [
                form for frame, form in direction_indexed[direction]
                if start_frame <= frame <= end_frame
            ]
            if not relevant_forms:
                continue
            best_formation = Counter(relevant_forms).most_common(1)[0][0]
            if end_frame - start_frame >= min_length:
                dominant_formations.append((start_frame, end_frame, best_formation, direction))
        return dominant_formations

    

    def save_dominant_formations_by_defense_phase(self, dominant_formations, classified_formations, output_file):
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["開始フレーム", "終了フレーム", "フォーメーション", "方向", "信頼度", "内訳"])
            for start_frame, end_frame, dominant_formation, direction in dominant_formations:
                relevant = [
                    (form, confidence)
                    for frame_num, dir_, form, confidence in classified_formations
                    if start_frame <= frame_num <= end_frame and dir_ == direction
                ]
                dominant_confidences = [conf for form, conf in relevant if form == dominant_formation]
                avg_confidence = round(sum(dominant_confidences) / len(dominant_confidences), 2) if dominant_confidences else 0
                counts = Counter([form for form, _ in relevant])
                breakdown_str = ', '.join(f"{k}: {v}" for k, v in counts.items())
                writer.writerow([start_frame, end_frame, dominant_formation, direction, avg_confidence, breakdown_str])

        all_formations = [form for _, _, form, _ in classified_formations]
        formation_counts = Counter(all_formations)
        count_output_file = output_file.replace(".csv", "_counts.csv")

        with open(count_output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["フォーメーション", "出現数"])
            for formation, count in formation_counts.items():
                writer.writerow([formation, count])

    def _merge_short_phases(self, phases, min_length):
        """
        direction_changeで終わる短いフェーズを、前後と結合。
        outer_returnによる短いフェーズはそのまま保持。
        """
        merged = []
        i = 0
        while i < len(phases):
            start_frame, end_frame, direction, reason = phases[i]
            phase_length = end_frame - start_frame

            if phase_length < min_length and reason == 'direction_change':
                # 次のフェーズとdirectionが同じなら結合
                if i + 1 < len(phases) and phases[i+1][2] == direction:
                    next_end = phases[i+1][1]
                    merged.append((start_frame, next_end, direction))
                    i += 2  # 結合したので次へスキップ
                    continue
                # 前のフェーズとdirectionが同じなら結合
                elif merged and merged[-1][2] == direction:
                    prev_start, prev_end, _ = merged.pop()
                    merged.append((prev_start, end_frame, direction))
                    i += 1
                    continue
                else:
                    # 結合できないが記録（完全除外はしない）
                    merged.append((start_frame, end_frame, direction))
            else:
                # 長いフェーズ or outer_return 終了の短いフェーズ
                merged.append((start_frame, end_frame, direction))
            i += 1

        return merged


if __name__ == "__main__":
    csv_file = "../data/transform/transformed_player_points.csv"
    output_file = "../data/output/formations_output_test.csv"

    print("Processing...")
    start_time = time.time()

    classifier = FormationClassifier(csv_file)
    classified_formations = classifier.classify_formations()
    defense_phases = classifier.detect_defense_phases()
    dominant_formations = classifier.get_dominant_formations_by_defense_phase(classified_formations, defense_phases)
    classifier.save_dominant_formations_by_defense_phase(dominant_formations, classified_formations, output_file)

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")
