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

    # def load_csv(self):
    #     frames = defaultdict(list)
    #     with open(self.csv_file, 'r') as file:
    #         reader = csv.reader(file)
    #         header = next(reader)
    #         idx = {name: i for i, name in enumerate(header)}
    #         for row in reader:
    #             frame_num = int(row[idx['frame_num']])
    #             x = float(row[idx['x']])
    #             y = float(row[idx['y']])
    #             direction = row[idx['direction']]
    #             team_color = row[idx['team_color']]
    #             player_id = row[idx['id']]
    #             frames[(frame_num, direction)].append((x, y, team_color, player_id))
    #     return frames
    
    def load_csv(self):
        frames = defaultdict(list)
        # チームごとのオフセット
        self.RED_X_OFFSET = 0.1
        self.RED_Y_OFFSET = -0.1
        self.WHITE_X_OFFSET = 0.05
        self.WHITE_Y_OFFSET = 0.0

        # self.RED_X_OFFSET = 0
        # self.RED_Y_OFFSET = 0
        # self.WHITE_X_OFFSET = 0
        # self.WHITE_Y_OFFSET = 0

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

    # def classify_formations(self):
    #     classified = []
    #     for (frame_num, direction), positions in self.attack_formations.items():
    #         red_count = 0
    #         defender_team = 'red' if direction == 'right' else 'white'
    #         for x, y, team_color, _ in positions:
    #             if team_color != defender_team:
    #                 continue
    #             if direction == "right" and (0.4 < x < 0.55) and (0.2 < y < 0.8):
    #                 red_count += 1
    #             elif direction == "left" and (0.45 < x < 0.6) and (0.2 < y < 0.8):
    #                 red_count += 1
    #         if red_count == 1:
    #             formation = "1-5 Formation"
    #         elif red_count == 0:
    #             formation = "0-6 Formation"
    #         elif red_count == 2:
    #             formation = "2-4 Formation"
    #         elif red_count == 3:
    #             formation = "3-3 Formation"
    #         else:
    #             formation = "Unknown Formation"
    #         confidence = 1.0
    #         classified.append((frame_num, direction, formation, confidence))
    #     return classified
    
    def classify_formations(self):
        classified = []
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


    # def classify_formations(self):
    #     classified = []

    #     # 理想フォーメーションテンプレート（同じクラス内に持たせるか外部でもOK）
    #     formation_templates = {
    #         "0-6_right": [(0.8, 0.175), (0.7, 0.3), (0.65, 0.4), (0.65, 0.6), (0.7, 0.7), (0.8, 0.825)],
    #         "0-6_left": [(0.2, 0.175), (0.3, 0.3), (0.35, 0.4), (0.35, 0.6), (0.3, 0.7), (0.2, 0.825)],
    #         "1-5_right": [(0.8, 0.175), (0.7, 0.3), (0.65, 0.5), (0.5, 0.5), (0.7, 0.7), (0.8, 0.825)],
    #         "1-5_left": [(0.2, 0.175), (0.3, 0.3), (0.35, 0.5), (0.5, 0.5), (0.3, 0.7), (0.2, 0.825)],
    #         "1-2-3_right": [(0.75, 0.225), (0.575, 0.35), (0.675, 0.5), (0.5, 0.5), (0.575, 0.65), (0.75, 0.775)],
    #         "1-2-3_left": [(0.25, 0.225), (0.425, 0.35), (0.325, 0.5), (0.5, 0.5), (0.425, 0.65), (0.25, 0.775)],
    #         "3-3_right": [(0.5, 0.3), (0.7, 0.3), (0.675, 0.5), (0.5, 0.5), (0.5, 0.7), (0.7, 0.7)],
    #         "3-3_left": [(0.5, 0.3), (0.3, 0.3), (0.325, 0.5), (0.5, 0.5), (0.5, 0.7), (0.3, 0.7)],
    #         "2-4_right": [(0.7, 0.3), (0.5, 0.4), (0.675, 0.4), (0.5, 0.6), (0.675, 0.6), (0.7, 0.7)],
    #         "2-4_left": [(0.3, 0.3), (0.5, 0.4), (0.325, 0.4), (0.5, 0.6), (0.325, 0.6), (0.3, 0.7)],
    #     }

    #     for (frame_num, direction), positions in self.attack_formations.items():
    #         defender_team = 'red' if direction == 'right' else 'white'
    #         defenders = [(x, y) for x, y, team_color, _ in positions if team_color == defender_team]

    #         if len(defenders) < 6:
    #             continue

    #         # ゴール側に近い6人を選ぶ（xの大小で判定）
    #         sorted_defenders = sorted(
    #             defenders,
    #             key=lambda pos: pos[0] if direction == 'right' else -pos[0]
    #         )
    #         selected_defenders = np.array(sorted_defenders[:6])

    #         # 最もマッチ度が高いフォーメーションを探す
    #         best_formation = None
    #         best_cost = float('inf')

    #         for name, template in formation_templates.items():
    #             if (direction == 'right' and not name.endswith('right')) or \
    #                (direction == 'left' and not name.endswith('left')):
    #                 continue  # 方向が一致するものだけを評価

    #             template_positions = np.array(template)
    #             cost_matrix = cdist(selected_defenders, template_positions)
    #             row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #             total_cost = cost_matrix[row_ind, col_ind].sum()

    #             if total_cost < best_cost:
    #                 best_cost = total_cost
    #                 best_formation = name.replace("_right", "").replace("_left", "")

    #         confidence = round(1 / (1 + best_cost), 3)  # 距離が小さいほど信頼度高い
    #         classified.append((frame_num, direction, best_formation, confidence))

    #     return classified

    # def classify_formations(self):
    #     """全選手から6人を選び、最も一致度の高い組を使ってフォーメーション分類"""
    #     classified_formations = []

    #     all_items = list(self.attack_formations.items())

    #     for (frame_num, direction), positions in tqdm(all_items, desc="Processing frames"):
    #         if len(positions) < 6:
    #             continue

    #         best_formation = None
    #         best_confidence = 0
    #         min_total_distance = float('inf')

    #         # すべての6人組み合わせを評価
    #         for six_positions in combinations(positions, 6):
    #             coords_only = [(x, y) for x, y, _, _ in six_positions]
    #             coords_only_arr = np.array(coords_only)

    #             for formation, ideal_positions in formation_positions.items():
    #                 if len(ideal_positions) != 6:
    #                     continue
    #                 ideal_positions_arr = np.array(ideal_positions)

    #                 cost_matrix = cdist(coords_only_arr, ideal_positions_arr)
    #                 row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #                 total_distance = cost_matrix[row_ind, col_ind].sum() / 6

    #                 if total_distance < min_total_distance:
    #                     min_total_distance = total_distance
    #                     best_formation = formation
    #                     best_confidence = 1 / (1 + total_distance)

    #         if best_formation:
    #             classified_formations.append((frame_num, direction, best_formation, best_confidence))

    #     return classified_formations

    # def detect_defense_phases(self, min_length=0):  # ← min_lengthのスキップ条件は既に除去済み
    #     frames_by_direction = defaultdict(list)
    #     for (frame_num, direction), players in self.attack_formations.items():
    #         frames_by_direction[direction].append((frame_num, players))

    #     all_frames = sorted(self.attack_formations.items())  # 全direction対象で順番に見ていく
    #     phases = []

    #     i = 0
    #     while i < len(all_frames):
    #         (frame_num, direction), players = all_frames[i]
    #         defender_team = 'red' if direction == 'right' else 'white'
    #         defenders = [(x, y, pid) for x, y, team, pid in players if team == defender_team]

    #         if len(defenders) >= 6:
    #             # 9mライン外にいる防御選手を取得
    #             outer_defenders = [
    #                 pid for x, y, pid in defenders
    #                 if ((direction == 'right' and 0.4 < x < 0.55) or
    #                     (direction == 'left' and 0.45 < x < 0.6)) and (0.2 < y < 0.8)
    #             ]
    #             if not outer_defenders:
    #                 i += 1
    #                 continue

    #             start_frame = frame_num
    #             start_direction = direction
    #             j = i + 1

    #             while j < len(all_frames):
    #                 (next_frame, next_direction), next_players = all_frames[j]

    #                 # ✅ 終了条件②: direction が変わったら終了
    #                 if next_direction != start_direction:
    #                     break

    #                 # まだ direction が同じなら9mラインチェックを続ける
    #                 next_defenders = {
    #                     (pid, x, y) for x, y, team, pid in next_players if team == defender_team
    #                 }
    #                 remaining = [
    #                     pid for pid in outer_defenders if any(
    #                         p == pid and
    #                         ((direction == 'right' and 0.4 < x < 0.55) and (0.2 < y < 0.8) or
    #                          (direction == 'left' and 0.45 < x < 0.6)) and (0.2 < y < 0.8)
    #                         for p, x, y in next_defenders
    #                     )
    #                 ]
    #                 if not remaining:
    #                     break  # 全員戻ったので終了

    #                 j += 1

    #             # 抽出したインデックス j-1 のフレーム番号を int として取得
    #             if j < len(all_frames):
    #                 end_frame = all_frames[j - 1][0][0]
    #             else:
    #                 end_frame = all_frames[-1][0][0]

    #             phases.append((start_frame, end_frame, direction))
    #             i = j  # 次のフェーズ探索へ
    #         else:
    #             i += 1

    #     return phases

    # def detect_defense_phases(self):
    #     all_frames = sorted(self.attack_formations.items())
    #     phases = []

    #     current_direction = None
    #     i = 0

    #     while i < len(all_frames):
    #         (frame_num, direction), players = all_frames[i]

    #         # direction が切り替わったとき、新しいフェーズを探す
    #         if direction != current_direction:
    #             current_direction = direction
    #             defender_team = 'red' if direction == 'right' else 'white'

    #             # フェーズの開始フレームを特定（9mライン外の条件は無視）
    #             while i < len(all_frames):
    #                 (start_frame, dir_check), players = all_frames[i]
    #                 if dir_check != direction:
    #                     break

    #                 defenders = [(x, y, pid) for x, y, team, pid in players if team == defender_team]
    #                 if len(defenders) >= 6:
    #                     break  # 6人以上いればフェーズ開始
    #                 i += 1

    #             if i >= len(all_frames):
    #                 break

    #             start_frame = all_frames[i][0][0]
    #             defenders = [(x, y, pid) for x, y, team, pid in players if team == defender_team]

    #             # outer_defenders を開始時点で記録しておく
    #             outer_defenders = [
    #                 pid for x, y, pid in defenders
    #                 if ((direction == 'right' and 0.4 < x < 0.55) or
    #                     (direction == 'left' and 0.45 < x < 0.6)) and (0.2 < y < 0.8)
    #             ]

    #             # フェーズの終了を検出
    #             j = i + 1
    #             while j < len(all_frames):
    #                 (next_frame, next_direction), next_players = all_frames[j]
    #                 if next_direction != direction:
    #                     break  # direction が切り替わったらフェーズ終了

    #                 next_defenders = {
    #                     (pid, x, y) for x, y, team, pid in next_players if team == defender_team
    #                 }

    #                 # outer_defenders にいた誰かが戻ったら終了
    #                 for pid in outer_defenders:
    #                     for p, x, y in next_defenders:
    #                         if p == pid:
    #                             if direction == 'right' and not (0.4 < x < 0.55 and 0.2 < y < 0.8):
    #                                 j -= 1  # 直前のフレームで終了
    #                                 break
    #                             elif direction == 'left' and not (0.45 < x < 0.6 and 0.2 < y < 0.8):
    #                                 j -= 1
    #                                 break
    #                     else:
    #                         continue
    #                     break
    #                 else:
    #                     j += 1
    #                     continue
    #                 break  # 誰か戻っていた

    #             end_frame = all_frames[j][0][0] if j < len(all_frames) else all_frames[-1][0][0]

    #             min_phase_length = 50
    #             if end_frame - start_frame < min_phase_length:
    #                 # directionの切替で終了した場合はスキップする
    #                 if j < len(all_frames) and all_frames[j][0][1] != direction:
    #                     i = j + 1
    #                     continue

    #             phases.append((start_frame, end_frame, direction))
    #             i = j + 1
    #         else:
    #             i += 1

    #     return phases

    def detect_defense_phases(self, min_phase_length=50):
        all_frames = sorted(self.attack_formations.items())
        phases = []

        current_direction = None
        i = 0
        while i < len(all_frames):
            (frame_num, direction), players = all_frames[i]

            if direction != current_direction:
                current_direction = direction
                defender_team = 'red' if direction == 'right' else 'white'

                # フェーズ開始探し
                while i < len(all_frames):
                    (start_frame, dir_check), players = all_frames[i]
                    if dir_check != direction:
                        break
                    defenders = [(x, y, pid) for x, y, team, pid in players if team == defender_team]
                    if len(defenders) >= 6:
                        break
                    i += 1

                if i >= len(all_frames):
                    break

                start_frame = all_frames[i][0][0]
                defenders = [(x, y, pid) for x, y, team, pid in players if team == defender_team]
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
                    if next_direction != direction:
                        end_reason = 'direction_change'
                        break

                    next_defenders = {
                        (pid, x, y) for x, y, team, pid in next_players if team == defender_team
                    }

                    someone_returned = False
                    for pid in outer_defenders:
                        for p, x, y in next_defenders:
                            if p == pid:
                                if direction == 'right' and not (0.4 < x < 0.55 and 0.2 < y < 0.8):
                                    someone_returned = True
                                elif direction == 'left' and not (0.45 < x < 0.6 and 0.2 < y < 0.8):
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



    # def get_dominant_formations_by_defense_phase(self, classified_formations, defense_phases, min_length=0):
    #     """
    #     各守備フェーズ内で信頼度加重平均が最も高いフォーメーションを代表とする。
    #     """
    #     direction_indexed = defaultdict(list)
    #     for frame_num, direction, formation, confidence in classified_formations:
    #         direction_indexed[direction].append((frame_num, formation, confidence))

    #     dominant_formations = []
    #     for start_frame, end_frame, direction in defense_phases:
    #         # フェーズ内のフォーメーション＋信頼度を抽出
    #         relevant = [
    #             (form, conf)
    #             for f, form, conf in direction_indexed[direction]
    #             if start_frame <= f <= end_frame
    #         ]
    #         if not relevant:
    #             continue

    #         # フォーメーションごとに信頼度合計とカウントを集計
    #         formation_conf = defaultdict(float)
    #         formation_cnt  = defaultdict(int)
    #         for form, conf in relevant:
    #             formation_conf[form] += conf
    #             formation_cnt[form]  += 1

    #         # 平均信頼度が最大のフォーメーションを選択
    #         best_formation = max(
    #             formation_conf,
    #             key=lambda f: formation_conf[f] / formation_cnt[f]
    #         )

    #         if end_frame - start_frame >= min_length:
    #             dominant_formations.append((start_frame, end_frame, best_formation, direction))

    #     return dominant_formations
    
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
