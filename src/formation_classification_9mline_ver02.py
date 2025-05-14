"""
CSVデータから選手の位置情報をフレームごとに取得．
CSVデータは次の形式（frame_num,id,team_color,x,y,direction）
定義した9mラインよりも外側にいる防御選手の数でフレームごとにフォーメーションを推定していく．
防御選手とは読み込んだCSVのdirectionがrightの場合はteam_colorがredの選手．leftの時はwhiteの選手で限定する．
フレームを防御フェーズに分類し，その中で一番多く推定されたフォーメーションをその防御フェーズにおけるフォーメーションとする．
防御フェーズの決め方は以下とする．
CSVデータから一番最初に防御選手を6人以上検出しているフレームを開始フレームとする．
開始フレームで9mラインよりも外側にいる防御選手と同じIDの選手が9mラインよりも内側に戻ったタイミングを終了フェーズとする．
ただし，もしいなければdirectionが切り替わったタイミングを終了フェーズとする．
CSVデータの一番最後までこの処理を実行して，防御フェーズを決定する．
各防御フェーズの開始フレームから終了フレーム，その間のフレーム間で一番多く推定されたフォーメーションを最終的なフォーメーションの結果としてCSVに出力する
"""

import csv
from collections import defaultdict, Counter
import numpy as np
import time
from tqdm import tqdm

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
        """CSVファイルを読み込み、フレームごとに選手位置をまとめる"""
        frames = defaultdict(list)
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                frame_num = int(row[0])
                player_id = row[1]
                team_color = row[2]
                x = float(row[3])
                y = float(row[4])
                direction = row[5]
                # store id, team_color, x, y
                frames[(frame_num, direction)].append((player_id, team_color, x, y))
        return frames

##################フォーメーションの判別を行うメソッドここから##################

    def classify_formations(self):
        """フレームごとに9mラインの外側にいる防御選手の数でフォーメーションを推定"""
        classified_formations = []
        for (frame_num, direction), players in sorted(self.attack_formations.items()):
            # カウント対象：方向に応じた team_color の防御選手
            outside_count = 0
            for pid, team_color, x, y in players:
                if direction == "right" and team_color == "red" and 0.4 < x < 0.55 and 0.2 < y < 0.8:
                    outside_count += 1
                elif direction == "left" and team_color == "white" and 0.45 < x < 0.6 and 0.2 < y < 0.8:
                    outside_count += 1
            # 推定フォーメーション
            if   outside_count == 0: formation = "0-6 Formation"
            elif outside_count == 1: formation = "1-5 Formation"
            elif outside_count == 2: formation = "2-4 Formation"
            elif outside_count == 3: formation = "3-3 Formation"
            else:                    formation = "Unknown Formation"
            classified_formations.append((frame_num, direction, formation, 1.0))
        return classified_formations

    def get_dominant_formations(self, classified_formations, min_length=50):
        """
        directionごとにフェーズを区切り、短いフェーズは結合し、
        防御フェーズ内で最も多く出現したフォーメーションを決定する
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

        # フェーズを結合
        phases = combine_phases(phases)

        # 各フェーズ内で防御フェーズを特定し、最も多いフォーメーションを決定
        dominant_formations = []
        
        for phase in phases:
            frame_nums = [p[0] for p in phase]
            formations = [p[2] for p in phase]
            confidences = [p[3] for p in phase]
            direction = phase[0][1]
            
            start_frame = frame_nums[0]
            end_frame = frame_nums[-1]
            
            # 各フレームでの防御選手を確認
            defense_phases = []
            current_defense_phase = None
            outside_defense_ids = None
            
            for idx, frame_num in enumerate(frame_nums):
                # 防御選手を取得
                frame_data = self.attack_formations.get((frame_num, direction), [])
                defense_players = [
                    (pid, x, y) for pid, team_color, x, y in frame_data
                    if ((direction == "right" and team_color == "red") or 
                        (direction == "left" and team_color == "white"))
                ]
                
                # 防御選手が6人以上いる場合のみ処理
                if len(defense_players) >= 6:
                    # 外側にいる防御選手IDを取得
                    current_outside_ids = [
                        pid for pid, x, y in defense_players
                        if ((direction == "right" and 0.4 < x < 0.55 and 0.2 < y < 0.8) or 
                            (direction == "left" and 0.45 < x < 0.6 and 0.2 < y < 0.8))
                    ]
                    
                    # 新しい防御フェーズの開始
                    if current_defense_phase is None:
                        current_defense_phase = [idx]
                        outside_defense_ids = current_outside_ids
                    # 既存の防御フェーズに追加
                    else:
                        current_defense_phase.append(idx)
                        
                        # 外側にいた選手が内側に戻った場合、防御フェーズ終了
                        if outside_defense_ids and all(
                            pid not in current_outside_ids for pid in outside_defense_ids
                        ):
                            defense_phases.append((current_defense_phase[0], idx))
                            current_defense_phase = None
                            outside_defense_ids = None
                
                # 防御選手が6人未満の場合、現在の防御フェーズ終了
                elif current_defense_phase is not None:
                    defense_phases.append((current_defense_phase[0], idx - 1))
                    current_defense_phase = None
                    outside_defense_ids = None
            
            # 終了していない防御フェーズがあれば追加
            if current_defense_phase is not None:
                defense_phases.append((current_defense_phase[0], len(frame_nums) - 1))
            
            # 各防御フェーズについて最も多いフォーメーションを決定
            for start_idx, end_idx in defense_phases:
                phase_start_frame = frame_nums[start_idx]
                phase_end_frame = frame_nums[end_idx]
                
                if phase_end_frame - phase_start_frame >= min_length:
                    phase_formations = formations[start_idx:end_idx+1]
                    phase_confidences = confidences[start_idx:end_idx+1]
                    
                    # フォーメーションの出現数と信頼度を計算
                    formation_confidence = defaultdict(float)
                    formation_count = defaultdict(int)
                    
                    for form, conf in zip(phase_formations, phase_confidences):
                        formation_confidence[form] += conf
                        formation_count[form] += 1
                    
                    if formation_count:
                        # 最も信頼度が高いフォーメーションを選択
                        best_formation = max(
                            formation_confidence, 
                            key=lambda f: formation_confidence[f] / formation_count[f]
                        )
                        dominant_formations.append((phase_start_frame, phase_end_frame, best_formation, direction))
        
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
    print("Classifying formations...")
    classified_formations = classifier.classify_formations()
    print(f"Found {len(classified_formations)} classified formations")

    dominant_formations = classifier.get_dominant_formations(classified_formations)
    print(f"Found {len(dominant_formations)} defensive phases")
    
    print("Saving results...")
    classifier.save_dominant_formations(dominant_formations, classified_formations, output_file)

    end_time = time.time()  # End timing
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")



