"""
マニュアルのフェーズに基づいて、フォーメーションを抽出するスクリプトです．
研究用で本番で使用はしません．
このスクリプトは、手動で指定されたフレーム範囲に基づいてフォーメーションを抽出します．
"""

import pandas as pd

# ファイル読み込み
result_df = pd.read_csv('../data/output/formations_output_test.csv', header=0)
result_df.columns = ['開始フレーム', '終了フレーム', 'フォーメーション', '方向', '信頼度','内訳']

# フレーム列を整数型に変換（エラー処理付き）
def safe_convert_to_int(column):
    try:
        return column.astype(int)
    except ValueError:
        print(f"列 {column.name} に数値以外の値が含まれています。データを確認してください。")
        return pd.to_numeric(column, errors='coerce').fillna(0).astype(int)

result_df['開始フレーム'] = safe_convert_to_int(result_df['開始フレーム'])
result_df['終了フレーム'] = safe_convert_to_int(result_df['終了フレーム'])

# 目視フェーズ読み込み
manual_df = pd.read_csv('../data/manual_phases.csv')
manual_df['start_frame'] = manual_df['start_frame'].astype(int)
manual_df['end_frame'] = manual_df['end_frame'].astype(int)

# フォーメーション抽出
formations = []

for _, row in manual_df.iterrows():
    manual_start = row['start_frame']
    manual_end = row['end_frame']

    overlap = result_df[
        (result_df['終了フレーム'] >= manual_start) & (result_df['開始フレーム'] <= manual_end)
    ].copy()

    # if not overlap.empty:
    #     best_formation = overlap.loc[overlap['信頼度'].astype(float).idxmax()]['フォーメーション']
    # else:
    #     best_formation = 'Unknown'  

    if not overlap.empty:
        # 出現回数が最も多いフォーメーションを選択
        best_formation = overlap['フォーメーション'].value_counts().idxmax()
    else:
        best_formation = 'Unknown'

    formations.append(best_formation)

# 一列のデータフレームとして保存
formations_df = pd.DataFrame({'formation': formations})
formations_df.to_csv('../data/output_tmp.csv', index=False)

print("フォーメーション列のみを保存しました。")