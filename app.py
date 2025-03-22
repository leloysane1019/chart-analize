import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown
import yfinance as yf
import mplfinance as mpf
import datetime

# Google DriveのファイルIDとモデルパス
file_id = "1ZVb43bovhpM-L1AzdHQSCurL-Kx5N5kj"
model_path = "chart_pattern_model.h5"

# モデルがなければGoogle Driveからダウンロード
if not os.path.exists(model_path):
    st.write("モデルをダウンロード中です...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# モデル読み込み
model = load_model(model_path)

# タイトル
st.title("チャート画像パターン予測AI")
st.write("株コードを入力すると、過去60日のチャートを自動生成してAIが予測します。")

# 株コード入力欄（例：7203.T）
symbol = st.text_input("株コードを入力してください（例：7203.T）")

# ボタンが押されたらチャート生成＋予測
if st.button("予測する") and symbol:
    try:
        # 今日の日付と60営業日前を計算
        end = datetime.date.today()
        start = end - datetime.timedelta(days=90)  # 60営業日分を確保するため多めに取得

        # データ取得
        data = yf.download(symbol, start=start, end=end)

        # カラムがMultiIndexなら平坦化
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # 欠損除去
        data = data.dropna()

        # 必要な列のみfloatに変換
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = data[col].astype(float)

        # 60本以上あるか確認
        if len(data) < 60:
            st.warning("データが60日分未満のため、チャートが生成できません。")
        else:
            # 最後の60日分をチャートとして使う
            chart_data = data[-60:]
            chart_path = f"{symbol}_latest.png"
            mpf.plot(chart_data, type='candle', style='charles', savefig=chart_path)

            # チャート画像を表示
            st.image(chart_path, caption=f"{symbol} のチャート（過去60日）", use_column_width=True)

            # AI予測
            img = Image.open(chart_path).resize((224, 224))
            x = image.img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            prediction = model.predict(x)[0][0]

            st.write(f"予測スコア: **{prediction:.3f}**")
            if prediction > 0.5:
                st.success("→ このチャートは **上がりそう！**")
            else:
                st.warning("→ このチャートは **下がりそう...**")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")