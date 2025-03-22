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
import pandas as pd

# Google DriveのファイルIDと保存先パス
file_id = "1ZVb43bovhpM-L1AzdHQSCurL-Kx5N5kj"
model_path = "chart_pattern_model.h5"

# モデルがなければGoogle Driveからダウンロード
@st.cache_resource
def load_ai_model():
    if not os.path.exists(model_path):
        st.write("モデルをダウンロード中です...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

model = load_ai_model()

# タイトル
st.title("チャート画像パターン予測AI")
st.write("株コードを入力すると、過去60日のチャートを自動生成してAIが予測します。")

# 株コード入力欄（例：7203.T）
symbol = st.text_input("株コードを入力してください（例：7203.T）")

# ボタンが押されたらチャート生成＋予測
if st.button("予測する") and symbol:
    try:
        # 日付設定（過去90日分を取得）
        end = datetime.date.today()
        start = end - datetime.timedelta(days=90)

        # 株価データ取得
        data = yf.download(symbol, start=start, end=end)

        # カラムがMultiIndexの場合フラット化
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data = data.dropna()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = data[col].astype(float)

        if len(data) < 60:
            st.warning("60日以上のデータがないため、予測できません。")
        else:
            # チャート画像生成
            chart_data = data[-60:]
            chart_path = f"{symbol}_latest.png"
            mpf.plot(chart_data, type='candle', style='charles', savefig=chart_path)

            # 表示
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