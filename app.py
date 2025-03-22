import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import gdown

# Google DriveのファイルIDと保存先パス
file_id = "1ZVb43bovhpM-L1AzdHQSCurL-Kx5N5kj"
model_path = "chart_pattern_model.h5"

# モデルがなければGoogle Driveからダウンロード
if not os.path.exists(model_path):
    st.write("モデルをダウンロード中です...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# モデル読み込み
model = load_model(model_path)

# アプリタイトル
st.title("チャート画像パターン予測AI")
st.write("チャート画像をアップロードすると、株価が上がるか下がるかをAIが予測します。")

# 画像アップローダー
uploaded_file = st.file_uploader("チャート画像を選んでください", type=["png", "jpg", "jpeg"])

# 画像がアップロードされたとき
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="アップロードされた画像", use_column_width=True)

    # 画像前処理
    img = img.resize((224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # AIで予測
    prediction = model.predict(x)[0][0]
    st.write(f"予測スコア（1に近いほど上昇）: **{prediction:.3f}**")

    # 結果表示
    if prediction > 0.5:
        st.success("→ このチャートは **上がりそう！**")
    else:
        st.warning("→ このチャートは **下がりそう...**")