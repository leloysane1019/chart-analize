import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import requests

# モデルをGoogle Driveからダウンロード（初回のみ）
model_path = "chart_pattern_model.h5"
model_url = "https://drive.google.com/uc?export=download&id=1ZVb43bovhpM-L1AzdHQSCurL-Kx5N5kj"

if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        r = requests.get(model_url)
        f.write(r.content)

# モデル読み込み
model = load_model(model_path)

# Streamlitアプリ
st.title("チャート画像パターン予測AI")
st.write("画像をアップロードすると、上がるか下がるかをAIが予測します。")

uploaded_file = st.file_uploader("チャート画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="アップロードされた画像", use_column_width=True)

    img = img.resize((224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)[0][0]
    st.write(f"予測スコア: **{prediction:.3f}**")

    if prediction > 0.5:
        st.success("→ このチャートは **上がりそう！**")
    else:
        st.warning("→ このチャートは **下がりそう...**")