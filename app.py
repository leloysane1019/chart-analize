import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# モデル読み込み
model = load_model("chart_pattern_model.h5")

st.title("チャート画像パターン予測AI")
st.write("画像をアップロードすると、今後上がるか下がるか予測します。")

# 画像アップロード
uploaded_file = st.file_uploader("チャート画像を選んでください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="アップロードされた画像", use_column_width=True)

    # 画像を前処理
    img = img.resize((224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # 予測
    prediction = model.predict(x)[0][0]
    st.write(f"予測スコア（1に近いほど上昇）: **{prediction:.3f}**")

    if prediction > 0.5:
        st.success("→ このチャートは **上がりそう！**")
    else:
        st.warning("→ このチャートは **下がりそう...**")