# app.py
import streamlit as st, pathlib, torch, PIL.Image
from anime_infer import run_infer

# ---- 页面美化 ----
st.set_page_config(
    page_title="AnimeGAN2 在线动漫化",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="auto"
)

# ---- 侧边栏 ----
with st.sidebar:
    st.title("🎌 AnimeGAN2")
    st.markdown("上传真人照片，3~8 秒生成动漫风格。")
    model_name = st.selectbox("选择风格模型",
                              ["face_paint_512_v2.pt",
                               "face_paint_512_v1.pt",
                               "celeba_distill.pt",
                               "paprika.pt"])
    device = st.radio("运行设备", ["cpu", "cuda"], disabled=not torch.cuda.is_available())

# ---- 主区域 ----
st.title("📸 真人变动漫")
uploaded = st.file_uploader("拖拽或点击上传图片", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    img = PIL.Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    col1.image(img, caption='原图', use_column_width=True)

    tmp = pathlib.Path("tmp"); tmp.mkdir(exist_ok=True)
    inp = tmp / "in.png"; out = tmp / "out.png"
    img.save(inp)

    with st.spinner("AI 正在动漫化，请稍候…"):
        run_infer(checkpoint=f"weights/{model_name}",
                  input_dir=str(tmp),
                  output_dir=str(tmp),
                  device=device,
                  upsample_align=False)
    result = PIL.Image.open(out)
    col2.image(result, caption='动漫化', use_column_width=True)

    # 下载按钮
    btn = st.download_button(
        label="⬇️ 下载结果",
        data=out.read_bytes(),
        file_name="anime.png",
        mime="image/png"
    )
    st.success("完成！右侧图片可右键另存。")
