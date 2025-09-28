# app.py  清爽·极速版
import streamlit as st
import pathlib
import torch
import PIL.Image
import io
from anime_infer import run_infer

# ----------- 页面美化 -----------
st.set_page_config(page_title="XuのAnimeGAN2", page_icon="🎨", layout="centered")

with st.sidebar:
    st.title("🎌 XuのAnimeGAN2")
    st.markdown("上传真人照片，3~8 秒生成动漫风格。")

    # ① 模型选择
    model_name = st.selectbox(
        "选择风格模型",
        ["face_paint_512_v2.pt", "face_paint_512_v1.pt", "celeba_distill.pt", "paprika.pt"]
    )

    # ② 中文解释（动态更新）
    desc = {
        "face_paint_512_v2.pt": "✨ 人像 V2：笔触柔和、肤色自然，**最推荐**",
        "face_paint_512_v1.pt": "🎨 人像 V1：线条偏粗、色彩饱和",
        "celeba_distill.pt": "📸 真人蒸馏：速度最快，适合自拍，细节略少",
        "paprika.pt": "🌄 风景/建筑：色彩鲜艳，**不要用人脸**",
    }
    st.caption(desc.get(model_name, ""))

    # ③ 设备选择
    device = st.radio("运行设备", ["cpu", "cuda"], disabled=not torch.cuda.is_available())

# ----------- 缓存推理函数（同图2秒内返回） -----------
@st.cache_data(show_spinner=False)
def _run_anime(img_bytes: bytes, model: str, device: str) -> bytes:
    """缓存+压缩：输入原始字节，返回动漫化图片字节"""
    tmp = pathlib.Path("tmp")
    tmp.mkdir(exist_ok=True)
    inp = tmp / "in.png"
    out = tmp / "in.png"

    # ① 打开即压缩到 720p，保持比例
    img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((720, 720), PIL.Image.LANCZOS)   # 网络传输↓70%
    img.save(inp, quality=85)                      # 再压质量

    # ② 推理（无 print，页面干净）
    run_infer(checkpoint=f"weights/{model}",
              input_dir=str(tmp),
              output_dir=str(tmp),
              device=device,
              upsample_align=False)

    # ③ 返回字节
    return out.read_bytes()

# ----------- 主界面 -----------
st.title("📸 真人变动漫")
uploaded = st.file_uploader("拖拽或点击上传图片", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    col1, col2 = st.columns(2)
    col1.image(uploaded, caption='原图', use_container_width=True)

    # 一键动漫化
    with st.spinner("AI 正在动漫化，请稍候…"):
        anime_bytes = _run_anime(uploaded.getvalue(), model_name, device)

    col2.image(anime_bytes, caption='动漫化', use_container_width=True)
    st.download_button("⬇️ 下载结果", data=anime_bytes,
                      file_name="anime.png", mime="image/png")
    st.success("完成！右侧图片可右键另存。")
