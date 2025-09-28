# app.py  清爽无日志版
import streamlit as st
import pathlib
import torch
import PIL.Image
from anime_infer import run_infer

# ----------- 页面美化 -----------
st.set_page_config(page_title="XuのAnimeGAN2", page_icon="🎨", layout="centered")

with st.sidebar:
    st.title("🎌 AnimeGAN2")
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

# ----------- 主区域 -----------
st.title("📸 真人变动漫")
uploaded = st.file_uploader("拖拽或点击上传图片", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    img = PIL.Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    col1.image(img, caption='原图', use_container_width=True)

    tmp = pathlib.Path("tmp")
    tmp.mkdir(exist_ok=True)
    inp = tmp / "in.png"
    out = tmp / "in.png"
    img.save(inp)

    # ========== 轻量级兜底 ==========
    try:
        with st.spinner("AI 正在动漫化，请稍候…"):
            run_infer(checkpoint=f"weights/{model_name}",
                      input_dir=str(tmp),
                      output_dir=str(tmp),
                      device=device,
                      upsample_align=False)

        if not out.exists():
            raise RuntimeError("生成失败，请重试或换张图片。")
        result = PIL.Image.open(out)
        col2.image(result, caption='动漫化', use_container_width=True)
        st.download_button("⬇️ 下载结果", data=out.read_bytes(),
                          file_name="anime.png", mime="image/png")
        st.success("完成！右侧图片可右键另存。")

    except Exception:
        st.error("❌ 生成失败，请重试或更换图片。")
        # 不再展开 traceback，用户看不到技术堆栈
