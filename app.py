# app.py
import streamlit as st
import pathlib
import torch
import PIL.Image
import traceback
from anime_infer import run_infer

# ----------- 页面美化 -----------
st.set_page_config(
    page_title="AnimeGAN2 在线动漫化",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="auto"
)

# ----------- 侧边栏 -----------
with st.sidebar:
    st.title("🎌 AnimeGAN2")
    st.markdown("上传真人照片，3~8 秒生成动漫风格。")
    model_name = st.selectbox(
        "选择风格模型",
        ["face_paint_512_v2.pt", "face_paint_512_v1.pt",
         "celeba_distill.pt", "paprika.pt"]
    )
    device = st.radio(
        "运行设备",
        ["cpu", "cuda"],
        disabled=not torch.cuda.is_available()
    )

# ----------- 主区域 -----------
st.title("📸 真人变动漫")
uploaded = st.file_uploader("拖拽或点击上传图片", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    img = PIL.Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    col1.image(img, caption='原图', use_container_width=True)

    # 临时目录 & 路径
    tmp = pathlib.Path("tmp")
    tmp.mkdir(exist_ok=True)
    inp = tmp / "in.png"
    out = tmp / "out.png"
    img.save(inp)

    # ========== 核心：捕获一切异常 ==========
    try:
        with st.spinner("AI 正在动漫化，请稍候…"):
            run_infer(
                checkpoint=f"weights/{model_name}",
                input_dir=str(tmp),
                output_dir=str(tmp),
                device=device,          # 云端先强制 cpu 也可
                upsample_align=False
            )
        # 检查是否真生成图片
        if not out.exists():
            raise FileNotFoundError(
                f"模型跑完但找不到 {out}，请检查权重路径/依赖/日志。"
            )
        result = PIL.Image.open(out)
        col2.image(result, caption='动漫化', use_container_width=True)

        # 下载按钮
        btn = st.download_button(
            label="⬇️ 下载结果",
            data=out.read_bytes(),
            file_name="anime.png",
            mime="image/png"
        )
        st.success("完成！右侧图片可右键另存。")

    except Exception as e:
        st.error("❌ 推理失败")
        st.text(str(e))
        with st.expander("查看详细 traceback"):
            st.code(traceback.format_exc())
