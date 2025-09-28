#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnimeGAN2 - 背景主题切换 | 左中右 | 零闪退
"""
import sys, pathlib, datetime, tempfile, shutil
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ----------- 配置 -----------
WEIGHTS_DIR = pathlib.Path(__file__).parent / "weights"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

CHINESE_MAP = {
    "face_paint_512_v1.pt": "FacePaint 512 V1（人像尝鲜，笔触更粗）",
    "face_paint_512_v2.pt": "FacePaint 512 V2（人像/插画，效果最均衡）",
    "celeba_distill.pt":    "Celeba 人脸蒸馏（真人自拍，速度最快）",
    "paprika.pt":           "Paprika 风景（室外/建筑，色彩鲜艳）",
}

THEMES = {
    "动漫城":   {"img": "theme_city.jpg",   "color": "#2b2d42"},
    "龙九乙茶屋": {"img": "theme_teahouse.jpg", "color": "#3e2723"},
    "樱花夜":   {"img": "theme_sakura.jpg", "color": "#4a148c"},
}

def scan_models():
    return sorted([p for p in WEIGHTS_DIR.glob("*.pt") if p.is_file()]) if WEIGHTS_DIR.exists() else []

def make_out_dir():
    parent = pathlib.Path(__file__).parent / "outputs"
    parent.mkdir(exist_ok=True)
    sub = parent / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sub.mkdir(exist_ok=True)
    return sub

# ----------- 线程：直接调用函数，无子进程 -----------
from anime_infer import run_infer   # ★ 关键导入

class ConvertThread(QThread):
    log    = pyqtSignal(str)
    prog   = pyqtSignal(int)
    finished_one = pyqtSignal(object)

    def __init__(self, files, model, device):
        super().__init__()
        self.files, self.model, self.device = files, model, device

    def run(self):
        for idx, file in enumerate(self.files, 1):
            self.log.emit(f"[{idx}/{len(self.files)}] {file.name}")
            temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="animegan_"))
            temp_file = temp_dir / file.name
            try:
                Image.open(file).save(temp_file)
                # ★ 直接调用函数，不再起任何进程
                run_infer(
                    checkpoint=str(self.model),
                    input_dir=str(temp_dir),
                    output_dir=str(temp_dir),
                    device=self.device
                )
                out_file = temp_dir / file.name
                if out_file.exists():
                    with Image.open(out_file) as img:
                        img = img.convert("RGB")
                        img.thumbnail((512, 512), Image.LANCZOS)
                        self.finished_one.emit(img.copy())
                    self.log.emit("  ✔ 生成图预览已更新")
                else:
                    self.log.emit("  ✘ 未找到生成图")
            except Exception as e:
                self.log.emit(f"  ✘ 线程异常: {e}")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.prog.emit(idx)
        self.log.emit("=== 全部完成 ===")


# ================== 以下 UI 代码与原文件相同 ==================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("动漫风格转换--Xu")
        self.resize(1300, 800)
        self.setAcceptDrops(True)
        self.models = scan_models()
        if not self.models:
            QMessageBox.warning(self, "提示", f"请把 .pt 权重放到\n{WEIGHTS_DIR}\n再重启"); sys.exit()
        self.out_dir = make_out_dir()
        self.current_img = None
        self.thread = None
        self._init_ui()

    def _init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        grid = QGridLayout(central)

        left_gb = QGroupBox("文件列表（拖拽/右键）")
        self.list_w = QListWidget(); self.list_w.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_w.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_w.customContextMenuRequested.connect(self._list_menu)
        v = QVBoxLayout(left_gb); v.addWidget(self.list_w)
        grid.addWidget(left_gb, 0, 0)

        mid_gb = QGroupBox("原图 / 生成图 / 日志")
        v = QVBoxLayout(mid_gb)
        v.addWidget(QLabel("原图预览:"))
        self.src_lbl = QLabel("暂无原图"); self.src_lbl.setAlignment(Qt.AlignCenter); self.src_lbl.setMinimumSize(400, 300); self.src_lbl.setStyleSheet("border:1px solid #555")
        v.addWidget(self.src_lbl, alignment=Qt.AlignHCenter)
        v.addWidget(QLabel("生成图预览:"))
        self.dst_lbl = QLabel("暂无生成图"); self.dst_lbl.setAlignment(Qt.AlignCenter); self.dst_lbl.setMinimumSize(400, 300); self.dst_lbl.setStyleSheet("border:1px solid #555")
        v.addWidget(self.dst_lbl, alignment=Qt.AlignHCenter)
        self.bar = QProgressBar(); v.addWidget(self.bar)
        self.log_te = QTextEdit(); self.log_te.setMaximumHeight(120)
        v.addWidget(QLabel("实时日志:")); v.addWidget(self.log_te)
        grid.addWidget(mid_gb, 0, 1)

        right_gb = QGroupBox("控制面板")
        v = QVBoxLayout(right_gb)
        v.addWidget(QLabel("风格模型:"))
        self.model_cb = QComboBox()
        for m in self.models:
            self.model_cb.addItem(CHINESE_MAP.get(m.name, m.stem))
        v.addWidget(self.model_cb)
        v.addWidget(QLabel("设备:"))
        self.dev_cb = QComboBox(); self.dev_cb.addItems(["cpu", "cuda:0"]); v.addWidget(self.dev_cb)
        v.addWidget(QLabel("下载目录:"))
        self.out_lbl = QLabel(str(self.out_dir)); self.out_lbl.setWordWrap(True); v.addWidget(self.out_lbl)
        choose_btn = QPushButton("更改目录"); choose_btn.clicked.connect(self._choose_out); v.addWidget(choose_btn)
        v.addStretch()
        down_btn = QPushButton("开始下载"); down_btn.clicked.connect(self._download); v.addWidget(down_btn)
        start_btn = QPushButton("开始转换"); start_btn.clicked.connect(self._start); v.addWidget(start_btn)

        theme_lbl = QLabel("主题:")
        self.theme_cb = QComboBox()
        self.theme_cb.addItems(list(THEMES.keys()))
        self.theme_cb.currentTextChanged.connect(self._apply_theme)
        v.addWidget(theme_lbl)
        v.addWidget(self.theme_cb)

        grid.addWidget(right_gb, 0, 2)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1); grid.setColumnStretch(2, 1)
        self._apply_theme("动漫城")

    # ------------ 主题 & 功能 ------------
    def _apply_theme(self, name):
        if name not in THEMES: return
        img_path = pathlib.Path(__file__).parent / THEMES[name]["img"]
        color = THEMES[name]["color"]
        if not img_path.exists():
            self.log_te.append(f">>> 主题图片不存在: {img_path}"); return
        pixmap = QPixmap(str(img_path)).scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        palette = QPalette(); palette.setBrush(QPalette.Window, QBrush(pixmap)); self.setPalette(palette)
        self.setStyleSheet(f"""
        QMainWindow, QGroupBox, QLabel, QPushButton, QComboBox, QTextEdit, QProgressBar{{
            background-color: {color}22; color: #ffffff; border: 1px solid {color}55; border-radius: 6px; padding: 4px;
        }}
        QPushButton:hover{{ background-color: {color}44; }}
        """)

    def _choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择下载目录")
        if d:
            self.out_dir = pathlib.Path(d)
            self.out_lbl.setText(str(self.out_dir))

    def _list_menu(self, pos):
        menu = QMenu()
        menu.addAction("删除选中", lambda: [self.list_w.takeItem(self.list_w.row(i)) for i in self.list_w.selectedItems()])
        menu.addAction("清空列表", self.list_w.clear)
        menu.exec_(self.list_w.mapToGlobal(pos))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = pathlib.Path(url.toLocalFile())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                self.list_w.clear()
                QListWidgetItem(str(p), self.list_w)
                self._show_src(p); break
            elif p.is_dir():
                self.list_w.clear()
                for ext in IMAGE_EXTS:
                    for fp in p.rglob(f"*{ext}"): QListWidgetItem(str(fp), self.list_w)
                if self.list_w.count(): self._show_src(pathlib.Path(self.list_w.item(0).text()))
                break

    def _show_src(self, file: pathlib.Path):
        self.src_lbl.setPixmap(QPixmap(str(file)).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_dst(self, img: Image.Image):
        self.current_img = img
        self.log_te.append(f">>> 收到生成图: {img.format} {img.size} mode={img.mode}")
        rgb = img.convert("RGB"); data = rgb.tobytes()
        qimg = QImage(data, rgb.width, rgb.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(400, 300, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.dst_lbl.setPixmap(pixmap)
        self.dst_lbl.update()
        self.log_te.append(">>> 生成图已刷新（Fast）")

    def _download(self):
        if self.current_img is None:
            QMessageBox.information(self, "提示", "尚未生成图片，请先转换！"); return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存生成图", str(self.out_dir / "anime.png"),
                                                   "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if save_path:
            self.current_img.save(save_path)
            self.log_te.append(f"✔ 已保存到: {save_path}")

    def _start(self):
        if self.list_w.count() == 0:
            QMessageBox.warning(self, "提示", "列表为空！"); return
        files = [pathlib.Path(self.list_w.item(i).text()) for i in range(self.list_w.count())]
        self.thread = ConvertThread(files, self.models[self.model_cb.currentIndex()], self.dev_cb.currentText())
        self.thread.log.connect(self.log_te.append)
        self.thread.prog.connect(self.bar.setValue)
        self.bar.setMaximum(len(files))
        start_btn = self.sender() or [b for b in self.findChildren(QPushButton) if b.text() == "开始转换"][0]
        start_btn.setEnabled(False)
        self.thread.finished.connect(lambda: start_btn.setEnabled(True))
        self.thread.finished_one.connect(self._show_dst)
        self.thread.start()


if __name__ == '__main__':
    import traceback
    sys.excepthook = lambda cls, exc, tb: (
        print(">>> 未处理异常 <<<", file=sys.stderr),
        print(f"{cls.__name__}: {exc}", file=sys.stderr),
        print("".join(traceback.format_exception(cls, exc, tb)), file=sys.stderr)
    )
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
