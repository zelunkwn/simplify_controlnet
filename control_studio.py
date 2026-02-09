import sys, os, cv2
import numpy as np
from datetime import datetime
from PIL import Image

from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# --- MOCK MEDIAPIPE (Prevents Crashes) ---
import sys
from unittest.mock import MagicMock
sys.modules["mediapipe"] = MagicMock()
sys.modules["mediapipe.python.solutions"] = MagicMock()

# --- WORKER: DEPTH ---
class DepthWorker(QThread):
    finished = pyqtSignal(object)
    def __init__(self, path):
        super().__init__()
        self.path = path
    def run(self):
        try:
            from transformers import pipeline
            # Use Local Model
            model_path = "./local_depth_model_large"
            if os.path.exists(model_path):
                pipe = pipeline(task="depth-estimation", model=model_path)
            else:
                pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf")
            
            pil_img = Image.open(self.path)
            result = pipe(pil_img)
            self.finished.emit(result["depth"])
        except Exception as e:
            print(f"Depth Error: {e}")
            self.finished.emit(None)

# --- WORKER: POSE ---
class PoseWorker(QThread):
    finished = pyqtSignal(object)
    def __init__(self, path):
        super().__init__()
        self.path = path
    def run(self):
        try:
            from controlnet_aux import OpenposeDetector
            model = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            pil_img = Image.open(self.path)
            # Detect Pose
            pose = model(pil_img, include_hand=True, include_face=True)
            self.finished.emit(pose)
        except Exception as e:
            print(f"Pose Error: {e}")
            self.finished.emit(None)

# --- MAIN GUI ---
class ControlStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CONTROL STUDIO v1.3 - STABLE BUILD")
        self.setGeometry(50, 50, 1400, 800)
        self.setStyleSheet("QMainWindow { background-color: #1e1e1e; } QLabel { color: #ccc; font-family: 'Segoe UI'; } QPushButton { background-color: #333; color: white; border: 1px solid #555; padding: 10px; border-radius: 5px; }")
        
        self.src_path = None
        self.img_depth = None
        self.img_pose = None
        self.initUI()

    def initUI(self):
        central = QWidget(); self.setCentralWidget(central); layout = QHBoxLayout(central)

        # SIDEBAR
        sidebar = QWidget(); sidebar.setFixedWidth(300); sb_lay = QVBoxLayout(sidebar)
        
        sb_lay.addWidget(QLabel("/// CONTROLNET SUITE ///"))
        sb_lay.addSpacing(10)

        self.btn_load = QPushButton("📂 LOAD IMAGE"); self.btn_load.clicked.connect(self.load_image)
        sb_lay.addWidget(self.btn_load)
        
        self.btn_run = QPushButton("⚡ EXTRACT ALL"); self.btn_run.clicked.connect(self.run_all)
        self.btn_run.setStyleSheet("background-color: #6200ea; font-weight: bold;")
        self.btn_run.setEnabled(False)
        sb_lay.addWidget(self.btn_run)
        
        sb_lay.addSpacing(20)
        self.btn_save = QPushButton("💾 BATCH SAVE"); self.btn_save.clicked.connect(self.save_all)
        self.btn_save.setStyleSheet("background-color: #00c853; color: black; font-weight: bold;")
        self.btn_save.setEnabled(False)
        sb_lay.addWidget(self.btn_save)

        self.status = QLabel("Ready."); self.status.setWordWrap(True); sb_lay.addWidget(self.status)
        sb_lay.addStretch()
        layout.addWidget(sidebar)

        # VIEWPORT (TABS)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #444; } QTabBar::tab { background: #222; color: #fff; padding: 10px; } QTabBar::tab:selected { background: #444; }")
        
        # Tabs
        self.lbl_orig = self.create_tab("Original")
        self.lbl_depth = self.create_tab("Depth Mask")
        self.lbl_pose = self.create_tab("OpenPose")
        self.lbl_combo = self.create_tab("Overlay (Check Alignment)")
        
        layout.addWidget(self.tabs)

    def create_tab(self, name):
        lbl = QLabel(name); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tabs.addTab(lbl, name)
        return lbl

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.src_path = path
            self.display(path, self.lbl_orig)
            self.btn_run.setEnabled(True)
            self.status.setText(f"Loaded: {os.path.basename(path)}")

    def run_all(self):
        self.status.setText("⏳ Running models... Please wait...")
        self.btn_run.setEnabled(False)
        self.img_depth = None; self.img_pose = None
        
        # Chain the workers
        self.worker_d = DepthWorker(self.src_path)
        self.worker_d.finished.connect(self.on_depth_done)
        self.worker_d.start()

    def on_depth_done(self, img):
        self.img_depth = img
        self.display(img, self.lbl_depth)
        self.status.setText("✅ Depth Done. Running Pose...")
        
        self.worker_p = PoseWorker(self.src_path)
        self.worker_p.finished.connect(self.on_pose_done)
        self.worker_p.start()

    def on_pose_done(self, img):
        self.img_pose = img
        self.display(img, self.lbl_pose)
        self.create_overlay()
        self.btn_save.setEnabled(True); self.btn_run.setEnabled(True)
        self.status.setText("✅ All Extractions Complete.")

    def create_overlay(self):
        if self.img_depth and self.img_pose:
            # 1. Base is Depth
            depth = self.img_depth.convert("RGBA")
            target_size = depth.size
            
            # 2. Resize Pose to match Depth
            pose = self.img_pose.convert("RGBA")
            if pose.size != target_size:
                pose = pose.resize(target_size, Image.Resampling.BILINEAR)
            
            # 3. Make Pose Transparent (Non-Transposed)
            data = np.array(pose)
            
            # Extract channels safely
            red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
            
            # Define Black (0,0,0)
            black_areas = (red == 0) & (green == 0) & (blue == 0)
            
            # Set Alpha to 0 where black
            data[:,:,3][black_areas] = 0
            
            pose_transparent = Image.fromarray(data)
            
            # 4. Composite
            overlay = Image.alpha_composite(depth, pose_transparent)
            self.display(overlay, self.lbl_combo)
            self.tabs.setCurrentIndex(3)

    def display(self, img_data, label):
        if isinstance(img_data, str): pix = QPixmap(img_data)
        elif img_data is None: return
        else:
            if img_data.mode != "RGBA": img_data = img_data.convert("RGBA")
            data = img_data.tobytes("raw", "RGBA")
            qim = QImage(data, img_data.width, img_data.height, QImage.Format.Format_RGBA8888)
            pix = QPixmap.fromImage(qim)
        label.setPixmap(pix.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def save_all(self):
        if not self.src_path: 
            return

        # 1. Define the target directory
        # Using 'r' for raw string to handle Windows backslashes correctly
        save_dir = r"C:\Users\USER\OneDrive\Desktop\controlnet"

        try:
            # 2. Create the folder if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            base_name = os.path.splitext(os.path.basename(self.src_path))[0]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 3. Save Depth Map
            if self.img_depth:
                depth_path = os.path.join(save_dir, f"{base_name}_depth_{ts}.png")
                self.img_depth.save(depth_path)
            
            # 4. Save Pose Map
            if self.img_pose:
                # Save pose resized to match original image/depth map
                pose_to_save = self.img_pose.resize(self.img_depth.size, Image.Resampling.LANCZOS)
                pose_path = os.path.join(save_dir, f"{base_name}_pose_{ts}.png")
                pose_to_save.save(pose_path)
                
            self.status.setText(f"✅ Batch Saved to: Desktop/controlnet")
            
        except Exception as e:
            self.status.setText(f"❌ Save Error: {str(e)}")
            print(f"Save Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv); window = ControlStudio(); window.show(); sys.exit(app.exec())