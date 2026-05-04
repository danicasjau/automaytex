
# mayautotex - auto texturing for maya - pipline integration - GUI
# daniel casadevall jauhiainen

""" IMPORTING LIBRARIES """
import sys
import os
import psutil
import ctypes

## PYSIDE IMPORTS

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QCheckBox,
        QFrame, QFileDialog, QSpinBox, QTextEdit
    )
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QPainter, QColor, QPen, QFont
except ImportError:
    from PySide2.QtWidgets import (  # type: ignore
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QCheckBox,
        QFrame, QFileDialog, QSpinBox, QTextEdit
    )
    from PySide2.QtCore import Qt, QTimer, Signal  # type: ignore
    from PySide2.QtGui import QFont, QColor, QPainter, QPen  # type: ignore


## PIPELINE IMPORTS
from config import configuration
import backServer as bk


def get_nvml_vram():
    try:
        nvml = ctypes.CDLL(r"C:\Windows\System32\nvml.dll")
        nvml.nvmlInit_v2()
        handle = ctypes.c_void_p()
        nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
        
        class struct_c_nvmlMemory(ctypes.Structure):
            _fields_ = [
                ('total', ctypes.c_ulonglong),
                ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong),
            ]
            
        mem = struct_c_nvmlMemory()
        nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
        nvml.nvmlShutdown()
        
        return mem.used / (1024**2), mem.total / (1024**2)
    except Exception:
        return 0, 0


# -----------------------------
# MAIN GUI CLASS
# -----------------------------

class automaytexGUI(QMainWindow):
    texturize_signal = Signal(object)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AutomaytexMaya")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("Fusion")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        self.reference_images = []
        self.server_process = None
        
        self.populate()

    def set_callback(self, callback):
        self.texturize_callback = callback

    def populate(self):
        main_widget = QWidget()

        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8) 
        main_layout.setSpacing(6)
        
        # Left panel - Main settings
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 1)

        # Right panel - Images and LoRa
        settings_panel = self._create_settings_panel()
        right_panel.addWidget(settings_panel, 1)

        # Model panel
        model_panel = self._create_model_panel()
        right_panel.addWidget(model_panel, 1)
    

    def _create_model_panel(self):
        panel = QFrame()
        panel.setFixedHeight(400)
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --------------------------------------
        # RAM & VRAM SQUARES + NUMBER LABELS
        # --------------------------------------
        usage_row = QVBoxLayout()

        # TOP ROW = SQUARE BARS
        square_row = QHBoxLayout()
        self.ram_square = SquareUsage((255, 150, 40))     # orange
        self.vram_square = SquareUsage((40, 130, 255))    # blue

        square_row.addWidget(QLabel("RAM"))
        square_row.addWidget(self.ram_square)
        square_row.addSpacing(10)
        square_row.addWidget(QLabel("VRAM"))
        square_row.addWidget(self.vram_square)

        usage_row.addLayout(square_row)

        # BOTTOM ROW = NUMERIC VALUES
        numeric_row = QHBoxLayout()

        self.ram_label = QLabel("RAM: 0 / 0 GB")
        self.ram_label.setStyleSheet("color: black;")
        numeric_row.addWidget(self.ram_label)

        numeric_row.addSpacing(20)

        self.vram_label = QLabel("VRAM: 0 / 0 GB")
        self.vram_label.setStyleSheet("color: black;")
        numeric_row.addWidget(self.vram_label)

        usage_row.addLayout(numeric_row)

        layout.addLayout(usage_row)

        # --------------------------------------
        # MEMORY GRAPH
        # --------------------------------------
        self.graph = MemoryGraph()
        self.graph.setFixedHeight(50)
        layout.addWidget(self.graph)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_usage)
        self.timer.start(100)

        # --------------------------------------
        # Model Type
        # --------------------------------------
        layout.addWidget(QLabel("Model Type"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["sdxl", "fast_sdxl", "flash_sdxl"])
        layout.addWidget(self.model_combo)

        # --------------------------------------
        # System Preferred
        # --------------------------------------
        layout.addWidget(QLabel("Preferred System"))
        self.system_combo = QComboBox()
        self.system_combo.addItems(["cpu", "gpu", "both"])
        layout.addWidget(self.system_combo)

        # --------------------------------------
        # Quantization
        # --------------------------------------
        layout.addWidget(QLabel("Quantization"))
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["None", "fp16", "int8", "int4", "bf16", "fp32"])
        layout.addWidget(self.quant_combo)

        # --------------------------------------
        # Load button
        # --------------------------------------
        self.start_server = QPushButton("Start Server")
        self.start_server.clicked.connect(self._start_server)
        layout.addWidget(self.start_server)

        layout.addSpacing(10)

        self.load_button = QPushButton("Load Models")
        self.load_button.clicked.connect(self._load_models)
        layout.addWidget(self.load_button)

        self.unload_button = QPushButton("Unload Models")
        self.unload_button.clicked.connect(self._unload_models)
        layout.addWidget(self.unload_button)

        layout.addStretch()
        return panel
    
    # SYSTEM UPDATE
    def update_usage(self):
        # -------------------------------
        # RAM
        # -------------------------------
        v = psutil.virtual_memory()
        ram_used = v.used / (1024**3)
        ram_total = v.total / (1024**3)
        ram_percent = v.percent

        self.ram_square.set_value(ram_percent)
        self.ram_label.setText(f"RAM: {ram_used:.1f} / {ram_total:.1f} GB")

        # -------------------------------
        # VRAM (GPU)
        # -------------------------------
        vram_used, vram_total = get_nvml_vram()
        vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0

        self.vram_square.set_value(vram_percent)
        self.vram_label.setText(f"VRAM: {vram_used / 1024:.1f} / {vram_total / 1024:.1f} GB")

    # EXTERNAL LIB CALL
    def _load_models(self):
        bk._load_all_models()
        print("Models loaded")

    def _unload_models(self):
        bk._unload_all_models()
        print("Models unloaded")

    def _start_server(self):
        self.server_process = bk.start_server()
        print(self.server_process)


    def _create_left_panel(self):
        """Create left panel with main settings and generate button."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        panel.setLineWidth(1)
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Title
        title = QLabel("automaytex")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # material name
        layout.addWidget(QLabel("// Material Name: "))
        self.material_name_input = QLineEdit()
        self.material_name_input.setText("material01")
        self.material_name_input.setPlaceholderText("Enter material name...")
        layout.addWidget(self.material_name_input)

        
        # Separator after title
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setLineWidth(1)
        layout.addWidget(separator1)
        
        # Steps Slider
        layout.addWidget(QLabel("Steps"))
        steps_layout = QHBoxLayout()
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setMinimum(1)
        self.steps_slider.setMaximum(100)
        self.steps_slider.setValue(20)
        self.steps_label = QLabel("20")
        self.steps_label.setMaximumWidth(40)
        self.steps_slider.valueChanged.connect(lambda v: self.steps_label.setText(str(v)))
        steps_layout.addWidget(self.steps_slider)
        steps_layout.addWidget(self.steps_label)
        layout.addLayout(steps_layout)

                # CFG Slider
        layout.addWidget(QLabel("CFG Scale"))
        cfg_layout = QHBoxLayout()
        self.cfg_slider = QSlider(Qt.Horizontal)
        self.cfg_slider.setMinimum(1)
        self.cfg_slider.setMaximum(160)
        self.cfg_slider.setValue(7)
        self.cfg_label = QLabel("7.0")
        self.cfg_label.setMaximumWidth(40)
        self.cfg_slider.valueChanged.connect(lambda v: self.cfg_label.setText(f"{v / 10.0:.1f}"))
        cfg_layout.addWidget(self.cfg_slider)
        cfg_layout.addWidget(self.cfg_label)
        layout.addLayout(cfg_layout)

                # Reference scale Slider
        layout.addWidget(QLabel("Reference semblance Scale"))
        rif_layout = QHBoxLayout()
        self.rif_slider = QSlider(Qt.Horizontal)
        self.rif_slider.setMinimum(1)
        self.rif_slider.setMaximum(30)
        self.rif_slider.setValue(1)
        self.rif_label = QLabel("1.0")
        self.rif_label.setMaximumWidth(40)
        self.rif_slider.valueChanged.connect(lambda v: self.rif_label.setText(f"{v / 10.0:.1f}"))
        rif_layout.addWidget(self.rif_slider)
        rif_layout.addWidget(self.rif_label)
        layout.addLayout(rif_layout)
        
        # Noise Slider
        layout.addWidget(QLabel("Noise"))
        noise_layout = QHBoxLayout()
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(0)
        self.noise_label = QLabel("0.0")
        self.noise_label.setMaximumWidth(40)
        self.noise_slider.valueChanged.connect(lambda v: self.noise_label.setText(f"{v / 100.0:.2f}"))
        noise_layout.addWidget(self.noise_slider)
        noise_layout.addWidget(self.noise_label)
        layout.addLayout(noise_layout)


        # Separator before generated images
        separator4 = QFrame()
        separator4.setFrameShape(QFrame.HLine)
        separator4.setLineWidth(1)
        layout.addWidget(separator4)
        
        # Save Path
        layout.addWidget(QLabel("// Texture Output Save Path: "))
        save_path_layout = QHBoxLayout()
        self.save_path_input = QLineEdit()
        self.save_path_input.setReadOnly(True)
        self.save_path_btn = QPushButton("Browse...")
        self.save_path_btn.clicked.connect(self._select_save_path)
        save_path_layout.addWidget(self.save_path_input)
        save_path_layout.addWidget(self.save_path_btn)
        layout.addLayout(save_path_layout)

        # Render Type
        layout.addWidget(QLabel("// Render Type: "))
        render_layout = QHBoxLayout()
        self.render_combo = QComboBox()
        self.render_combo.addItems(["cube", "thetredral"])
        render_layout.addWidget(self.render_combo)
        layout.addLayout(render_layout)

        # Material Type
        layout.addWidget(QLabel("// Texture size: "))
        self.texture_combo = QComboBox()
        self.texture_combo.addItems(["512", "1024", "2048", "4096"])
        layout.addWidget(self.texture_combo)

        self.retarget_uv_check = QCheckBox("Retarget UV")
        self.retarget_uv_check.setChecked(True)
        layout.addWidget(self.retarget_uv_check)

        # Material Type
        layout.addWidget(QLabel("// Maya Material Type: "))
        self.material_combo = QComboBox()
        self.material_combo.addItems(["mtlx", "standard surface", "arnold aiStandard", "lambert"])
        layout.addWidget(self.material_combo)

        # Assign Maya Material checkbox
        self.assign_maya_check = QCheckBox("Auto Assign maya material")
        self.assign_maya_check.setChecked(True)
        layout.addWidget(self.assign_maya_check)

        self.height_scale = QSpinBox()
        self.height_scale.setMinimum(1)
        self.height_scale.setMaximum(10)
        self.height_scale.setValue(1)
        self.height_scale.setPrefix("Height Map Scale: ")
        layout.addWidget(self.height_scale)
        
        # Separator before generated images
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setLineWidth(1)
        layout.addWidget(separator3)
        
        # Image generation checkboxes
        layout.addWidget(QLabel("// MAPS to GENERATE: "))

        imageGen_layout = QHBoxLayout()

        self.diffuse_check = QCheckBox("Diffuse")
        self.diffuse_check.setChecked(True)
        imageGen_layout.addWidget(self.diffuse_check)
        
        self.roughness_check = QCheckBox("Roughness")
        self.roughness_check.setChecked(True)
        imageGen_layout.addWidget(self.roughness_check)
        
        self.metalness_check = QCheckBox("Metalness")
        self.metalness_check.setChecked(True)
        imageGen_layout.addWidget(self.metalness_check)
        
        self.normal_check = QCheckBox("Normal")
        self.normal_check.setChecked(False)
        imageGen_layout.addWidget(self.normal_check)
        
        self.height_check = QCheckBox("Height")
        self.height_check.setChecked(False)
        imageGen_layout.addWidget(self.height_check)
        
        layout.addLayout(imageGen_layout)
        
        # Spacer
        layout.addStretch()

        # Texturize button
        self.texturize_btn = QPushButton("Texturize")
        btn_font = QFont()
        btn_font.setPointSize(11)
        btn_font.setBold(True)
        self.texturize_btn.setFont(btn_font)
        self.texturize_btn.setMinimumHeight(40)
        self.texturize_btn.clicked.connect(self._on_texturize)
        layout.addWidget(self.texturize_btn)
        
        return panel
    
    def _create_settings_panel(self):
        """Create right panel with reference images and LoRa settings."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        panel.setLineWidth(1)
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Prompt
        layout.addWidget(QLabel("Positive Prompt"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter texture prompt...")
        layout.addWidget(self.prompt_input)
        
        # Prompt negative
        layout.addWidget(QLabel("Negative Prompt"))
        self.n_prompt_input = QTextEdit()
        self.n_prompt_input.setPlaceholderText("Enter negative texture prompt...")
        layout.addWidget(self.n_prompt_input)

        # Reference Images
        layout.addWidget(QLabel("Reference Images"))
        ref_layout = QVBoxLayout()
        self.ref_list_label = QLabel("No images selected")
        self.ref_list_label.setWordWrap(True)
        self.ref_list_label.setStyleSheet("background-color: #222222; padding: 5px; border-radius: 3px;")
        ref_layout.addWidget(self.ref_list_label)
        
        ref_btn_layout = QHBoxLayout()
        ref_add_btn = QPushButton("+")
        ref_add_btn.setMaximumWidth(40)
        ref_add_btn.clicked.connect(self._add_reference_image)
        ref_remove_btn = QPushButton("X")
        ref_remove_btn.setMaximumWidth(40)
        ref_remove_btn.clicked.connect(self._remove_reference_image)
        ref_btn_layout.addWidget(ref_add_btn)
        ref_btn_layout.addWidget(ref_remove_btn)
        ref_btn_layout.addStretch()
        ref_layout.addLayout(ref_btn_layout)
        layout.addLayout(ref_layout)
        
        # Spacer
        layout.addStretch()
        
        return panel
    
    def _select_save_path(self) -> None:
        """Open dialog to select save path."""
        path = QFileDialog.getExistingDirectory(self, "Select Save Path")
        if path:
            self.save_path = path
            self.save_path_input.setText(path)
    
    def _add_reference_image(self) -> None:
        """Add reference image(s) to list."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Reference Images",
            "",
            "Image Files (*.png *.exr *.jpg);;All Files (*)"
        )
        for file in files:
            if file not in self.reference_images:
                self.reference_images.append(file)
        self._update_ref_display()
    
    def _remove_reference_image(self) -> None:
        """Remove selected reference image from list."""
        if self.reference_images:
            self.reference_images.pop()
            self._update_ref_display()
    
    def _update_ref_display(self) -> None:
        """Update reference images display label."""
        if self.reference_images:
            display_text = "\n".join([os.path.basename(p) for p in self.reference_images])
            self.ref_list_label.setText(display_text)
        else:
            self.ref_list_label.setText("No images selected")
    
    def _add_lora_path(self) -> None:
        """Add LoRa model to list."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select LoRa Models",
            "",
            "LoRa Files (*.gguf *.safetensors);;All Files (*)"
        )
        for file in files:
            if file not in self.lora_paths:
                self.lora_paths.append(file)
        self._update_lora_display()
    
    def _remove_lora_path(self) -> None:
        """Remove selected LoRa model from list."""
        if self.lora_paths:
            self.lora_paths.pop()
            self._update_lora_display()
    
    def _update_lora_display(self) -> None:
        """Update LoRa display label."""
        if self.lora_paths:
            display_text = "\n".join([os.path.basename(p) for p in self.lora_paths])
            self.lora_list_label.setText(display_text)
        else:
            self.lora_list_label.setText("No LoRa selected")
    
    def _get_system_info(self) -> str:
        """Get current system information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            
            info = f"CPU: {cpu_percent}% | RAM: {ram_percent}% | Est. Gen Time: ~2-5 min"
            return info
        except Exception as e:
            return "System info unavailable"
    
    def _on_texturize(self):
        settings = self.extract_generation_settings()

        self.texturize_signal.emit(settings)

    def extract_generation_settings(self):
        dConf = configuration()

        dConf.positive_prompt = self.prompt_input.toPlainText()
        dConf.negative_prompt = self.n_prompt_input.toPlainText()
        dConf.texture_resolution = self.texture_combo.currentText()
        dConf.inference_steps = self.steps_slider.value()

        dConf.cfg_scale = self.cfg_slider.value() / 10.0
        dConf.noise = self.noise_slider.value() / 100.0
        dConf.camera_scale = 1

        dConf.seed = 123456789  # could add a random seed generator or input field
        dConf.generated_images = []

        if self.diffuse_check.isChecked():
            dConf.generated_images.append("diffuse")
        if self.roughness_check.isChecked():
            dConf.generated_images.append("roughness")
        if self.metalness_check.isChecked():
            dConf.generated_images.append("metalness")
        if self.normal_check.isChecked():
            dConf.generated_images.append("normal")
        if self.height_check.isChecked():
            dConf.generated_images.append("height")

        dConf.retargetUV = self.retarget_uv_check.isChecked()
        
        dConf.base_model = self.model_combo.currentText().lower()
        dConf.system_prfered = self.system_combo.currentText()
        quant_text = self.quant_combo.currentText()

        dConf.quantization = quant_text if quant_text != "None" else None
        dConf.assign_maya_material = self.assign_maya_check.isChecked()
        dConf.material_type = self.material_combo.currentText()
        dConf.output_path = self.save_path_input.text() if self.save_path_input.text() else dConf.output_path
        
        dConf.renderMode = self.render_combo.currentText()

        dConf.material_name = self.material_name_input.text()

        dConf.pathsetter()
        # manual validation method to handle correctly data.
        # dConf.validate()
        
        return dConf

# -----------------------------
# SQUARE USAGE BAR
# -----------------------------
class SquareUsage(QWidget):
    def __init__(self, color):
        super().__init__()
        self.value = 0
        self.color = QColor(*color)
        self.setFixedSize(250, 40)

    def set_value(self, v):
        self.value = max(0, min(v, 100))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        h = int(self.height() * (self.value / 100))
        painter.fillRect(0, self.height() - h, self.width(), h, self.color)

# -----------------------------
# RAM / VRAM GRAPH WIDGET
# -----------------------------
class MemoryGraph(QWidget):
    def __init__(self, max_points=60):
        super().__init__()
        self.max_points = max_points
        self.ram_history = []
        self.vram_history = []

        # 20 FPS timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(50)

    def update_data(self):
        ram = psutil.virtual_memory().percent
        # GPU VRAM fallback to 0% if no GPU libs available
        vram_used, vram_total = get_nvml_vram()
        vram = (vram_used / vram_total * 100) if vram_total > 0 else 0

        self.ram_history.append(ram)
        self.vram_history.append(vram)

        if len(self.ram_history) > self.max_points:
            self.ram_history.pop(0)
            self.vram_history.pop(0)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if len(self.ram_history) < 2:
            return

        w = self.width()
        h = self.height()

        step = w / (self.max_points - 1)

        pen_ram = QPen(QColor(255, 150, 40), 2)
        pen_vram = QPen(QColor(40, 130, 255), 2)

        # Draw RAM
        painter.setPen(pen_ram)
        for i in range(len(self.ram_history) - 1):
            p1 = (i * step, h - (self.ram_history[i] * h / 100))
            p2 = ((i + 1) * step, h - (self.ram_history[i + 1] * h / 100))
            painter.drawLine(*p1, *p2)

        # Draw VRAM
        painter.setPen(pen_vram)
        for i in range(len(self.vram_history) - 1):
            p1 = (i * step, h - (self.vram_history[i] * h / 100))
            p2 = ((i + 1) * step, h - (self.vram_history[i + 1] * h / 100))
            painter.drawLine(*p1, *p2)


def main():
    app = QApplication(sys.argv)
    window = automaytexGUI()
    window.show()
    sys.exit(app.exec())