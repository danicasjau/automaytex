
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
        QFrame, QFileDialog, QSpinBox, QTextEdit, QToolButton, QSizePolicy, 
        QProgressBar
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QSize
    from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QIcon
except ImportError:
    from PySide2.QtWidgets import (  # type: ignore
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QCheckBox,
        QFrame, QFileDialog, QSpinBox, QTextEdit, QToolButton, QSizePolicy
    )
    from PySide2.QtCore import Qt, QTimer, Signal  # type: ignore
    from PySide2.QtGui import QFont, QColor, QPainter, QPen, QPixmap, QIcon  # type: ignore


## PIPELINE IMPORTS
from config import configuration
import backServer as bk
import advancedsettings


# -----------------------------
# MAIN GUI CLASS
# -----------------------------

class automaytexGUI(QMainWindow):
    texturize_signal = Signal(object)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AutomaytexMaya")
        self.setGeometry(100, 100, 400, 700)
        self.setStyleSheet("Fusion")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        self.reference_images = []
        self.server_process = None

        # Create a single persistent AdvancedSettings instance
        self.adv = advancedsettings.AdvancedSettings()
        
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
    

    def toggle_settings(self, checked, settings, btn, name):
        settings.setVisible(checked)
        btn.setText(f"{'▼' if checked else '▶'} {name}")

    # Helper to create synced Slider + Entry
    def create_slider_entry(self, label_text, min_v, max_v, default_v, divisor=1.0, is_int=False, box=None):
        box.addWidget(QLabel(label_text))
        row_layout = QHBoxLayout()
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_v)
        slider.setMaximum(max_v)
        slider.setValue(default_v)
        
        entry = QLineEdit()
        entry.setFixedWidth(45)
        
        def update_entry(v):
            val = str(v) if is_int else f"{v / divisor:.1f}"
            if divisor == 100.0: val = f"{v / divisor:.2f}"
            entry.setText(val)

        def update_slider():
            try:
                val = float(entry.text())
                slider.setValue(int(val if is_int else val * divisor))
            except ValueError:
                pass

        update_entry(default_v)
        slider.valueChanged.connect(update_entry)
        entry.editingFinished.connect(update_slider)
        
        row_layout.addWidget(slider)
        row_layout.addWidget(entry)
        box.addLayout(row_layout)
        return slider, entry

    def _add_separator(self, layout):
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

    def _create_top_logo(self, layout):
        # 1. Logo Setup
        logo = QLabel()
        logo_pixmap = QPixmap(fr"{os.environ['BASE_DIR']}/documentation/data/logo.png")

        # Optional: Scale the logo if it's too large, maintaining aspect ratio
        logo.setPixmap(logo_pixmap.scaled(260, 260, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo)

        # 2. Title Setup
        title = QLabel("automaytex")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 3. Version and Author (Subtitle Style)
        metadata = QLabel("Version 2.1.0 | Author: Daniel Casadevall")
        metadata_font = QFont()
        metadata_font.setPointSize(9)
        metadata_font.setItalic(True)
        metadata.setFont(metadata_font)
        metadata.setStyleSheet("color: gray;") # Adds a nice subtle touch
        metadata.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(metadata)

    def _create_settings_header(self, name, layout):
        # --- Fast Settings Collapsible Layout ---
        fast_settings_btn = QToolButton()
        fast_settings_btn.setText(f"▼ {name}")
        fast_settings_btn.setCheckable(True)
        fast_settings_btn.setChecked(True)
        fast_settings_btn.setStyleSheet("border: 2px solid black; font-weight: bold; font-size: 14px; border-radius: 5px; ")

        fast_settings_container = QWidget()
        fast_settings_vbox = QVBoxLayout(fast_settings_container)
        layout.addWidget(fast_settings_btn)
        layout.addWidget(fast_settings_container)
        fast_settings_btn.clicked.connect(lambda: self.toggle_settings(fast_settings_btn.isChecked(), fast_settings_container, fast_settings_btn, name))

        return fast_settings_vbox

    def _create_left_panel(self):
        """Create left panel with main settings and generate button."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        panel.setLineWidth(1)
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Title
        self._create_top_logo(layout)

        self._add_separator(layout)
        

        # material name
        inputlayout = QHBoxLayout()
        inputlayout.addWidget(QLabel(">>> Material Name: "))
        self.material_name_input = QLineEdit()
        self.material_name_input.setText("emptyMaterial")
        self.material_name_input.setPlaceholderText("Enter material name...")
        inputlayout.addWidget(self.material_name_input)

        layout.addLayout(inputlayout)

        # Image generation checkboxes
        layout.addWidget(QLabel(">>> Maps to generate: "))
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

        # Save Path
        save_path_layout = QHBoxLayout()
        save_path_layout.addWidget(QLabel(">>> Output Path: "))
        self.save_path_input = QLineEdit()
        self.save_path_input.setReadOnly(True)

        self.save_path_btn = QPushButton("")
        self.save_path_btn.setIcon(QIcon(fr"{os.environ['BASE_DIR']}/data/icons/folder.png"))
        self.save_path_btn.setIconSize(QSize(25, 25))

        self.save_path_btn.clicked.connect(self._select_save_path)
        save_path_layout.addWidget(self.save_path_input)
        save_path_layout.addWidget(self.save_path_btn)
        layout.addLayout(save_path_layout)

        # Separator after title
        self._add_separator(layout)
                
        self.model_settings = self._create_settings_header(name="MODEL SETTINGS", layout=layout)
        # Implementation

        starttex = ">>> "
        hlay = QHBoxLayout()
        hlay.addWidget(QLabel(starttex + "Model Type"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["sdxl", "fast_sdxl", "sd1.5", "flux"])
        hlay.addWidget(self.model_combo)
        self.model_settings.addLayout(hlay)

        hlay = QHBoxLayout()
        hlay.addWidget(QLabel(starttex + "Preferred System"))
        self.system_combo = QComboBox()
        self.system_combo.addItems(["gpu", "cpu", "both (offloading)"])
        hlay.addWidget(self.system_combo)
        self.model_settings.addLayout(hlay)

        hlay = QHBoxLayout()
        hlay.addWidget(QLabel(starttex + "Quantization"))
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["fp16", "int8", "int4", "bf16", "fp32"])
        hlay.addWidget(self.quant_combo)
        self.model_settings.addLayout(hlay)

        # Advanced Settings
        advanced_settings_btn = QToolButton()
        advanced_settings_btn.setText("Advanced Model Settings")
        
        advanced_settings_btn.clicked.connect(self.show_advanced_settings)
        self.model_settings.addWidget(advanced_settings_btn)

        self._add_separator(layout)

        self.fast_settings_vbox = self._create_settings_header(name="GENERATION SETTINGS", layout=layout)
        # Implementation
        self.steps_slider, self.steps_entry = self.create_slider_entry("Steps", min_v=1, max_v=100, default_v=20, is_int=True, box=self.fast_settings_vbox)
        self.cfg_slider, self.cfg_entry = self.create_slider_entry("CFG Scale", min_v=1, max_v=160, default_v=70, divisor=10.0, box=self.fast_settings_vbox)
        self.rif_slider, self.rif_entry = self.create_slider_entry("Reference semblance Scale", min_v=1, max_v=30, default_v=10, divisor=10.0, box=self.fast_settings_vbox)
        self.noise_slider, self.noise_entry = self.create_slider_entry("Noise", min_v=0, max_v=100, default_v=0, divisor=100.0, box=self.fast_settings_vbox)


        # Separator before generated images
        self._add_separator(layout)
        


        # Material Type
        layout.addWidget(QLabel(">>> Texture size: "))
        self.texture_combo = QComboBox()
        self.texture_combo.addItems(["512", "1024", "2048", "4096", "8192"])
        layout.addWidget(self.texture_combo)


        self._add_separator(layout)


        self.progress = QProgressBar()
        self.progress.setFixedHeight(20)
        self.progress.setStyleSheet("QProgressBar { border-radius: 4px; text-align: center; } "
                          "QProgressBar::chunk { background-color: #d67129; border-radius: 4px; }")
        self.progress.setValue(100)
        layout.addWidget(self.progress)

        
        # Spacer
        layout.addStretch()

        # Texturize button
        self.texturize_btn = QPushButton("Generate Textures")
        btn_font = QFont()
        btn_font.setPointSize(11)
        btn_font.setBold(True)
        self.texturize_btn.setFont(btn_font)
        self.texturize_btn.setMinimumHeight(40)
        self.texturize_btn.clicked.connect(self._on_texturize)
        layout.addWidget(self.texturize_btn)

        self.finalinfo = QLabel("automaytex - v1.2.0 - 2026 - Daniel Casadevall Jauhiainen - La Salle")
        self.finalinfo.setStyleSheet("color: gray")
        self.finalinfo.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.finalinfo)
        
        return panel
    
    def show_advanced_settings(self):
        """Show the persistent AdvancedSettings dialog."""
        self.adv.show()
        self.adv.raise_()
        self.adv.activateWindow()


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
        dConf = self.adv.extract_generation_settings()

        # --- Override with main GUI values ---
        dConf.positive_prompt   = self.prompt_input.toPlainText() if hasattr(self, 'prompt_input') and self.prompt_input.toPlainText() else dConf.positive_prompt
        dConf.texture_resolution = self.texture_combo.currentText()
        dConf.inference_steps   = self.steps_slider.value()
        dConf.cfg_scale         = self.cfg_slider.value() / 10.0
        dConf.noise             = self.noise_slider.value() / 100.0
        dConf.base_model        = self.model_combo.currentText().lower()
        dConf.system_prfered    = self.system_combo.currentText()
        quant_text              = self.quant_combo.currentText()
        dConf.quantization      = quant_text if quant_text != "None" else None
        dConf.material_name     = self.material_name_input.text()
        dConf.output_path       = self.save_path_input.text() if self.save_path_input.text() else dConf.output_path

        dConf.generated_images = []
        if self.diffuse_check.isChecked():   dConf.generated_images.append("diffuse")
        if self.roughness_check.isChecked(): dConf.generated_images.append("roughness")
        if self.metalness_check.isChecked(): dConf.generated_images.append("metalness")
        if self.normal_check.isChecked():    dConf.generated_images.append("normal")
        if self.height_check.isChecked():    dConf.generated_images.append("height")

        dConf.seed = 123456789

        dConf.pathsetter()
        return dConf

def main():
    app = QApplication(sys.argv)
    window = automaytexGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()