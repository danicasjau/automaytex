try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QCheckBox,
        QFrame, QFileDialog, QSpinBox
    )
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QColor
except ImportError:
    from PySide2.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QCheckBox,
        QFrame, QFileDialog, QSpinBox
    )
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QFont, QColor

from dataclasses import dataclass
from typing import List, Optional
import os

try:
    import psutil
except:
    pass


@dataclass
class Settings:
    generated_images: List[str]  # ["diffuse", "roughness", "metalness", "normal", "height"]
    material_type: str  # "mtlx" or "standard"
    model: str  # "flux", "sdxl", "sd1.5"
    prompt: str
    reference_images: List[str]
    lora_paths: List[str]
    cfg: float
    steps: int
    save_path: str
    noise: float
    assign_maya_material: bool
    system_preferred: str  # "cpu", "gpu", or "both"


class AutomaytexGUI(QMainWindow):
    """Main GUI window for AutomaytexMaya texture generation tool."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutomaytexMaya")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("Fusion")
        
        # Data storage
        self.reference_images = []
        self.lora_paths = []
        self.save_path = ""
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10) 
        main_layout.setSpacing(15)
        
        # Left panel - Main settings
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Images and LoRa
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)

    
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
        
        # Separator after title
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setLineWidth(1)
        layout.addWidget(separator1)
        
        # Model selection
        layout.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["flux", "sdxl", "sd1.5"])
        layout.addWidget(self.model_combo)
        
        # CFG Slider
        layout.addWidget(QLabel("CFG Scale"))
        cfg_layout = QHBoxLayout()
        self.cfg_slider = QSlider(Qt.Horizontal)
        self.cfg_slider.setMinimum(1)
        self.cfg_slider.setMaximum(30)
        self.cfg_slider.setValue(7)
        self.cfg_label = QLabel("7.0")
        self.cfg_label.setMaximumWidth(40)
        self.cfg_slider.valueChanged.connect(lambda v: self.cfg_label.setText(f"{v / 10.0:.1f}"))
        cfg_layout.addWidget(self.cfg_slider)
        cfg_layout.addWidget(self.cfg_label)
        layout.addLayout(cfg_layout)
        
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
        
        # Separator after sliders
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setLineWidth(1)
        layout.addWidget(separator2)
        
        # Material Type
        layout.addWidget(QLabel("Material Type"))
        self.material_combo = QComboBox()
        self.material_combo.addItems(["mtlx", "standard"])
        layout.addWidget(self.material_combo)
        
        # Save Path
        layout.addWidget(QLabel("Save Path"))
        save_path_layout = QHBoxLayout()
        self.save_path_input = QLineEdit()
        self.save_path_input.setReadOnly(True)
        self.save_path_btn = QPushButton("Browse...")
        self.save_path_btn.clicked.connect(self._select_save_path)
        save_path_layout.addWidget(self.save_path_input)
        save_path_layout.addWidget(self.save_path_btn)
        layout.addLayout(save_path_layout)
        
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
        
        # System Preferred
        layout.addWidget(QLabel("System Preferred"))
        self.system_combo = QComboBox()
        self.system_combo.addItems(["cpu", "gpu", "both (offloading)"])
        self.system_combo.setCurrentText("gpu")
        layout.addWidget(self.system_combo)
        
        # Assign Maya Material checkbox
        self.assign_maya_check = QCheckBox("Assign maya material")
        self.assign_maya_check.setChecked(True)
        layout.addWidget(self.assign_maya_check)
        
        # Separator before generated images
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setLineWidth(1)
        layout.addWidget(separator3)
        
        # Image generation checkboxes
        layout.addWidget(QLabel("Generate Images"))
        self.diffuse_check = QCheckBox("Diffuse")
        self.diffuse_check.setChecked(True)
        layout.addWidget(self.diffuse_check)
        
        self.roughness_check = QCheckBox("Roughness")
        self.roughness_check.setChecked(True)
        layout.addWidget(self.roughness_check)
        
        self.metalness_check = QCheckBox("Metalness")
        self.metalness_check.setChecked(True)
        layout.addWidget(self.metalness_check)
        
        self.normal_check = QCheckBox("Normal")
        self.normal_check.setChecked(False)
        layout.addWidget(self.normal_check)
        
        self.height_check = QCheckBox("Height")
        self.height_check.setChecked(False)
        layout.addWidget(self.height_check)
        
        # Spacer
        layout.addStretch()
        
        # System info and generate button
        layout.addWidget(QLabel("System Info"))
        self.system_info_label = QLabel(self._get_system_info())
        self.system_info_label.setWordWrap(True)
        system_font = QFont()
        system_font.setPointSize(9)
        self.system_info_label.setFont(system_font)
        layout.addWidget(self.system_info_label)
        
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
    
    def _create_right_panel(self):
        """Create right panel with reference images and LoRa settings."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        panel.setLineWidth(1)
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Settings section with max width constraint
        settings_label = QLabel("Settings")
        settings_label.setMaximumHeight(100)
        layout.addWidget(settings_label)
        
        # Prompt
        layout.addWidget(QLabel("Prompt"))
        self.prompt_input = QLineEdit()
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
        
        # LoRa
        layout.addWidget(QLabel("LoRa Models"))
        lora_layout = QVBoxLayout()
        self.lora_list_label = QLabel("No LoRa selected")
        self.lora_list_label.setWordWrap(True)
        self.lora_list_label.setStyleSheet("background-color: #222222; padding: 5px; border-radius: 3px;")
        lora_layout.addWidget(self.lora_list_label)
        
        lora_btn_layout = QHBoxLayout()
        lora_add_btn = QPushButton("+")
        lora_add_btn.setMaximumWidth(40)
        lora_add_btn.clicked.connect(self._add_lora_path)
        lora_remove_btn = QPushButton("X")
        lora_remove_btn.setMaximumWidth(40)
        lora_remove_btn.clicked.connect(self._remove_lora_path)
        lora_btn_layout.addWidget(lora_add_btn)
        lora_btn_layout.addWidget(lora_remove_btn)
        lora_btn_layout.addStretch()
        lora_layout.addLayout(lora_btn_layout)
        layout.addLayout(lora_layout)
        
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
    
    def _on_texturize(self) -> None:
        """Handle texturize button click."""
        settings = self.get_settings()
        # Backend logic would be called here
        print(f"Texturize clicked with settings: {settings}")
    
    def get_settings(self) -> Settings:
        """Get current settings as Settings dataclass."""
        generated_images = []
        if self.diffuse_check.isChecked():
            generated_images.append("diffuse")
        if self.roughness_check.isChecked():
            generated_images.append("roughness")
        if self.metalness_check.isChecked():
            generated_images.append("metalness")
        if self.normal_check.isChecked():
            generated_images.append("normal")
        if self.height_check.isChecked():
            generated_images.append("height")
        
        return Settings(
            generated_images=generated_images,
            material_type=self.material_combo.currentText(),
            model=self.model_combo.currentText(),
            prompt=self.prompt_input.text(),
            reference_images=self.reference_images,
            lora_paths=self.lora_paths,
            cfg=self.cfg_slider.value() / 10.0,
            steps=self.steps_slider.value(),
            save_path=self.save_path,
            noise=self.noise_slider.value() / 100.0,
            assign_maya_material=self.assign_maya_check.isChecked(),
            system_preferred=self.system_combo.currentText()
        )


def main():
    """Main entry point for the application."""
    import sys
    app = QApplication(sys.argv)
    window = AutomaytexGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()