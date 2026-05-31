# GUI - MayAutoTex
# La gui per a un plugin de maya que autotexturitza meshes.
# daniel casadevall jauhiainen

""" IMPORTING LIBRARIES """
import sys
import os
import psutil

## PYSIDE IMPORTS

try:
    from PySide6.QtWidgets import ( # type: ignore
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QCheckBox,
        QFrame, QFileDialog, QSpinBox, QTextEdit, QToolButton, QSizePolicy, 
        QProgressBar, QMessageBox
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QSize # type: ignore
    from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QIcon # type: ignore
except ImportError:
    from PySide2.QtWidgets import (  # type: ignore
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QCheckBox,
        QFrame, QFileDialog, QSpinBox, QTextEdit, QToolButton, QSizePolicy, 
        QProgressBar,QMessageBox
    )
    from PySide2.QtCore import Qt, QTimer, Signal  # type: ignore
    from PySide2.QtGui import QFont, QColor, QPainter, QPen, QPixmap, QIcon  # type: ignore

import mlGuiAdvanced

# -----------------------------
# MAIN GUI CLASS
# -----------------------------

class automaytexGUI(QMainWindow):
    # signal to set the texturize command signal. Per començar a texturitzar.
    texturize_signal = Signal(object)

    def __init__(self):
        super().__init__()

        # set inital window configs
        self.setWindowTitle("AutomaytexMaya")
        self.setGeometry(100, 100, 400, 700)
        self.setStyleSheet("Fusion")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # variables dels inputs. Main reference images.
        self.reference_images = []
        self.server_process = None

        # Create a single persistent AdvancedSettings instances. Una altre window, per fer els advanced settings.
        self.adv = mlGuiAdvanced.AdvancedSettings()
        self.adv.main_gui_extract_func = self.extract_generation_settings

        # function populat call to create all elements. 
        self.populate()

    def set_callback(self, callback):
        """
        Funcio per initialitzar i assignar el callback, per texturitzar despres.
        """
        self.texturize_callback = callback

    def populate(self):
        """
        Funcio populate que crea tots els widgets, labels i elements necesaris per la gui.
        """
        
        main_widget = QWidget() # setting main widget window

        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8) 
        main_layout.setSpacing(6)

        # creacio del left panel
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)
        # creacio del right panel
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 1)

    def toggle_settings(self, checked, settings, btn, name):
        """
        Fast function to set and hide settings apartats. Funciona amb un entry de settings, i utilitza .setVisible(bool)
        """
        settings.setVisible(checked)
        btn.setText(f"{'▼' if checked else '▶'} {name}")

    
    def create_slider_entry(self, label_text, min_v, max_v, default_v, divisor=1.0, is_int=False, box=None):
        """
        Funcio utilitat per crear sliders nous
        """
        
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
        """
        Funcio utilitat per afegir un separador rapidament
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

    def _create_top_logo(self, layout):
        # Label per poder colocar el logo
        logo = QLabel()
        logo_pixmap = QPixmap(fr"{os.environ['BASE_DIR']}/documentation/data/logo.png")

        # Scale the logo if it's too large, maintaining aspect ratio
        logo.setPixmap(logo_pixmap.scaled(260, 260, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo)

        # posa el titol de Setup
        title = QLabel("automaytex")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # per colocar el subtitol i mes info
        metadata = QLabel("Version 2.1.0 | Author: Daniel Casadevall")
        metadata_font = QFont()
        metadata_font.setPointSize(9)
        metadata_font.setItalic(True)
        metadata.setFont(metadata_font)
        metadata.setStyleSheet("color: gray;")
        metadata.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(metadata)

    def _create_settings_header(self, name, layout):
        # utilitat per posar el header rapid dels settings. funcio privada
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
        """creacio del panel left amb main settings and generate button."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        panel.setLineWidth(1)
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # Title
        self._create_top_logo(layout)

        self._add_separator(layout)

        # Tips Label. Label per poder donar tips, amb color destacat en style Sheet
        self.tips_label = QLabel("")
        self.tips_label.setStyleSheet(
            "color: #e6b800; 
            font-weight: bold; 
            font-size: 12px; 
            background-color: #333333; 
            padding: 5px; 
            border-radius: 3px;
        ")
        self.tips_label.setWordWrap(True)
        self.tips_label.setVisible(False)
        layout.addWidget(self.tips_label)

        # material name, per poder posar el nom del material, entry en classe
        inputlayout = QHBoxLayout()
        inputlayout.addWidget(QLabel(">>> Material Name: "))
        self.material_name_input = QLineEdit()
        self.material_name_input.setText("emptyMaterial")
        self.material_name_input.setPlaceholderText("Enter material name...")
        inputlayout.addWidget(self.material_name_input)

        layout.addLayout(inputlayout) 

        # Image generation checkboxes, per selecionar les textures que generara
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

        # Save Path, gui per colocar el output path
        save_path_layout = QHBoxLayout()
        save_path_layout.addWidget(QLabel(">>> Output Path: "))
        self.save_path_input = QLineEdit()
        self.save_path_input.setReadOnly(True)

        self.save_path_btn = QPushButton("")
        self.save_path_btn.setIcon(QIcon(fr"{os.environ['BASE_DIR']}/data/icons/folder.png")) # agafa el enviorn general, per aconseguir el base dir i posar el icon de folder
        self.save_path_btn.setIconSize(QSize(25, 25))

        self.save_path_btn.clicked.connect(self._select_save_path)
        save_path_layout.addWidget(self.save_path_input)
        save_path_layout.addWidget(self.save_path_btn)
        layout.addLayout(save_path_layout)

        # separator after title
        self._add_separator(layout)

        # positive prompt entry text, text de multilinea. 
        layout.addWidget(QLabel(">>> Positive Prompt"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter texture prompt...")
        layout.addWidget(self.prompt_input)    

        self._add_separator(layout)

        self.model_settings = self._create_settings_header(name="MODEL SETTINGS", layout=layout)
        
        # Implementation
        starttex = ">>> "
        
        hlay = QHBoxLayout()
        hlay.addWidget(QLabel(starttex + "Model Type"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["sdxl", "fast_sdxl"])
        self.model_combo.currentIndexChanged.connect(self._update_tips)
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

        # Advanced Settings gui button
        advanced_settings_btn = QToolButton()
        advanced_settings_btn.setText("Advanced Model Settings")

        advanced_settings_btn.clicked.connect(self.show_advanced_settings)
        self.model_settings.addWidget(advanced_settings_btn)

        self._add_separator(layout)

        self.fast_settings_vbox = self._create_settings_header(name="GENERATION SETTINGS", layout=layout)
        
        # els sliders per crear i posar la configuracio del model
        self.steps_slider, self.steps_entry = self.create_slider_entry("Steps", min_v=1, max_v=100, default_v=20, is_int=True, box=self.fast_settings_vbox)
        self.cfg_slider, self.cfg_entry = self.create_slider_entry("CFG Scale", min_v=1, max_v=160, default_v=70, divisor=10.0, box=self.fast_settings_vbox)
        self.rif_slider, self.rif_entry = self.create_slider_entry("Reference semblance Scale", min_v=1, max_v=30, default_v=10, divisor=10.0, box=self.fast_settings_vbox)
        self.noise_slider, self.noise_entry = self.create_slider_entry("Noise", min_v=0, max_v=100, default_v=0, divisor=100.0, box=self.fast_settings_vbox)

        # Separator before generated images
        self._add_separator(layout)

        # Material size, per selecionar la siize de les texutres, dropdown.
        layout.addWidget(QLabel(">>> Texture size: "))
        self.texture_combo = QComboBox()
        self.texture_combo.addItems(["1024", "2048", "4096"])
        layout.addWidget(self.texture_combo)

        self._add_separator(layout)

        self.progress = QProgressBar()
        self.progress.setFixedHeight(20)
        # progress bar per veure el progress a mesura que avança el codi
        self.progress.setStyleSheet(
        "QProgressBar { border-radius: 4px; text-align: center; } "
        "QProgressBar::chunk { background-color: #d67129; border-radius: 4px; }
        ")
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Spacer
        layout.addStretch()

        # Texturize button. Btn final de la gui per executar les textures.
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
        # Sync current main GUI settings to advanced panel
        dConf = self.extract_generation_settings()
        self.adv.sync_from_main_gui(dConf)

        self.adv.show()
        self.adv.raise_()
        self.adv.activateWindow()

    def update_progress(self, value):
        """Update the progress bar value. Funcio per augmentar la progress bar"""
        self.progress.setValue(value)
        QApplication.processEvents()

    def show_finish_message(self):
        """Show a success message box quan sacaba el missatge"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Automaytex | Done")
        msg.setText(f"Texture generation completed successfully! \n Files located at: {os.path.join(self.save_path_input.text(), 'textures')}")
        msg.setIcon(QMessageBox.Information)
        msg.exec()

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

    def _select_save_path(self):
        """Open dialog to select save path."""
        path = QFileDialog.getExistingDirectory(self, "Select Save Path")
        if path:
            self.save_path = path
            self.save_path_input.setText(path)

    def _add_reference_image(self):
        """Add reference image to list."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Reference Images",
            "",
            "Image Files (*.png *.exr *.jpg);;All Files (*)"
        ) # fitxer QFileDialog per accedir als fitxers i trobar les imatges per utilitzar de input en el codi. No activat.
        for file in files:
            if file not in self.reference_images:
                self.reference_images.append(file)
        self._update_ref_display()

    def _remove_reference_image(self):
        """Remove selected reference image from list. """
        if self.reference_images:
            self.reference_images.pop()
            self._update_ref_display()

    def _update_ref_display(self):
        """Update reference images display label."""
        if self.reference_images:
            display_text = "\n".join([os.path.basename(p) for p in self.reference_images])
            self.ref_list_label.setText(display_text)
        else:
            self.ref_list_label.setText("No images selected")

    def _add_lora_path(self):
        """Add LoRa model to list. Igual que les referencies imatges."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select LoRa Models",
            "",
            "LoRa Files (*.gguf *.safetensors);;All Files (*)"
        )
        for file in files:
            if file not in self.lora_paths:
                self.lora_paths.append(file)
        self._update_lora_display() # per upgrade la gui dels loras.

    def _remove_lora_path(self):
        """Remove selected LoRa model from list."""
        if self.lora_paths:
            self.lora_paths.pop()
            self._update_lora_display()

    def _update_lora_display(self):
        """Update LoRa display label."""
        if self.lora_paths:
            display_text = "\n".join([os.path.basename(p) for p in self.lora_paths])
            self.lora_list_label.setText(display_text)
        else:
            self.lora_list_label.setText("No LoRa selected")

    def _get_system_info(self):
        """Get current system information. Amb psutil aconsegueix el system info per ensenyar en la gui"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            ram_percent = ram.percent

            info = f"CPU: {cpu_percent}% | RAM: {ram_percent}% | Est. Gen Time: ~2-5 min"
            return info
        except Exception as e:
            return "System info unavailable"

    def _on_texturize(self):
        self.update_progress(0)
        settings = self.extract_generation_settings()

        self.texturize_signal.emit(settings)
        self.update_progress(100)
        # self.show_finish_message() # desactivat temporalment 

    def extract_generation_settings(self):
        """
        Funcio que extreu les dades de la gui per enviar al backend. Extract_generation_Settings. Crea una classe dConf del adv advanced settings.
        """
    
        dConf = self.adv.extract_maya_settings()

        user_prompt = self.prompt_input.toPlainText().strip() if hasattr(self, 'prompt_input') else ""
        general_prompt = dConf.positive_prompt.strip() if dConf.positive_prompt else ""

        # si es vol utilitzar general prompt
        if user_prompt and general_prompt:
            dConf.positive_prompt = f"{user_prompt}, {general_prompt}"
        elif user_prompt:
            dConf.positive_prompt = user_prompt
        # Else, it keeps the general_prompt as the positive prompt

        # Modifica la classe dConf per posar els valors de la gui, els settings del user
        dConf.texture_resolution = self.texture_combo.currentText()
        dConf.inference_steps = self.steps_slider.value()
        
        dConf.cfg_scale = self.cfg_slider.value() / 10.0
        dConf.noise = self.noise_slider.value() / 100.0
        dConf.base_model = self.model_combo.currentText().lower()
        
        dConf.system_prfered = self.system_combo.currentText()
        quant_text = self.quant_combo.currentText()
        dConf.quantization = quant_text if quant_text != "None" else None
        
        dConf.material_name = self.material_name_input.text()
        
        output_path_folder = self.save_path_input.text() if self.save_path_input.text() else dConf.output_path
        dConf.output_path = os.path.join(output_path_folder, dConf.material_name)

        print("Output path set to: ", dConf.output_path)

        dConf.generated_images = []
        
        if self.diffuse_check.isChecked(): dConf.generated_images.append("diffuse")
        if self.roughness_check.isChecked(): dConf.generated_images.append("roughness")
        if self.metalness_check.isChecked(): dConf.generated_images.append("metalness")
        if self.normal_check.isChecked(): dConf.generated_images.append("normal")
        if self.height_check.isChecked(): dConf.generated_images.append("height")

        dConf.seed = 123456789 # per colocar la seed de generacio
        dConf.pathsetter()
        return dConf # retorna dConf per la logica

    def _update_tips(self):
        # tip de generacio per el fast_sdxl que utilitzi 4 steps i CFG 2. 
        if self.model_combo.currentText() == "fast_sdxl":
            self.tips_label.setText("Tip: Recommended steps: 4 and CFG 2

