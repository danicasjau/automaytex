

""" IMPORTING LIBRARIES """

import os

try:
    from PySide6.QtWidgets import ( # type: ignore
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QCheckBox,
        QFrame, QFileDialog, QTextEdit, QScrollArea,
        QProgressBar, QListWidget, QStackedWidget, QDialog, QDoubleSpinBox
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QSize # type: ignore
    from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QIcon # type: ignore

except ImportError:
    from PySide2.QtWidgets import (  # type: ignore
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QSlider, QCheckBox,
        QFrame, QFileDialog, QTextEdit,
        QProgressBar, QListWidget, QStackedWidget, QDialog
    )
    from PySide2.QtCore import Qt, QTimer, Signal  # type: ignore
    from PySide2.QtGui import QFont, QColor, QPainter, QPen, QPixmap, QIcon  # type: ignore

import psutil
import ctypes
import backServer as bk

import importlib
import importlib.util
import json

from config import configuration

class AdvancedSettings(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Settings Configuration")
        self.resize(700, 450)
        self.starttex = ">>> "
        self.lora_paths = []

        # Load model paths from models.json
        self._load_model_paths_from_json()

        # Main Layout
        self.main_vlayout = QVBoxLayout(self)

        # Top section: Sidebar on left, Content on right
        self.top_hlayout = QHBoxLayout()
        self.main_vlayout.addLayout(self.top_hlayout)

        # 1. Sidebar
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(120)
        self.sidebar.addItems(["Maya", "Models", "Server", "System", "Generation"])
        self.sidebar.currentRowChanged.connect(self.display_panel)

        # 2. Stacked Widget (The Panels)
        self.stack = QStackedWidget()
        
        self.panel_maya = self.create_maya_panel()
        self.panel_models = self.create_models_panel()
        self.panel_server = self.create_server_panel()
        self.panel_system = self.create_system_panel()
        self.panel_generation = self.create_generation_panel()

        self.stack.addWidget(self.panel_maya)
        self.stack.addWidget(self.panel_models)
        self.stack.addWidget(self.panel_server)
        self.stack.addWidget(self.panel_system)
        self.stack.addWidget(self.panel_generation)

        self.top_hlayout.addWidget(self.sidebar)
        self.top_hlayout.addWidget(self.stack)

        # 3. Bottom section: Save Button
        self.save_btn = QPushButton("Save and Close")
        self.save_btn.setFixedHeight(35)
        self.save_btn.setStyleSheet("font-weight: bold; background-color: #444; border-radius: 5px;")
        self.save_btn.clicked.connect(self.accept)
        self.main_vlayout.addWidget(self.save_btn)
        
        self.sidebar.setCurrentRow(0)  # Start on Maya

        # Apply defaults from config.py
        self._apply_config_defaults()

    def display_panel(self, index):
        self.stack.setCurrentIndex(index)

    # --- PANEL GENERATORS ---

    def create_maya_panel(self):
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setSpacing(10)
        
        # --- Title ---
        title_label = QLabel("MAYA CONFIGURATION")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        main_layout.addWidget(title_label)
        
        # --- Render Type Row ---
        render_row = QHBoxLayout()
        render_row.addWidget(QLabel("Render Type:"))
        self.render_combo = QComboBox()
        # Fixed spelling of tetrahedral
        self.render_combo.addItems(["thetredral"]) # , "cube", "maya-bake"
        self.render_combo.setCurrentText("thetredral")
        render_row.addWidget(self.render_combo)
        main_layout.addLayout(render_row)

        # --- Material Type Row ---
        material_row = QHBoxLayout()
        material_row.addWidget(QLabel("Maya Material Type:"))
        self.material_combo = QComboBox()
        self.material_combo.addItems(["mtlx", "standard surface", "arnold aiStandard", "lambert"])
        material_row.addWidget(self.material_combo)
        main_layout.addLayout(material_row)

        # Skydome render path
        skydome_path_lay = QHBoxLayout()
        skydome_path_lay.addWidget(QLabel(f"{self.starttex} Hdri path: "))
        self.skydome_render_path = QLineEdit("C:/models")
        skydome_path_lay.addWidget(self.skydome_render_path)
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(30)
        skydome_path_lay.addWidget(btn_browse)
        main_layout.addLayout(skydome_path_lay)

        # --- Bake Render Size Row ---
        render_size_row = QHBoxLayout()
        render_size_row.addWidget(QLabel("Bake Render Size:"))
        
        self.render_size_width_combo = QComboBox()
        self.render_size_width_combo.setEditable(True)
        self.render_size_width_combo.addItems(["512", "1024", "2048", "4096"]) 
        self.render_size_width_combo.setCurrentText("1024")
        render_size_row.addWidget(self.render_size_width_combo)

        render_size_row.addWidget(QLabel("x"))

        self.render_size_height_combo = QComboBox()
        self.render_size_height_combo.setEditable(True)
        self.render_size_height_combo.addItems(["512", "1024", "2048", "4096"]) 
        self.render_size_height_combo.setCurrentText("1024")
        render_size_row.addWidget(self.render_size_height_combo)
        
        main_layout.addLayout(render_size_row)

        # --- Height Scale Row ---
        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("Height Map Scale:"))
        self.height_scale = QDoubleSpinBox()
        self.height_scale.setRange(0.01, 10.0)
        self.height_scale.setDecimals(2)
        self.height_scale.setSingleStep(0.1)
        self.height_scale.setValue(1.0)
        height_row.addWidget(self.height_scale)
        main_layout.addLayout(height_row)

        # --- UV Chunk Size & Camera Scale Row ---
        uv_row = QHBoxLayout()
        uv_row.addWidget(QLabel("UV Chunk Size:"))
        self.uv_chunk_spin = QDoubleSpinBox()
        self.uv_chunk_spin.setRange(0.01, 8192.0)
        self.uv_chunk_spin.setDecimals(2)

        self.uv_chunk_spin.setValue(500.0)
        uv_row.addWidget(self.uv_chunk_spin)
        
        uv_row.addWidget(QLabel("Camera Scale:"))
        self.camera_scale_spin = QDoubleSpinBox()
        self.camera_scale_spin.setRange(0.01, 100.0)
        self.camera_scale_spin.setDecimals(2)
        self.camera_scale_spin.setSingleStep(0.01)
        self.camera_scale_spin.setValue(1.0)
        uv_row.addWidget(self.camera_scale_spin)
        main_layout.addLayout(uv_row)

        # --- Seam Fixer Settings Row ---
        seam_row = QHBoxLayout()
        seam_row.addWidget(QLabel("Seam Strength:"))
        self.seam_strength = QDoubleSpinBox()
        self.seam_strength.setRange(0.0, 1.0)
        self.seam_strength.setDecimals(2)
        self.seam_strength.setSingleStep(0.05)
        self.seam_strength.setValue(0.55)
        seam_row.addWidget(self.seam_strength)
        
        seam_row.addWidget(QLabel("Seam steps:"))
        self.seam_steps = QDoubleSpinBox()
        self.seam_steps.setRange(1.0, 1000.0)
        self.seam_steps.setDecimals(2)
        self.seam_steps.setValue(25.0)
        seam_row.addWidget(self.seam_steps)
        main_layout.addLayout(seam_row)

        # --- Checkboxes Row (Options) ---
        options_row = QHBoxLayout()
        
        self.enable_udims_check = QCheckBox("Enable UDIMs")
        self.enable_udims_check.setChecked(False)
        options_row.addWidget(self.enable_udims_check)

        self.retarget_uv_check = QCheckBox("Retarget UV")
        self.retarget_uv_check.setChecked(True)
        options_row.addWidget(self.retarget_uv_check)

        self.assign_maya_check = QCheckBox("Auto Assign Material")
        self.assign_maya_check.setChecked(True)
        options_row.addWidget(self.assign_maya_check)
        
        main_layout.addLayout(options_row)


        main_layout.addStretch()
        return widget

    def create_generation_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        title_label = QLabel("GENERATION CONFIGURATION")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        layout.addWidget(title_label)

        seed = self.create_slider_entry("Seed", -1, 2147483647, -1, 1.0, True)
        layout.addLayout(seed)

        # Prompt
        layout.addWidget(QLabel("General Positive Prompt"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter texture prompt...")
        layout.addWidget(self.prompt_input)
                # Prompt negative
        layout.addWidget(QLabel("Negative Prompt"))
        self.n_prompt_input = QTextEdit()
        self.n_prompt_input.setPlaceholderText("Enter negative texture prompt...")
        layout.addWidget(self.n_prompt_input)

        # Reference Images
        layout.addWidget(QLabel("LoRa Paths Add: "))
        ref_layout = QVBoxLayout()
        self.ref_list_label = QLabel("No LoRa models selected")
        self.ref_list_label.setWordWrap(True)
        self.ref_list_label.setStyleSheet("background-color: #222222; padding: 5px; border-radius: 3px;")
        ref_layout.addWidget(self.ref_list_label)
        
        ref_btn_layout = QHBoxLayout()
        ref_add_btn = QPushButton("+")
        ref_add_btn.setMaximumWidth(40)
        ref_add_btn.clicked.connect(self._add_lora_path)
        ref_remove_btn = QPushButton("X")
        ref_remove_btn.setMaximumWidth(40)
        ref_remove_btn.clicked.connect(self._remove_lora_path)
        ref_btn_layout.addWidget(ref_add_btn)
        ref_btn_layout.addWidget(ref_remove_btn)
        ref_btn_layout.addStretch()
        ref_layout.addLayout(ref_btn_layout)
        layout.addLayout(ref_layout)

        return widget


    def create_server_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        title_label = QLabel("SERVER CONFIGURATION")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        layout.addWidget(title_label)

        port = QHBoxLayout()
        port.addWidget(QLabel(f"Server Port:"))
        self.server_port_input = QLineEdit("8001")
        port.addWidget(self.server_port_input)
        layout.addLayout(port)

        ip = QHBoxLayout()
        ip.addWidget(QLabel(f"Server IP:"))
        self.server_ip_input = QLineEdit("127.0.0.1")
        ip.addWidget(self.server_ip_input)
        layout.addLayout(ip)
        
        label_baseurl = QHBoxLayout()
        label_baseurl.addWidget(QLabel(f"Server Base URL:"))
        self.server_baseurl_input = QLineEdit(f"http://{self.server_ip_input.text()}:{self.server_port_input.text()}")
        label_baseurl.addWidget(self.server_baseurl_input)
        layout.addLayout(label_baseurl)

        self.open_server_console = QCheckBox("Create New Console for Server")
        self.open_server_console.setChecked(True)
        # self.open_server_console.clicked.connect(self.open_server_console)
        layout.addWidget(self.open_server_console)

        layout.addStretch()

        # --------------------------------------
        # SERVER CONTROL BOX
        # --------------------------------------
        server_frame = QFrame()
        server_frame.setFrameShape(QFrame.StyledPanel)
        server_frame.setFrameShadow(QFrame.Raised)

        server_layout = QVBoxLayout(server_frame)
        server_layout.addWidget(QLabel("Server Control"))

        btn_start = QPushButton("Start Server")
        btn_start.clicked.connect(self._start_server)
        server_layout.addWidget(btn_start)

        btn_stop = QPushButton("Stop Server")
        btn_stop.clicked.connect(self._stop_server)
        server_layout.addWidget(btn_stop)

        server_layout.addStretch()

        layout.addWidget(server_frame)

        
        return widget

    def create_models_panel(self):
        widget = QWidget()
        outer_layout = QVBoxLayout(widget)

        title_label = QLabel("MODELS CONFIGURATION")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        outer_layout.addWidget(title_label)

        # General Models Path
        gen_path_lay = QHBoxLayout()
        gen_path_lay.addWidget(QLabel(f"{self.starttex} Default Install Path: "))
        self.gen_path_input = QLineEdit("C:/models")
        gen_path_lay.addWidget(self.gen_path_input)
        btn_browse_gen = QPushButton("...")
        btn_browse_gen.setFixedWidth(30)
        btn_browse_gen.clicked.connect(
            lambda: self.gen_path_input.setText(
                QFileDialog.getExistingDirectory(self, "Select Default Models Directory") or self.gen_path_input.text()
            )
        )
        gen_path_lay.addWidget(btn_browse_gen)
        outer_layout.addLayout(gen_path_lay)
        outer_layout.addSpacing(8)

        # Scrollable area for model rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self._models_scroll_layout = QVBoxLayout(scroll_widget)
        self._models_scroll_layout.setSpacing(12)
        scroll.setWidget(scroll_widget)
        outer_layout.addWidget(scroll, 1)

        # Store references for install-all and status refresh
        self._model_row_widgets = {}   # model_id -> dict of row widgets

        conf = configuration()
        try:
            with open(conf.models_json, "r") as f:
                self._models_catalogue = json.load(f)
        except Exception:
            self._models_catalogue = {"models": []}

        for entry in self._models_catalogue.get("models", []):
            self._add_model_row(entry)

        self._models_scroll_layout.addStretch()

        # Install All button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.install_all = QPushButton("Install All Missing")
        self.install_all.setFixedHeight(30)
        self.install_all.clicked.connect(self._install_all_models)
        btn_row.addWidget(self.install_all)
        outer_layout.addLayout(btn_row)

        return widget

    def _add_model_row(self, entry):
        """Build and add a single model row into the scroll layout."""
        model_id   = entry.get("name", "unknown")
        model_name = entry.get("model_name", model_id)
        install_path = entry.get("installation_path", "")
        install_name = entry.get("installation_name", "")
        full_path    = os.path.join(install_path, install_name)
        hf_url       = entry.get("hugging_face_url", "")

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        row_layout = QVBoxLayout(frame)
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.setSpacing(4)

        # Title + status
        installed = os.path.exists(full_path)
        status_color = "#4caf50" if installed else "#f44336"
        status_text  = "✔ Installed" if installed else "✘ Not Found"

        top_lay = QHBoxLayout()
        name_lbl = QLabel(f"<b>{model_name}</b>  <small>({model_id})</small>")
        status_lbl = QLabel(f"<b style='color:{status_color};'>{status_text}</b>")
        top_lay.addWidget(name_lbl)
        top_lay.addStretch()
        top_lay.addWidget(status_lbl)
        row_layout.addLayout(top_lay)

        # HF URL (read-only display)
        url_lbl = QLabel(f"<small><a href='{hf_url}'>{hf_url[:70]}{'...' if len(hf_url)>70 else ''}</a></small>")
        url_lbl.setOpenExternalLinks(True)
        row_layout.addWidget(url_lbl)

        # Install path editor + browse
        path_lay = QHBoxLayout()
        path_lbl = QLabel("Install path:")
        path_lbl.setFixedWidth(80)
        path_input = QLineEdit(install_path)
        path_btn = QPushButton("...")
        path_btn.setFixedWidth(28)
        path_btn.clicked.connect(
            lambda _=None, pi=path_input: pi.setText(
                QFileDialog.getExistingDirectory(self, "Select Installation Directory") or pi.text()
            )
        )
        path_lay.addWidget(path_lbl)
        path_lay.addWidget(path_input)
        path_lay.addWidget(path_btn)
        row_layout.addLayout(path_lay)

        # Progress bar (hidden until download starts)
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFixedHeight(14)
        progress.setVisible(False)
        row_layout.addWidget(progress)

        # Install button
        btn_lay = QHBoxLayout()
        inst_btn = QPushButton("Install / Download")
        inst_btn.setFixedHeight(26)
        inst_btn.clicked.connect(
            lambda _=None, e=entry, pi=path_input, pb=progress, sl=status_lbl:
                self._install_model(e, pi, pb, sl)
        )
        btn_lay.addStretch()
        btn_lay.addWidget(inst_btn)
        row_layout.addLayout(btn_lay)

        self._models_scroll_layout.addWidget(frame)
        self._model_row_widgets[model_id] = {
            "path_input": path_input,
            "progress":   progress,
            "status_lbl": status_lbl,
            "entry":      entry,
        }

    def _install_model(self, entry, path_input, progress_bar, status_lbl):
        """Download a single model file/folder into the user-specified path."""
        import concurrent.futures
        try:
            from PySide6.QtWidgets import QApplication as _QApp  # type: ignore
        except ImportError:
            from PySide2.QtWidgets import QApplication as _QApp  # type: ignore

        dest_dir  = path_input.text().strip() or entry.get("installation_path", "")
        dest_name = entry.get("download_name", entry.get("installation_name", "model"))
        hf_url    = entry.get("hugging_face_url", "")
        model_id  = entry.get("name", "")

        if not hf_url:
            status_lbl.setText("<b style='color:#f44336;'>✘ No URL configured</b>")
            return

        # Detect HuggingFace tree URL → use snapshot_download
        is_repo = "/tree/" in hf_url or hf_url.rstrip("/").endswith(dest_name) is False
        dest_full = os.path.join(dest_dir, dest_name)

        os.makedirs(dest_dir, exist_ok=True)

        if os.path.exists(dest_full):
            status_lbl.setText("<b style='color:#4caf50;'>✔ Installed</b>")
            return

        progress_bar.setVisible(True)
        progress_bar.setValue(0)
        status_lbl.setText("<b style='color:#e6b800;'>⬇ Downloading…</b>")
        _QApp.processEvents()

        error_holder = [None]

        def _do_download():
            try:
                import requests as _req
                if "/tree/" in hf_url:
                    # Use HuggingFace Hub snapshot_download for repo trees
                    try:
                        from huggingface_hub import snapshot_download  # type: ignore
                        repo_id = hf_url.split("huggingface.co/")[1].split("/tree/")[0]
                        snapshot_download(repo_id=repo_id, local_dir=dest_full, local_dir_use_symlinks=False)
                    except ImportError:
                        error_holder[0] = "huggingface_hub not installed. Run: pip install huggingface_hub"
                else:
                    # Direct file download with streaming + progress
                    resp = _req.get(hf_url, stream=True, timeout=120)
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    with open(dest_full, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=65536):
                            fh.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                pct = int(downloaded / total * 100)
                                progress_bar.setValue(pct)
                                _QApp.processEvents()
            except Exception as ex:
                error_holder[0] = str(ex)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_do_download)
            while not future.done():
                _QApp.processEvents()
                import time; time.sleep(0.05)

        if error_holder[0]:
            status_lbl.setText(f"<b style='color:#f44336;'>✘ Error: {error_holder[0][:60]}</b>")
            progress_bar.setVisible(False)
            return

        progress_bar.setValue(100)
        status_lbl.setText("<b style='color:#4caf50;'>✔ Installed</b>")

        # Persist the new path back into models.json
        self._update_model_path_in_json(entry.get("name", ""), dest_dir)

    def _install_all_models(self):
        """Install every model that is not yet present on disk."""
        for model_id, widgets in self._model_row_widgets.items():
            e  = widgets["entry"]
            pi = widgets["path_input"]
            pb = widgets["progress"]
            sl = widgets["status_lbl"]
            dest_dir  = pi.text().strip() or e.get("installation_path", "")
            dest_name = e.get("installation_name", "")
            if not os.path.exists(os.path.join(dest_dir, dest_name)):
                self._install_model(e, pi, pb, sl)

    def _update_model_path_in_json(self, model_name: str, new_path: str):
        """Persist the new installation_path for a model back to models.json."""
        conf = configuration()
        json_path = conf.models_json
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            for m in data.get("models", []):
                if m.get("name") == model_name:
                    m["installation_path"] = new_path
                    break
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
            # Refresh our in-memory catalogue
            self._load_model_paths_from_json()
            print(f"[AdvancedSettings] Updated installation_path for '{model_name}' → {new_path}")
        except Exception as e:
            print(f"[AdvancedSettings] Could not update models.json: {e}")

    def create_system_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("SYSTEM CONFIGURATION"))
        layout.addStretch()
        
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
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_usage)

        self.info_layout = QVBoxLayout()
        self.info_layout.setSpacing(6)
        self.refresh_system_data()
        layout.addLayout(self.info_layout)

        # --------------------------------------
        # Load button
        # --------------------------------------

        layout.addSpacing(10)

        self.load_button = QPushButton("Load Models")
        self.load_button.clicked.connect(self._load_models)
        layout.addWidget(self.load_button)

        self.unload_button = QPushButton("Unload Models")
        self.unload_button.clicked.connect(self._unload_models)
        layout.addWidget(self.unload_button)

        layout.addStretch()

        return panel

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, 'timer'):
            self.timer.start(100)

    def hideEvent(self, event):
        super().hideEvent(event)
        if hasattr(self, 'timer'):
            self.timer.stop()

    ###############################
    ## UTILS
    ###############################



    def _load_model_paths_from_json(self):
        """Load model installation paths from models.json on startup."""
        conf = configuration()
        json_path = conf.models_json
        self._model_paths = {}  # name -> full installation path
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            for m in data.get("models", []):
                name = m.get("name", "")
                install_path = m.get("installation_path", "")
                install_name = m.get("installation_name", "")
                self._model_paths[name] = os.path.join(install_path, install_name)
        except Exception as e:
            print(f"[AdvancedSettings] Could not load models.json: {e}")

    def _apply_config_defaults(self):
        """Pre-populate GUI widgets with defaults from config.configuration."""
        d = configuration()

        # Maya panel
        idx = self.render_combo.findText(d.renderMode)
        if idx >= 0: self.render_combo.setCurrentIndex(idx)

        idx = self.material_combo.findText(d.material_type)
        if idx >= 0: self.material_combo.setCurrentIndex(idx)

        self.camera_scale_spin.setValue(float(d.camera_scale))
        self.uv_chunk_spin.setValue(float(d.uv_chunk_size))
        self.seam_strength.setValue(float(d.seam_fixer_strength))
        self.seam_steps.setValue(float(d.seam_fixer_steps))
        self.retarget_uv_check.setChecked(d.retargetUV)
        self.assign_maya_check.setChecked(d.assign_maya_material)

        # Generation panel
        if hasattr(self, 'n_prompt_input'):
            self.n_prompt_input.setPlainText(d.negative_prompt.strip())

    def sync_from_main_gui(self, dConf):
        """Update widgets with values from the main GUI configuration."""
        # The main GUI prompt should not overwrite the general definer prompt.
        pass

    def extract_maya_settings(self):
        dConf = configuration()
        # --- Maya panel ---
        dConf.renderMode = self.render_combo.currentText()
        dConf.material_type = self.material_combo.currentText()
        dConf.camera_scale = self.camera_scale_spin.value()
        dConf.uv_chunk_size = self.uv_chunk_spin.value()
        dConf.seam_fixer_strength = self.seam_strength.value()
        dConf.seam_fixer_steps = int(self.seam_steps.value())
        dConf.retargetUV = self.retarget_uv_check.isChecked()
        dConf.assign_maya_material = self.assign_maya_check.isChecked()

        # Bake render size (two dropdowns)
        try:
            dConf.bake_render_width = int(self.render_size_width_combo.currentText())
            dConf.bake_render_height = int(self.render_size_height_combo.currentText())
        except ValueError:
            pass

        # --- Generation panel ---
        if hasattr(self, 'prompt_input'):
            dConf.positive_prompt = self.prompt_input.toPlainText()
        if hasattr(self, 'n_prompt_input'):
            dConf.negative_prompt = self.n_prompt_input.toPlainText()
        dConf.lora_paths = list(self.lora_paths)

        # --- Model paths (from JSON, overrideable via Models panel) ---
        if self._model_paths.get("sdxl"):
            dConf.diffusion_model = self._model_paths["sdxl"]
        if self._model_paths.get("controlnet"):
            dConf.controlnet_model = self._model_paths["controlnet"]
        if self._model_paths.get("depth"):
            dConf.depth_model = self._model_paths["depth"]

        return dConf

    # SYSTEM UPDATE
    def update_usage(self):
        # -------------------------------
        # RAM
        # -------------------------------
        v = psutil.virtual_memory()
        ram_used = v.used / (1024**3)
        ram_total = v.total / (1024**3)
        ram_percent = v.percent

        try:
            self.ram_square.set_value(ram_percent)
            self.ram_label.setText(f"RAM: {ram_used:.1f} / {ram_total:.1f} GB")
        except RuntimeError:
            if hasattr(self, 'timer'):
                self.timer.stop()
            return

        # -------------------------------
        # VRAM (GPU)
        # -------------------------------
        vram_used, vram_total = get_nvml_vram()
        vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0

        try:
            self.vram_square.set_value(vram_percent)
            self.vram_label.setText(f"VRAM: {vram_used / 1024:.1f} / {vram_total / 1024:.1f} GB")
        except RuntimeError:
            if hasattr(self, 'timer'):
                self.timer.stop()

    
    # EXTERNAL LIB CALL
    def _load_models(self):
        current_conf = self.extract_generation_settings()
        bk._load_all_models(current_conf)
        print("Models loaded")

    def _unload_models(self):
        bk._unload_all_models()
        print("Models unloaded")

    def _start_server(self):
        self.server_process = bk.start_server()
        print(self.server_process)

    def _stop_server(self):
        bk.stop_server(self.server_process)
        print("Server stopped")

    def get_lib_version(self, lib_name):
        """Checks if a library is installed and returns its version."""
        spec = importlib.util.find_spec(lib_name)
        if spec is None:
            return "Not Found"
        try:
            module = importlib.import_module(lib_name)
            return getattr(module, "__version__", "Installed (Unknown Version)")
        except Exception:
            return "Error Loading"

    def refresh_system_data(self):
        # 2. GPU & CUDA Info (via Torch if available)
        """
        torch_version = self.get_lib_version("torch")
        if torch_version != "Not Found":
            try:
                import torch
                cuda_avail = torch.cuda.is_available()
                self.add_info("CUDA Available", str(cuda_avail))
                if cuda_avail:
                    self.add_info("GPU Device", torch.cuda.get_device_name(0))
                    self.add_info("CUDA Version", torch.version.cuda)
                
                self.add_info("cuDNN Version", str(torch.backends.cudnn.version()))
            except Exception as e:
                self.add_info("CUDA Error", str(e))
        else:
            self.add_info("Torch", "Not installed (cannot check GPU detailed stats)")
        """

        # 3. AI Environment Libraries
        # self.add_info("PyTorch", torch_version)
        self.add_info("Diffusers", self.get_lib_version("diffusers"))
        self.add_info("Transformers", self.get_lib_version("transformers"))

    def add_section(self, title):
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: bold; color: #888; text-decoration: underline; margin-top: 10px;")
        self.info_layout.addWidget(lbl)

    def add_info(self, key, value):
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        key_lbl = QLabel(f"<b>{key}:</b>")
        key_lbl.setFixedWidth(150)
        val_lbl = QLabel(str(value))
        val_lbl.setWordWrap(True)
        lay.addWidget(key_lbl)
        lay.addWidget(val_lbl)
        lay.addStretch()
        self.info_layout.addLayout(lay)

        
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
        if hasattr(self, 'ref_list_label'):
            if not self.lora_paths:
                self.ref_list_label.setText("No LoRa models selected")
            else:
                names = [os.path.basename(p) for p in self.lora_paths]
                self.ref_list_label.setText("\n".join(names))

    def create_slider_entry(self, label_text, min_v, max_v, default_v, divisor=1.0, is_int=False):
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
        return row_layout
    


#######################
## SYSTEM UTILS
#######################

class SquareUsage(QWidget):
    def __init__(self, color):
        super().__init__()
        self.value = 0
        self.color = QColor(*color)
        self.setFixedSize(250, 40)

    def set_value(self, v):
        try:
            self.value = max(0, min(v, 100))
            self.update()
        except RuntimeError:
            pass

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
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)

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

        try:
            self.update()
        except RuntimeError:
            if hasattr(self, 'timer'):
                self.timer.stop()

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, 'timer'):
            self.timer.start(50)

    def hideEvent(self, event):
        super().hideEvent(event)
        if hasattr(self, 'timer'):
            self.timer.stop()

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
