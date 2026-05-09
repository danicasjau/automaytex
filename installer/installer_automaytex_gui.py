"""
AutoMayTex Installer
Run: python automaytex_installer.py
Requires only stdlib (tkinter, subprocess, threading, winreg, shutil)
Place logo.png in the same folder as this script.
"""

import os
import sys
import glob
import shutil
import subprocess
import threading
import winreg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

DEFAULT_REPO = "https://github.com/danicasjau/automaytex.git"
MAYA_BASE    = r"C:\Program Files\Autodesk"

# ── Helpers ──────────────────────────────────────────────────────────────────

def find_maya_installs():
    hits = []
    if os.path.isdir(MAYA_BASE):
        for d in sorted(glob.glob(os.path.join(MAYA_BASE, "Maya*"))):
            mayapy = os.path.join(d, "bin", "mayapy.exe")
            if os.path.isfile(mayapy):
                hits.append((os.path.basename(d), d, mayapy))
    return hits

def get_python_version(mayapy_path):
    try:
        r = subprocess.run([mayapy_path, "--version"],
                           capture_output=True, text=True, timeout=10)
        ver = (r.stdout + r.stderr).strip().split()[-1]
        parts = ver.split(".")
        return ver, int(parts[0]), int(parts[1])
    except Exception:
        return "unknown", 0, 0

def check_cuda():
    if shutil.which("nvcc"):
        return True
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    return os.path.isdir(cuda_base) and bool(os.listdir(cuda_base))

def check_git():
    return shutil.which("git") is not None

def req_file_name(major, minor, cuda):
    suffix = "_cuda" if cuda else ""
    if   (major, minor) == (3, 10): return f"requirements_py310{suffix}.txt"
    elif (major, minor) == (3, 11): return f"requirements_py311{suffix}.txt"
    else:                            return f"requirements_py312{suffix}.txt"

def set_env_var(name, value):
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                            winreg.KEY_READ | winreg.KEY_WRITE) as k:
            try:
                existing, _ = winreg.QueryValueEx(k, name)
            except FileNotFoundError:
                existing = ""
            if value.lower() not in existing.lower():
                new_val = f"{existing};{value}".strip(";")
                winreg.SetValueEx(k, name, 0, winreg.REG_EXPAND_SZ, new_val)
    except Exception:
        pass

# ── App ───────────────────────────────────────────────────────────────────────

class InstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AutoMayTex Setup")
        self.geometry("520x480")
        self.resizable(False, False)

        # State
        self.maya_dir    = tk.StringVar()
        self.mayapy_path = tk.StringVar()
        self.install_dir = tk.StringVar(
            value=str(Path.home() / "Documents" / "maya" / "AutoMayTex"))
        self.repo_url    = tk.StringVar(value=DEFAULT_REPO)
        self.do_models   = tk.BooleanVar(value=True)
        self.cuda_found  = tk.BooleanVar(value=check_cuda())
        self._py_major   = 0
        self._py_minor   = 0
        self._maya_list  = []
        self._install_errors = []

        self.pages = []
        self.current = 0

        self._build_header()
        self._build_pages()
        self._build_footer()
        self._show(0)

    # ── Chrome ───────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, relief="flat", bd=0)
        hdr.pack(fill="x")

        # Centered logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
        self._logo_img = None
        if os.path.isfile(logo_path):
            try:
                self._logo_img = tk.PhotoImage(file=logo_path)
                w, h = self._logo_img.width(), self._logo_img.height()
                factor = max(1, max(w // 160, h // 80))
                if factor > 1:
                    self._logo_img = self._logo_img.subsample(factor, factor)
                tk.Label(hdr, image=self._logo_img).pack(pady=(16, 8))
            except Exception:
                tk.Label(hdr, text="AutoMayTex",
                         font=("Segoe UI", 13, "bold")).pack(pady=(16, 8))
        else:
            tk.Label(hdr, text="AutoMayTex",
                     font=("Segoe UI", 13, "bold")).pack(pady=(16, 8))

        ttk.Separator(hdr, orient="horizontal").pack(fill="x")

    def _build_footer(self):
        ttk.Separator(self, orient="horizontal").pack(side="bottom", fill="x")
        foot = tk.Frame(self)
        foot.pack(side="bottom", fill="x", padx=14, pady=8)

        self.btn_back = ttk.Button(foot, text="< Back", command=self._go_back)
        self.btn_back.pack(side="left")

        self.btn_next = ttk.Button(foot, text="Continue >", command=self._go_next)
        self.btn_next.pack(side="right")

        self.btn_cancel = ttk.Button(foot, text="Cancel", command=self._cancel)
        self.btn_cancel.pack(side="right", padx=6)

    def _build_pages(self):
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.pages = [
            self._make_page_welcome(),
            self._make_page_maya(),
            self._make_page_path(),
            self._make_page_repo(),
            self._make_page_models(),
            self._make_page_installing(),
            self._make_page_done(),
        ]

    def _show(self, n):
        self.current = n
        for i, p in enumerate(self.pages):
            if i == n:
                p.place(relx=0, rely=0, relwidth=1, relheight=1, in_=self.container)
                p.lift()
            else:
                p.place_forget()

        self.btn_back.config(state="normal" if 0 < n < 5 else "disabled")
        if n == 5:
            self.btn_next.config(state="disabled", text="Continue >")
            self.btn_cancel.config(state="disabled")
        elif n == 6:
            self.btn_next.config(state="normal", text="Finish")
            self.btn_cancel.config(state="disabled")
            self.btn_back.config(state="disabled")
        else:
            self.btn_next.config(state="normal", text="Continue >")
            self.btn_cancel.config(state="normal")

    def _go_next(self):
        n = self.current
        if n == 6:
            self.destroy()
            return
        if n == 1 and not self._validate_maya():
            return
        if n == 4:
            self._show(5)
            threading.Thread(target=self._run_install, daemon=True).start()
            return
        if n == 4:
            self._refresh_summary()
        self._show(n + 1)

    def _go_back(self):
        if self.current > 0:
            self._show(self.current - 1)

    def _cancel(self):
        if messagebox.askyesno("Cancel Setup", "Are you sure you want to cancel?"):
            self.destroy()

    # ── Page builders ─────────────────────────────────────────────────────────

    def _page_frame(self):
        return tk.Frame(self.container)

    def _section_title(self, parent, text):
        tk.Label(parent, text=text, font=("Segoe UI", 10, "bold"),
                 anchor="w").pack(fill="x", padx=22, pady=(18, 2))
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=22, pady=(0, 12))

    def _path_row(self, parent, label, var, browse_cmd):
        tk.Label(parent, text=label, anchor="w").pack(fill="x", padx=22, pady=(4, 1))
        row = tk.Frame(parent)
        row.pack(fill="x", padx=22, pady=(0, 6))
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=browse_cmd).pack(side="right", padx=(6, 0))

    # PAGE 0 — Welcome
    def _make_page_welcome(self):
        f = self._page_frame()
        self._section_title(f, "Welcome to AutoMayTex Setup")
        msg = (
            "This wizard will install the AutoMayTex plug-in into Autodesk Maya.\n\n"
            "Steps:\n"
            "  1.  Select your Maya installation\n"
            "  2.  Choose an installation folder\n"
            "  3.  Enter the repository URL\n"
            "  4.  Optionally install AI models\n"
            "  5.  Install dependencies and register the plug-in\n\n"
            "Click Continue to begin."
        )
        tk.Label(f, text=msg, justify="left", anchor="nw",
                 wraplength=460).pack(fill="x", padx=22)

        tk.Label(f, text="Pre-flight checks:", anchor="w",
                 font=("Segoe UI", 9, "bold")).pack(fill="x", padx=22, pady=(16, 2))
        for label, ok in [("Git for Windows", check_git()),
                           ("CUDA / NVIDIA GPU", check_cuda())]:
            sym  = "✔" if ok else "✘"
            note = "found" if ok else "not found"
            tk.Label(f, text=f"  {sym}  {label} — {note}",
                     anchor="w").pack(fill="x", padx=22, pady=1)
        return f

    # PAGE 1 — Maya
    def _make_page_maya(self):
        f = self._page_frame()
        self._section_title(f, "Select Maya Installation")

        self._maya_list = find_maya_installs()
        self._maya_radio_var = tk.IntVar(value=0)

        if self._maya_list:
            tk.Label(f, text="Detected installations:", anchor="w").pack(
                fill="x", padx=22, pady=(0, 4))
            for i, (name, path, _) in enumerate(self._maya_list):
                ttk.Radiobutton(f, text=f"{name}   ({path})",
                                variable=self._maya_radio_var, value=i,
                                command=lambda i=i: self._apply_maya(i)
                                ).pack(anchor="w", padx=32, pady=2)
            self._apply_maya(0)
            ttk.Separator(f, orient="horizontal").pack(fill="x", padx=22, pady=8)

        tk.Label(f, text="Or enter path manually:", anchor="w").pack(fill="x", padx=22)
        self._path_row(f, "Maya directory:", self.maya_dir, self._browse_maya)
        self._path_row(f, "mayapy.exe:", self.mayapy_path, self._browse_mayapy)

        row = tk.Frame(f)
        row.pack(fill="x", padx=22, pady=(4, 0))
        ttk.Button(row, text="Detect Python Version",
                   command=self._detect_python).pack(side="left")
        self._py_ver_lbl = tk.Label(row, text="", anchor="w")
        self._py_ver_lbl.pack(side="left", padx=10)
        return f

    def _apply_maya(self, idx):
        if idx < len(self._maya_list):
            _, path, mayapy = self._maya_list[idx]
            self.maya_dir.set(path)
            self.mayapy_path.set(mayapy)
            self._detect_python()

    def _browse_maya(self):
        d = filedialog.askdirectory(title="Select Maya folder")
        if d:
            self.maya_dir.set(d)
            mp = os.path.join(d, "bin", "mayapy.exe")
            self.mayapy_path.set(mp if os.path.isfile(mp) else "")
            self._detect_python()

    def _browse_mayapy(self):
        f = filedialog.askopenfilename(title="Select mayapy.exe",
                                       filetypes=[("Executable", "mayapy.exe")])
        if f:
            self.mayapy_path.set(f)
            self._detect_python()

    def _detect_python(self):
        mp = self.mayapy_path.get()
        if os.path.isfile(mp):
            ver, maj, mn = get_python_version(mp)
            self._py_major = maj
            self._py_minor = mn
            if hasattr(self, "_py_ver_lbl"):
                self._py_ver_lbl.config(text=f"Python {ver} detected")
        else:
            if hasattr(self, "_py_ver_lbl"):
                self._py_ver_lbl.config(text="mayapy.exe not found")

    def _validate_maya(self):
        if not os.path.isfile(self.mayapy_path.get()):
            messagebox.showerror("Error",
                "mayapy.exe not found.\nPlease select a valid Maya folder.")
            return False
        if self._py_major == 0:
            self._detect_python()
        return True

    # PAGE 2 — Install path
    def _make_page_path(self):
        f = self._page_frame()
        self._section_title(f, "Installation Folder")
        tk.Label(f, text="Choose where AutoMayTex will be installed.\n"
                          "The cloned repository and virtual environment will be placed here.",
                 justify="left", anchor="w", wraplength=460).pack(fill="x", padx=22, pady=(0, 10))
        self._path_row(f, "Installation path:", self.install_dir, self._browse_install)
        return f

    def _browse_install(self):
        d = filedialog.askdirectory(title="Choose installation folder")
        if d:
            self.install_dir.set(d)

    # PAGE 3 — Repo
    def _make_page_repo(self):
        f = self._page_frame()
        self._section_title(f, "Repository")
        tk.Label(f, text="The AutoMayTex source will be cloned from GitHub.\n"
                          "If the target folder already has a repository it will be updated (git pull).",
                 justify="left", anchor="w", wraplength=460).pack(fill="x", padx=22, pady=(0, 10))
        tk.Label(f, text="Repository URL:", anchor="w").pack(fill="x", padx=22, pady=(0, 2))
        ttk.Entry(f, textvariable=self.repo_url).pack(fill="x", padx=22)
        return f

    # PAGE 4 — Models + summary
    def _make_page_models(self):
        f = self._page_frame()
        self._section_title(f, "Optional: AI Models")
        tk.Label(f, text="model_installation.py can download pre-trained weights.\n"
                          "This may require several GB of space and internet bandwidth.",
                 justify="left", anchor="w", wraplength=460).pack(fill="x", padx=22, pady=(0, 8))
        ttk.Checkbutton(f, text="Download and install AI models",
                        variable=self.do_models).pack(anchor="w", padx=22, pady=4)

        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=22, pady=12)
        tk.Label(f, text="Installation summary:", anchor="w",
                 font=("Segoe UI", 9, "bold")).pack(fill="x", padx=22, pady=(0, 4))

        self._summary_frame = tk.Frame(f)
        self._summary_frame.pack(fill="x", padx=22)
        # Populate when shown
        f.bind("<Visibility>", lambda e: self._refresh_summary())
        return f

    def _refresh_summary(self):
        for w in self._summary_frame.winfo_children():
            w.destroy()
        rows = [
            ("Maya",       self.maya_dir.get()),
            ("Python",     f"{self._py_major}.{self._py_minor}"),
            ("CUDA",       "Yes" if self.cuda_found.get() else "No"),
            ("Install to", self.install_dir.get()),
            ("Repository", self.repo_url.get()),
        ]
        for k, v in rows:
            row = tk.Frame(self._summary_frame)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{k}:", width=12, anchor="w").pack(side="left")
            tk.Label(row, text=v, anchor="w",
                     font=("Courier New", 9)).pack(side="left")

    # PAGE 5 — Installing
    def _make_page_installing(self):
        f = self._page_frame()
        self._section_title(f, "Installing…")
        self._prog_lbl = tk.StringVar(value="Starting…")
        tk.Label(f, textvariable=self._prog_lbl,
                 anchor="w").pack(fill="x", padx=22, pady=(0, 4))
        self._prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(f, variable=self._prog_var,
                        maximum=100).pack(fill="x", padx=22, pady=(0, 8))
        self._log_box = tk.Text(f, height=12, font=("Courier New", 8),
                                state="disabled", wrap="word",
                                relief="sunken", bd=1)
        self._log_box.pack(fill="both", expand=True, padx=22, pady=(0, 4))
        sb = ttk.Scrollbar(f, command=self._log_box.yview)
        self._log_box.configure(yscrollcommand=sb.set)
        return f

    # PAGE 6 — Done (populated after install)
    def _make_page_done(self):
        f = self._page_frame()
        self._done_frame = f
        return f

    def _populate_done(self):
        for w in self._done_frame.winfo_children():
            w.destroy()
        errors = self._install_errors
        title = "Setup Complete" if not errors else "Setup Finished with Warnings"
        self._section_title(self._done_frame, title)

        if not errors:
            for item in ["Repository cloned",
                         "Python environment created",
                         "Dependencies installed",
                         "Maya plug-in registered"]:
                tk.Label(self._done_frame, text=f"  ✔  {item}",
                         anchor="w").pack(fill="x", padx=22, pady=2)
        else:
            for e in errors:
                tk.Label(self._done_frame, text=f"  ⚠  {e}",
                         anchor="w").pack(fill="x", padx=22, pady=2)

        ttk.Separator(self._done_frame, orient="horizontal").pack(
            fill="x", padx=22, pady=12)
        tk.Label(self._done_frame, text="Next steps:",
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(
            fill="x", padx=22, pady=(0, 4))
        for step in [
            "1.  Open Autodesk Maya",
            "2.  Windows → Settings/Preferences → Plug-in Manager",
            '3.  Find automayatex.py → enable "Loaded" and "Auto load"',
        ]:
            tk.Label(self._done_frame, text=step,
                     anchor="w").pack(fill="x", padx=22, pady=2)

    # ── Install thread ────────────────────────────────────────────────────────

    def _log(self, msg):
        def _do():
            self._log_box.config(state="normal")
            self._log_box.insert("end", msg + "\n")
            self._log_box.see("end")
            self._log_box.config(state="disabled")
        self.after(0, _do)

    def _set_prog(self, pct, label=""):
        def _do():
            self._prog_var.set(pct)
            if label:
                self._prog_lbl.set(label)
        self.after(0, _do)

    def _run_install(self):
        install_dir = self.install_dir.get()
        mayapy      = self.mayapy_path.get()
        repo_url    = self.repo_url.get()
        cuda        = self.cuda_found.get()
        do_models   = self.do_models.get()
        maj         = self._py_major
        mn          = self._py_minor
        errors      = []

        def run(cmd, cwd=None):
            self._log("$ " + " ".join(cmd))
            try:
                proc = subprocess.Popen(cmd, cwd=cwd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        text=True)
                for line in proc.stdout:
                    self._log(line.rstrip())
                proc.wait()
                return proc.returncode == 0
            except Exception as e:
                self._log(str(e))
                return False

        # 1. Directories
        self._set_prog(5, "Preparing installation folder…")
        os.makedirs(install_dir, exist_ok=True)
        self._log(f"Directory ready: {install_dir}")

        # 2. Git clone / pull
        self._set_prog(15, "Cloning repository…")
        if os.path.isdir(os.path.join(install_dir, ".git")):
            self._log("Existing repo found — pulling latest…")
            ok = run(["git", "pull"], cwd=install_dir)
        else:
            ok = run(["git", "clone", repo_url, install_dir])
        if not ok:
            errors.append("Git clone/pull failed")
        else:
            self._log("Repository ready.")

        # 3. Models (optional)
        if do_models:
            self._set_prog(30, "Installing AI models…")
            model_script = os.path.join(install_dir, "model_installation.py")
            if os.path.isfile(model_script):
                if not run([mayapy, model_script]):
                    self._log("WARNING: model_installation.py exited with errors.")
                else:
                    self._log("Models installed.")
            else:
                self._log("model_installation.py not found — skipping.")
        else:
            self._log("Model installation skipped.")

        # 4. Create venv
        self._set_prog(45, "Creating Python virtual environment…")
        venv_dir = os.path.join(install_dir, "venv")
        if not os.path.isdir(venv_dir):
            if not run([mayapy, "-m", "venv", venv_dir]):
                errors.append("Virtual environment creation failed")
            else:
                self._log(f"Venv created: {venv_dir}")
        else:
            self._log(f"Venv already exists: {venv_dir}")

        venv_pip    = os.path.join(venv_dir, "Scripts", "pip.exe")
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")

        # 5. Upgrade pip
        self._set_prog(52, "Upgrading pip…")
        run([venv_python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

        # 6. Requirements
        req_name = req_file_name(maj, mn, cuda)
        req_path = os.path.join(install_dir, req_name)
        if not os.path.isfile(req_path):
            req_path = os.path.join(install_dir, req_file_name(maj, mn, False))
        if not os.path.isfile(req_path):
            req_path = os.path.join(install_dir, "requirements.txt")

        self._set_prog(55, f"Installing dependencies ({os.path.basename(req_path)})…")
        if os.path.isfile(req_path):
            if not run([venv_pip, "install", "-r", req_path]):
                errors.append("pip install failed")
            else:
                self._log("Dependencies installed.")
        else:
            errors.append("Requirements file not found")
            self._log(f"ERROR: {req_path} not found.")

        # 7. Register plugin
        self._set_prog(88, "Registering Maya plug-in…")
        maya_ver_num = os.path.basename(self.maya_dir.get()).replace("Maya", "")
        plugins_dir  = os.path.join(os.path.expanduser("~"),
                                    "Documents", "maya", maya_ver_num, "plug-ins")
        os.makedirs(plugins_dir, exist_ok=True)
        plugin_src = os.path.join(install_dir, "automayatex.py")
        if os.path.isfile(plugin_src):
            shutil.copy2(plugin_src, os.path.join(plugins_dir, "automayatex.py"))
            self._log(f"Plugin copied to: {plugins_dir}")
            set_env_var("MAYA_PLUG_IN_PATH", plugins_dir)
            self._log("MAYA_PLUG_IN_PATH updated.")
        else:
            errors.append("automayatex.py not found in repo")
            self._log("WARNING: automayatex.py not found — skipping.")

        self._set_prog(100, "Done.")
        self._install_errors = errors
        self.after(400, self._finish)

    def _finish(self):
        self._populate_done()
        self._show(6)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform != "win32":
        print("AutoMayTex Installer is Windows-only.")
        sys.exit(1)
    app = InstallerApp()
    app.mainloop()