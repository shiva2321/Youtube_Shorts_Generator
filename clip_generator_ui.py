import os
import sys
import threading
import queue
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import hashlib
import re


class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.waittime = 500
        self.wraplength = 180
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def showtip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(
            self.tw,
            text=self.text,
            justify='left',
            background="#ffffe0",
            relief='solid',
            borderwidth=1,
            wraplength=self.wraplength
        )
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ClipGen Pro - GPU Whisper Edition")
        self.geometry("1000x850")

        style = ttk.Style()
        style.theme_use('clam')

        self.proc = None
        self.log_q = queue.Queue()
        self.running = False

        self.init_vars()
        self.build_ui()
        self.poll_log()

    def init_vars(self):
        self.video_path = tk.StringVar()
        self.transcript_path = tk.StringVar()
        self.outdir = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        self.workdir = tk.StringVar(value=os.path.join(os.getcwd(), "work"))

        self.num_clips = tk.IntVar(value=5)
        self.skip_intro = tk.DoubleVar(value=0.0)

        # Video length selection (in minutes)
        self.video_length = tk.DoubleVar(value=3.0)

        # Whisper Vars
        self.whisper_model = tk.StringVar(value="medium")
        self.whisper_lang = tk.StringVar(value="auto")

        # Overlay
        self.template_style = tk.StringVar(value="viral_shorts")
        self.channel_name = tk.StringVar(value="@MyChannel")
        # Use primary (hook vs punchline decision) for top / bottom by default.
        self.top_text = tk.StringVar(value="{primary}")
        # Back-end will convert this into "Subscribe {channel}" if applicable.
        self.bot_text = tk.StringVar(value="{primary}")
        self.font_path = tk.StringVar()

        # Export
        self.resolution = tk.StringVar(value="1080x1920")
        self.aspect_mode = tk.StringVar(value="fit")
        self.jobs = tk.IntVar(value=min(4, os.cpu_count() or 2))

        self.progress_var = tk.DoubleVar(value=0.0)

        # Auto-detect transcript option
        self.auto_detect_transcript = tk.BooleanVar(value=True)

    def build_ui(self):
        main = ttk.Frame(self, padding="10")
        main.pack(fill="both", expand=True)

        nb = ttk.Notebook(main)
        nb.pack(fill="both", expand=True)

        self.tab_input = ttk.Frame(nb)
        self.tab_overlay = ttk.Frame(nb)
        self.tab_export = ttk.Frame(nb)

        nb.add(self.tab_input, text="📁 Input & Transcription")
        nb.add(self.tab_overlay, text="🎨 Design")
        nb.add(self.tab_export, text="⚙️ Export")

        self._build_input_tab()
        self._build_overlay_tab()
        self._build_export_tab()
        self._build_footer(main)

    def _build_input_tab(self):
        # File Source
        f = ttk.LabelFrame(self.tab_input, text="Source Files", padding=10)
        f.pack(fill="x", padx=10, pady=10)

        grid_opts = {'sticky': 'ew', 'padx': 5, 'pady': 5}

        ttk.Label(f, text="Video File:").grid(row=0, column=0, **grid_opts)
        video_entry = ttk.Entry(f, textvariable=self.video_path)
        video_entry.grid(row=0, column=1, **grid_opts)
        video_entry.bind("<FocusOut>", self.check_for_existing_transcript)
        ttk.Button(
            f,
            text="Browse",
            command=lambda: self._browse_file(self.video_path, "Video")
        ).grid(row=0, column=2, **grid_opts)

        ttk.Label(f, text="SRT (Optional):").grid(row=1, column=0, **grid_opts)
        ttk.Entry(f, textvariable=self.transcript_path).grid(row=1, column=1, **grid_opts)
        ttk.Button(
            f,
            text="Browse",
            command=lambda: self._browse_file(self.transcript_path, "SRT")
        ).grid(row=1, column=2, **grid_opts)

        # Auto-detect transcript checkbox
        ttk.Checkbutton(
            f,
            text="Auto-detect existing transcript",
            variable=self.auto_detect_transcript
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        f.columnconfigure(1, weight=1)

        # Transcription Settings
        f_trans = ttk.LabelFrame(self.tab_input, text="Auto-Transcription (GPU)", padding=10)
        f_trans.pack(fill="x", padx=10, pady=5)

        ttk.Label(f_trans, text="Model Size:").grid(row=0, column=0, **grid_opts)
        ttk.Combobox(
            f_trans,
            textvariable=self.whisper_model,
            values=["small", "medium", "large"]
        ).grid(row=0, column=1, **grid_opts)

        ttk.Label(f_trans, text="Language:").grid(row=0, column=2, **grid_opts)
        ttk.Combobox(
            f_trans,
            textvariable=self.whisper_lang,
            values=["auto", "english", "japanese"]
        ).grid(row=0, column=3, **grid_opts)

        lbl_info = ttk.Label(
            f_trans,
            text="ℹ️ If SRT is not provided, this will run automatically. Uses GPU if available.",
            foreground="gray"
        )
        lbl_info.grid(row=1, column=0, columnspan=4, sticky="w", padx=5)

        # Clipping Logic
        f2 = ttk.LabelFrame(self.tab_input, text="Clipping Logic", padding=10)
        f2.pack(fill="x", padx=10, pady=10)

        ttk.Label(f2, text="Clip Count:").grid(row=0, column=0, **grid_opts)
        ttk.Spinbox(
            f2,
            from_=1,
            to=50,
            textvariable=self.num_clips,
            width=5
        ).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(f2, text="Skip Intro (s):").grid(row=0, column=2, **grid_opts)
        ttk.Entry(f2, textvariable=self.skip_intro, width=8).grid(
            row=0,
            column=3,
            sticky="w",
            padx=5,
            pady=5
        )

        # Video Length Selection
        ttk.Label(f2, text="Clip Length (each):").grid(row=1, column=0, **grid_opts)
        length_frame = ttk.Frame(f2)
        length_frame.grid(row=1, column=1, columnspan=3, sticky="ew", padx=5, pady=5)

        ttk.Scale(
            length_frame,
            variable=self.video_length,
            from_=1.0,
            to=5.0,
            orient="horizontal"
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))

        # Display current value
        length_label = ttk.Label(length_frame, width=6)
        length_label.pack(side="right")

        # Update label when slider changes
        def update_length_label(*args):
            length_label.config(text=f"{self.video_length.get():.1f} min")

        self.video_length.trace_add("write", update_length_label)
        update_length_label()  # Initialize label

        # Add tooltip with clarification
        ToolTip(length_frame, "Duration of EACH clip. Total = Clip Length × Number of Clips")

    def _build_overlay_tab(self):
        f = ttk.LabelFrame(self.tab_overlay, text="Style", padding=10)
        f.pack(fill="x", padx=10, pady=10)

        grid_opts = {'sticky': 'ew', 'padx': 5, 'pady': 5}

        ttk.Label(f, text="Template:").grid(row=0, column=0, **grid_opts)
        cb = ttk.Combobox(
            f,
            textvariable=self.template_style,
            values=["viral_shorts", "neon_vibes", "glass_modern", "cinematic", "simple"]
        )
        cb.grid(row=0, column=1, **grid_opts)

        ttk.Label(f, text="Channel Name:").grid(row=1, column=0, **grid_opts)
        ttk.Entry(f, textvariable=self.channel_name).grid(row=1, column=1, **grid_opts)

        ttk.Label(f, text="Custom Font:").grid(row=2, column=0, **grid_opts)
        ttk.Entry(f, textvariable=self.font_path).grid(row=2, column=1, **grid_opts)
        ttk.Button(
            f,
            text="Find...",
            command=lambda: self._browse_file(self.font_path, "Font")
        ).grid(row=2, column=2, **grid_opts)

        f.columnconfigure(1, weight=1)

    def _build_export_tab(self):
        f = ttk.LabelFrame(self.tab_export, text="Output Settings", padding=10)
        f.pack(fill="x", padx=10, pady=10)

        grid_opts = {'sticky': 'ew', 'padx': 5, 'pady': 5}

        ttk.Label(f, text="Output Dir:").grid(row=0, column=0, **grid_opts)
        ttk.Entry(f, textvariable=self.outdir).grid(row=0, column=1, **grid_opts)
        ttk.Button(
            f,
            text="Browse",
            command=lambda: self._browse_dir(self.outdir)
        ).grid(row=0, column=2, **grid_opts)

        ttk.Label(f, text="Resolution:").grid(row=1, column=0, **grid_opts)
        resolution_cb = ttk.Combobox(
            f,
            textvariable=self.resolution,
            values=["1080x1920", "1920x1080", "1080x1080", "720x1280", "1280x720", "540x960"]
        )
        resolution_cb.grid(row=1, column=1, **grid_opts)

        # Aspect Mode with improved description
        ttk.Label(f, text="Aspect Mode:").grid(row=2, column=0, **grid_opts)
        aspect_frame = ttk.Frame(f)
        aspect_frame.grid(row=2, column=1, **grid_opts)

        ttk.Radiobutton(
            aspect_frame,
            text="Fit",
            variable=self.aspect_mode,
            value="fit"
        ).pack(side="left", padx=(0, 10))

        ttk.Radiobutton(
            aspect_frame,
            text="Fill",
            variable=self.aspect_mode,
            value="fill"
        ).pack(side="left")

        # Add tooltips for aspect mode options
        fit_tip = ToolTip(aspect_frame, "Fit: Preserves original aspect ratio with black bars if needed (no cropping)")
        fill_tip = ToolTip(aspect_frame, "Fill: Fills the entire frame by zooming and cropping if necessary")

        ttk.Label(f, text="Workers:").grid(row=3, column=0, **grid_opts)
        ttk.Scale(
            f,
            variable=self.jobs,
            from_=1,
            to=16,
            orient="horizontal"
        ).grid(row=3, column=1, **grid_opts)

        # Add preview section with image showing aspect ratio effects
        preview_frame = ttk.LabelFrame(f, text="Aspect Ratio Preview", padding=10)
        preview_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=10)

        preview_text = tk.Text(
            preview_frame,
            height=4,
            bg="#f0f0f0",
            wrap="word"
        )
        preview_text.pack(fill="both")
        preview_text.insert("1.0",
                            "Preview of aspect ratio effects:\n\n"
                            "• Fit: Keeps the entire original scene visible with letterboxing/pillarboxing\n"
                            "• Fill: Zooms to fill the entire frame which may crop some edges of the scene\n\n"
                            "Choose 'Fit' if you want to ensure no part of the video is cropped.\n"
                            "Choose 'Fill' if you want to maximize screen usage without black bars."
                            )
        preview_text.config(state="disabled")

        f.columnconfigure(1, weight=1)

    def _build_footer(self, parent):
        f = ttk.Frame(parent, padding=10)
        f.pack(fill="x", side="bottom")

        self.pbar = ttk.Progressbar(f, variable=self.progress_var, maximum=100)
        self.pbar.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(f, text="START PROCESSING", command=self.run_process)
        self.btn_run.pack(side="right", padx=5)

        self.log_widget = tk.Text(
            parent,
            height=12,
            bg="#202020",
            fg="#eeeeee",
            font=("Consolas", 9)
        )
        self.log_widget.pack(fill="both", expand=True, padx=10)

    def _browse_file(self, var, type_name):
        types = [("All Files", "*.*")]
        if type_name == "Video":
            types = [("Video", "*.mp4 *.mkv *.mov")]
        elif type_name == "Font":
            types = [("Font", "*.ttf *.otf")]
        elif type_name == "SRT":
            types = [("Subtitle", "*.srt")]
        path = filedialog.askopenfilename(filetypes=types)
        if path:
            var.set(path)
            if type_name == "Video" and self.auto_detect_transcript.get():
                self.check_for_existing_transcript()

    def _browse_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def log(self, msg):
        self.log_widget.insert("end", str(msg) + "\n")
        self.log_widget.see("end")

    def poll_log(self):
        while not self.log_q.empty():
            msg = self.log_q.get_nowait()
            self.log(msg)
            if "Clip exported" in str(msg):
                self.progress_var.set(
                    self.progress_var.get() + (100 / self.num_clips.get())
                )
        self.after(100, self.poll_log)

    def get_video_filename(self):
        """Extract the filename without extension from the video path"""
        if not self.video_path.get():
            return None

        basename = os.path.basename(self.video_path.get())
        filename = os.path.splitext(basename)[0]
        # Clean the filename to make it suitable for folder names
        return re.sub(r'[^\w\-_\. ]', '_', filename)

    def get_video_hash(self):
        """Generate a unique hash for the video file to identify it"""
        if not self.video_path.get() or not os.path.exists(self.video_path.get()):
            return None

        # Use file size and name as a simple hash
        file_size = os.path.getsize(self.video_path.get())
        basename = os.path.basename(self.video_path.get())
        hash_input = f"{basename}_{file_size}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:10]

    def check_for_existing_transcript(self, event=None):
        """Check if a transcript already exists for this video"""
        if not self.auto_detect_transcript.get() or not self.video_path.get():
            return

        video_filename = self.get_video_filename()
        if not video_filename:
            return

        # Check in the output directory for a transcript
        video_output_dir = os.path.join(self.outdir.get(), video_filename)

        # Check for SRT files in the video's output directory
        if os.path.exists(video_output_dir):
            for file in os.listdir(video_output_dir):
                if file.endswith(".srt"):
                    srt_path = os.path.join(video_output_dir, file)
                    self.transcript_path.set(srt_path)
                    self.log(f"Found existing transcript: {srt_path}")
                    return

        # Also check in the workdir
        video_hash = self.get_video_hash()
        if video_hash:
            work_transcript = os.path.join(self.workdir.get(), f"{video_hash}.srt")
            if os.path.exists(work_transcript):
                self.transcript_path.set(work_transcript)
                self.log(f"Found existing transcript: {work_transcript}")
                return

        # Clear transcript path if none found
        self.transcript_path.set("")
        self.log("No existing transcript found. Will generate new one.")

    def run_process(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Select a video file first!")
            return

        self.running = True
        self.btn_run.config(state="disabled")
        self.progress_var.set(0)
        self.log("Starting Process...")

        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        video_filename = self.get_video_filename()
        video_hash = self.get_video_hash()

        # Create video-specific output directory
        video_output_dir = os.path.join(self.outdir.get(), video_filename)
        os.makedirs(video_output_dir, exist_ok=True)

        # Create work directory if it doesn't exist
        os.makedirs(self.workdir.get(), exist_ok=True)

        # Determine transcript path
        transcript_path = self.transcript_path.get()
        if not transcript_path:
            # If no transcript provided, set the path where it should be generated
            transcript_path = os.path.join(video_output_dir, f"{video_filename}_transcript.srt")

        # User sets the length of EACH clip, not total length
        # E.g., "3 minutes" means each clip is 3 minutes long
        clip_minutes = self.video_length.get()
        clip_seconds = int(clip_minutes * 60)
        total_seconds = clip_seconds * self.num_clips.get()
        total_minutes = total_seconds / 60

        self.log(f"Clip duration: {clip_minutes:.1f} minutes ({clip_seconds} seconds) per clip")
        self.log(f"Generating {self.num_clips.get()} clips × {clip_minutes:.1f} min = {total_minutes:.1f} minutes total")

        cmd = [
            sys.executable, "auto_important_clips.py",
            "--video", self.video_path.get(),
            "--outdir", video_output_dir,  # Use video-specific output directory
            "--workdir", self.workdir.get(),
            "--num-clips", str(self.num_clips.get()),
            "--template-style", self.template_style.get(),
            "--target-res", self.resolution.get(),
            "--aspect-mode", self.aspect_mode.get(),
            "--jobs", str(self.jobs.get()),
            "--overlay-top-text", self.top_text.get(),
            "--overlay-bottom-text", self.bot_text.get(),
            "--whisper-model", self.whisper_model.get(),
            "--language", self.whisper_lang.get(),
            "--output-prefix", video_filename,  # Use video filename as prefix for output files
            "--transcript-output", transcript_path,  # Specify where to save the transcript
            "--target-length", str(clip_seconds)  # Pass clip duration (each clip length)
        ]

        if self.transcript_path.get():
            cmd.extend(["--transcript", self.transcript_path.get()])

        if self.font_path.get():
            cmd.extend(["--overlay-font", self.font_path.get()])

        if self.channel_name.get():
            cmd.extend(["--channel-name", self.channel_name.get()])

        if self.skip_intro.get() > 0:
            cmd.extend(["--skip-intro", str(self.skip_intro.get())])


        try:
            self.log(f"Output directory: {video_output_dir}")
            self.log(f"Transcript will be saved to: {transcript_path}")
            self.log(f"Using aspect mode: {self.aspect_mode.get()}")
            self.log(f"Target resolution: {self.resolution.get()}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            for line in process.stdout:
                self.log_q.put(line.strip())

            process.wait()
            self.log_q.put("Process Completed.")

            # Update transcript path if it was generated
            if os.path.exists(transcript_path):
                self.transcript_path.set(transcript_path)

        except Exception as e:
            self.log_q.put(f"ERROR: {e}")
        finally:
            self.btn_run.config(state="normal")
            self.running = False


if __name__ == "__main__":
    app = App()
    app.mainloop()