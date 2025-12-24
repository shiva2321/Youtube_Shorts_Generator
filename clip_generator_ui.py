import os
import sys
import threading
import queue
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Important Clips Generator")
        self.geometry("1000x780")

        self.proc = None
        self.log_q = queue.Queue()
        self.running = False

        # --- Core Vars
        self.video_path = tk.StringVar()
        self.transcript_path = tk.StringVar()
        self.num_clips = tk.IntVar(value=5)
        self.skip_intro = tk.DoubleVar(value=0.0)
        self.skip_outro = tk.DoubleVar(value=0.0)
        self.len_preset = tk.StringVar(value="60-90")
        self.language = tk.StringVar(value="auto")
        self.scene_thresh = tk.DoubleVar(value=0.35)
        self.silence_db = tk.DoubleVar(value=-35.0)
        self.min_silence = tk.DoubleVar(value=0.6)
        self.outdir = tk.StringVar(value=os.path.abspath("out_clips"))
        self.workdir = tk.StringVar(value=os.path.abspath("work"))

        # Performance
        self.jobs = tk.IntVar(value=max(1, min(4, (os.cpu_count() or 4))))
        self.encoder = tk.StringVar(value="auto")  # auto/cpu/nvenc
        self.hw_decode = tk.BooleanVar(value=False)
        self.quality = tk.StringVar(value="high")  # high/fast

        # Engagement / aspect
        self.engagement = tk.StringVar(value="balanced")  # balanced/action/dialogue
        self.aspect = tk.StringVar(value="source")        # source/16:9/9:16/1:1/4:5/21:9
        self.aspect_mode = tk.StringVar(value="smart")    # smart/fit/fill
        self.target_res = tk.StringVar(value="")          # e.g. 1080x1920

        # ASR
        self.asr_backend = tk.StringVar(value="auto")      # auto/whispercpp/faster-whisper
        self.asr_model = tk.StringVar(value="small")       # tiny/base/small/...
        self.asr_device = tk.StringVar(value="auto")       # auto/cpu/cuda
        self.asr_compute_type = tk.StringVar(value="auto") # auto/float16/int8/...
        self.asr_threads = tk.IntVar(value=6)

        # Overlays
        self.overlay = tk.StringVar(value="auto")
        self.overlay_style = tk.StringVar(value="bars")

        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # --------------------
        # Paths
        # --------------------
        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        r = 0
        ttk.Label(frm, text="Video:").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.video_path, width=80).grid(row=r, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=self._pick_video).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(frm, text="Transcript (optional SRT):").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.transcript_path, width=80).grid(row=r, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=self._pick_transcript).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(frm, text="Output dir:").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.outdir, width=80).grid(row=r, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=self._pick_outdir).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(frm, text="Work dir (audio/transcripts):").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.workdir, width=80).grid(row=r, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=self._pick_workdir).grid(row=r, column=2, sticky="e")

        frm.columnconfigure(1, weight=1)

        # --------------------
        # Clip & scoring options
        # --------------------
        opts = ttk.LabelFrame(self, text="Clip & Scoring Options")
        opts.pack(fill="x", **pad)

        r = 0
        ttk.Label(opts, text="# clips:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(opts, from_=1, to=50, textvariable=self.num_clips, width=7).grid(row=r, column=1, sticky="w", **pad)

        ttk.Label(opts, text="Len preset:").grid(row=r, column=2, sticky="w", **pad)
        ttk.Combobox(
            opts,
            textvariable=self.len_preset,
            values=["60-90", "60-120"],
            width=10,
            state="readonly"
        ).grid(row=r, column=3, sticky="w", **pad)

        ttk.Label(opts, text="Language:").grid(row=r, column=4, sticky="w", **pad)
        ttk.Combobox(
            opts,
            textvariable=self.language,
            values=["auto", "en", "ja"],
            width=8,
            state="readonly"
        ).grid(row=r, column=5, sticky="w", **pad)

        r += 1
        ttk.Label(opts, text="Skip intro (sec):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(opts, from_=0, to=99999, increment=10, textvariable=self.skip_intro, width=10).grid(row=r, column=1, sticky="w", **pad)

        ttk.Label(opts, text="Skip outro (sec):").grid(row=r, column=2, sticky="w", **pad)
        ttk.Spinbox(opts, from_=0, to=99999, increment=10, textvariable=self.skip_outro, width=10).grid(row=r, column=3, sticky="w", **pad)

        r += 1
        ttk.Label(opts, text="Scene thresh:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(opts, from_=0.05, to=0.95, increment=0.05, textvariable=self.scene_thresh, width=10).grid(
            row=r, column=1, sticky="w", **pad
        )

        ttk.Label(opts, text="Silence dB:").grid(row=r, column=2, sticky="w", **pad)
        ttk.Spinbox(opts, from_=-80, to=-5, increment=1, textvariable=self.silence_db, width=10).grid(
            row=r, column=3, sticky="w", **pad
        )

        ttk.Label(opts, text="Min silence (sec):").grid(row=r, column=4, sticky="w", **pad)
        ttk.Spinbox(opts, from_=0.1, to=5.0, increment=0.1, textvariable=self.min_silence, width=10).grid(
            row=r, column=5, sticky="w", **pad
        )

        r += 1
        ttk.Label(opts, text="Engagement mode:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(
            opts,
            textvariable=self.engagement,
            values=["balanced", "action", "dialogue"],
            width=12,
            state="readonly"
        ).grid(row=r, column=1, sticky="w", **pad)

        # --------------------
        # Aspect & quality
        # --------------------
        aspect_frame = ttk.LabelFrame(self, text="Aspect & Quality")
        aspect_frame.pack(fill="x", **pad)

        r = 0
        ttk.Label(aspect_frame, text="Aspect:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(
            aspect_frame,
            textvariable=self.aspect,
            values=["source", "16:9", "9:16", "1:1", "4:5", "21:9"],
            width=8,
            state="readonly"
        ).grid(row=r, column=1, sticky="w", **pad)

        ttk.Label(aspect_frame, text="Aspect mode:").grid(row=r, column=2, sticky="w", **pad)
        ttk.Combobox(
            aspect_frame,
            textvariable=self.aspect_mode,
            values=["smart", "fit", "fill"],
            width=8,
            state="readonly"
        ).grid(row=r, column=3, sticky="w", **pad)

        ttk.Label(aspect_frame, text="Target res (WxH):").grid(row=r, column=4, sticky="w", **pad)
        ttk.Entry(aspect_frame, textvariable=self.target_res, width=12).grid(row=r, column=5, sticky="w", **pad)

        r += 1
        ttk.Label(aspect_frame, text="Quality:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(
            aspect_frame,
            textvariable=self.quality,
            values=["high", "fast"],
            width=8,
            state="readonly"
        ).grid(row=r, column=1, sticky="w", **pad)

        ttk.Label(aspect_frame, text="Overlay:").grid(row=r, column=2, sticky="w", **pad)
        ttk.Combobox(
            aspect_frame,
            textvariable=self.overlay,
            values=["off", "auto", "on"],
            width=8,
            state="readonly"
        ).grid(row=r, column=3, sticky="w", **pad)

        ttk.Label(aspect_frame, text="Overlay style:").grid(row=r, column=4, sticky="w", **pad)
        ttk.Combobox(
            aspect_frame,
            textvariable=self.overlay_style,
            values=["bars", "shorts"],
            width=10,
            state="readonly"
        ).grid(row=r, column=5, sticky="w", **pad)

        # --------------------
        # ASR Settings
        # --------------------
        asr_frame = ttk.LabelFrame(self, text="Transcription (ASR)")
        asr_frame.pack(fill="x", **pad)

        r = 0
        ttk.Label(asr_frame, text="Backend:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(
            asr_frame,
            textvariable=self.asr_backend,
            values=["auto", "whispercpp", "faster-whisper"],
            width=14,
            state="readonly",
        ).grid(row=r, column=1, sticky="w", **pad)

        ttk.Label(asr_frame, text="Model:").grid(row=r, column=2, sticky="w", **pad)
        ttk.Entry(asr_frame, textvariable=self.asr_model, width=12).grid(row=r, column=3, sticky="w", **pad)

        r += 1
        ttk.Label(asr_frame, text="Device:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(
            asr_frame,
            textvariable=self.asr_device,
            values=["auto", "cpu", "cuda"],
            width=10,
            state="readonly",
        ).grid(row=r, column=1, sticky="w", **pad)

        ttk.Label(asr_frame, text="Compute type:").grid(row=r, column=2, sticky="w", **pad)
        ttk.Entry(asr_frame, textvariable=self.asr_compute_type, width=12).grid(row=r, column=3, sticky="w", **pad)

        ttk.Label(asr_frame, text="Threads (whisper.cpp):").grid(row=r, column=4, sticky="w", **pad)
        ttk.Spinbox(asr_frame, from_=1, to=32, textvariable=self.asr_threads, width=6).grid(row=r, column=5, sticky="w", **pad)

        # --------------------
        # Performance
        # --------------------
        perf = ttk.LabelFrame(self, text="Performance")
        perf.pack(fill="x", **pad)

        r = 0
        ttk.Label(perf, text="Parallel exports (jobs):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(perf, from_=1, to=16, textvariable=self.jobs, width=10).grid(row=r, column=1, sticky="w", **pad)

        ttk.Label(perf, text="Encoder:").grid(row=r, column=2, sticky="w", **pad)
        ttk.Combobox(
            perf,
            textvariable=self.encoder,
            values=["auto", "cpu", "nvenc"],
            width=10,
            state="readonly"
        ).grid(row=r, column=3, sticky="w", **pad)

        ttk.Checkbutton(perf, text="Try HW decode (CUDA)", variable=self.hw_decode).grid(row=r, column=4, sticky="w", **pad)

        # --------------------
        # Buttons + progress
        # --------------------
        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)

        self.run_btn = ttk.Button(btns, text="Run", command=self.on_run)
        self.run_btn.pack(side="left")

        self.stop_btn = ttk.Button(btns, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)

        self.prog = ttk.Progressbar(btns, mode="indeterminate")
        self.prog.pack(side="left", fill="x", expand=True, padx=10)

        # --------------------
        # Log console
        # --------------------
        logfrm = ttk.LabelFrame(self, text="Log")
        logfrm.pack(fill="both", expand=True, **pad)

        self.log = tk.Text(logfrm, height=20, wrap="word")
        self.log.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(logfrm, command=self.log.yview)
        scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scroll.set)

        note = ttk.Label(
            self,
            text="Tip: First run may take longer to transcribe; subsequent runs reuse cached WAV/SRT when possible.",
        )
        note.pack(anchor="w", padx=12, pady=2)

    # --- Pickers
    def _pick_video(self):
        p = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.webm *.avi"), ("All files", "*.*")],
        )
        if p:
            self.video_path.set(p)

    def _pick_transcript(self):
        p = filedialog.askopenfilename(
            title="Select Transcript (SRT)",
            filetypes=[("SRT files", "*.srt"), ("All files", "*.*")],
        )
        if p:
            self.transcript_path.set(p)

    def _pick_outdir(self):
        p = filedialog.askdirectory(title="Select Output Folder")
        if p:
            self.outdir.set(p)

    def _pick_workdir(self):
        p = filedialog.askdirectory(title="Select Work Folder")
        if p:
            self.workdir.set(p)

    # --- Logging
    def _append_log(self, s: str):
        self.log.insert("end", s)
        self.log.see("end")

    def _drain_log_queue(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    # --- Run/Stop
    def on_run(self):
        if self.running:
            return

        engine = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_important_clips.py")
        if not os.path.exists(engine):
            messagebox.showerror("Missing file", f"Could not find:\n{engine}\n\nPut this UI file next to auto_important_clips.py")
            return

        video = self.video_path.get().strip()
        if not video or not os.path.exists(video):
            messagebox.showerror("Invalid input", "Please select a valid video file.")
            return

        outdir = self.outdir.get().strip()
        workdir = self.workdir.get().strip()
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(workdir, exist_ok=True)

        # Basic and advanced arguments
        cmd = [sys.executable, os.path.abspath("auto_important_clips.py")]
        cmd += ["--video", video]
        cmd += ["--num-clips", str(int(self.num_clips.get()))]
        cmd += ["--skip-intro-sec", str(float(self.skip_intro.get()))]
        cmd += ["--skip-outro-sec", str(float(self.skip_outro.get()))]
        cmd += ["--len-preset", self.len_preset.get()]
        cmd += ["--outdir", outdir]
        cmd += ["--workdir", workdir]
        cmd += ["--language", self.language.get()]
        cmd += ["--scene-thresh", str(float(self.scene_thresh.get()))]
        cmd += ["--silence-db", str(float(self.silence_db.get()))]
        cmd += ["--min-silence-sec", str(float(self.min_silence.get()))]
        cmd += ["--jobs", str(int(self.jobs.get()))]
        cmd += ["--encoder", self.encoder.get()]
        cmd += ["--quality", self.quality.get()]
        cmd += ["--engagement", self.engagement.get()]

        # Aspect/quality
        cmd += ["--aspect", self.aspect.get(), "--aspect-mode", self.aspect_mode.get()]
        if self.target_res.get().strip():
            cmd += ["--target-res", self.target_res.get().strip()]
        cmd += ["--quality", self.quality.get()]

        # Overlay
        cmd += ["--overlay", self.overlay.get(), "--overlay-style", self.overlay_style.get()]

        # ASR settings
        cmd += ["--asr-backend", self.asr_backend.get()]
        cmd += ["--asr-model", self.asr_model.get()]
        cmd += ["--asr-device", self.asr_device.get()]
        cmd += ["--asr-compute-type", self.asr_compute_type.get()]
        cmd += ["--asr-threads", str(int(self.asr_threads.get()))]

        if bool(self.hw_decode.get()):
            cmd += ["--hw-decode"]

        tr = self.transcript_path.get().strip()
        if tr:
            if not os.path.exists(tr):
                messagebox.showerror("Invalid input", "Transcript path does not exist.")
                return
            cmd += ["--transcript", tr]

        self.log.delete("1.0", "end")
        self._append_log("Launching:\n" + " ".join(shlex_quote(a) for a in cmd) + "\n\n")

        self.running = True
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.prog.start(10)

        t = threading.Thread(target=self._worker_run, args=(cmd,), daemon=True)
        t.start()

    def _worker_run(self, cmd: List[str]):
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_q.put(line)
            rc = self.proc.wait()
            self.log_q.put(f"\nProcess exited with code: {rc}\n")
        except Exception as e:
            self.log_q.put(f"\nERROR: {e}\n")
        finally:
            self.proc = None
            self.running = False
            self.after(0, self._on_done_ui)

    def _on_done_ui(self):
        self.prog.stop()
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def on_stop(self):
        if not self.proc:
            return
        try:
            self.log_q.put("\nStopping...\n")
            self.proc.terminate()
        except Exception as e:
            self.log_q.put(f"\nFailed to stop: {e}\n")


def shlex_quote(s: str) -> str:
    if not s:
        return '""'
    if any(c.isspace() for c in s) or any(c in s for c in ['"', "'"]):
        return '"' + s.replace('"', '\\"') + '"'
    return s


if __name__ == "__main__":
    app = App()
    app.mainloop()
