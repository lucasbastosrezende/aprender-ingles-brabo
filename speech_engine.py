"""Local Piper worker: keep the model loaded between utterances."""
import atexit
import json
import queue
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path


class PiperEngine:
    def __init__(self, command, transform=None):
        self.command = command
        self.transform = transform
        self.process = None
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        atexit.register(self.close)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def _start(self):
        self.close()
        self.completed = queue.Queue()
        self.process = subprocess.Popen(
            self.command + ["--json-input", "--quiet"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        def read_completions(pipe, completed):
            try:
                for line in iter(pipe.readline, b""):
                    completed.put(line)
            finally:
                completed.put(None)
        threading.Thread(target=read_completions,
                         args=(self.process.stdout, self.completed), daemon=True).start()

    def synthesize(self, text):
        with self.lock:
            if text in self.cache:
                self.cache.move_to_end(text)
                return self.cache[text]
            if self.process is None or self.process.poll() is not None:
                self._start()
            with tempfile.TemporaryDirectory(prefix="ingles-voice-") as folder:
                output = Path(folder) / "speech.wav"
                try:
                    request = json.dumps({"text": text, "output_file": str(output)})
                    self.process.stdin.write((request + "\n").encode("utf-8"))
                    self.process.stdin.flush()
                    if self.completed.get(timeout=45) is None:
                        raise RuntimeError("O processo de voz encerrou inesperadamente.")
                    audio = output.read_bytes()
                    if not audio.startswith(b"RIFF") or len(audio) <= 44:
                        raise RuntimeError("A voz retornou um arquivo de áudio inválido.")
                    if self.transform:
                        audio = self.transform(audio, text)
                except Exception:
                    self.close()
                    raise
            self.cache[text] = audio
            while len(self.cache) > 16:
                self.cache.popitem(last=False)
            return audio
