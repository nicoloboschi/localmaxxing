"""Launch and manage an mlx_lm OpenAI-compatible server for one model.

The server (mlx_lm.server) supports continuous batching via --decode-concurrency
and --prompt-concurrency, so genuinely concurrent HTTP requests are batched on
the GPU rather than serialized. We set both to the max concurrency we test (8)
so concurrency speedups are observable.
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


class MLXServer:
    def __init__(self, repo: str, port: int = 8080, max_concurrency: int = 8,
                 log_path: str | None = None, backend: str = "lm",
                 chat_template_args: str | None = None):
        self.repo = repo
        self.port = port
        self.max_concurrency = max_concurrency
        self.log_path = log_path
        # JSON string passed to mlx_lm.server --chat-template-args, e.g.
        # '{"enable_thinking": false}' to force direct-answer mode (needed so
        # reasoning models put the answer in message.content, which graders read).
        self.chat_template_args = chat_template_args
        # backend: "lm" -> mlx_lm.server (continuous batching);
        #          "vlm" -> mlx_vlm.server (for multimodal archs like gemma-4;
        #                   no continuous batching, requests serialize).
        self.backend = backend
        self.proc: subprocess.Popen | None = None
        self._log_fh = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        if self.backend == "vlm":
            cmd = [
                sys.executable, "-m", "mlx_vlm.server",
                "--model", self.repo,
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--log-level", "WARNING",
            ]
        else:
            cmd = [
                sys.executable, "-m", "mlx_lm", "server",
                "--model", self.repo,
                "--port", str(self.port),
                "--decode-concurrency", str(self.max_concurrency),
                "--prompt-concurrency", str(self.max_concurrency),
                "--log-level", "WARNING",
            ]
            if self.chat_template_args:
                cmd += ["--chat-template-args", self.chat_template_args]
        self._log_fh = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            cmd, stdout=self._log_fh, stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout: float = 600.0) -> bool:
        """Poll until the model is loaded and serving, or timeout."""
        deadline = time.time() + timeout
        url = f"{self.base_url}/v1/models"
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False  # server died during load
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(2.0)
        return False

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
            self.proc = None
        if self._log_fh not in (None, subprocess.DEVNULL):
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def chat_completion(base_url: str, messages: list, max_tokens: int = 256,
                    temperature: float = 0.0, timeout: float = 300.0,
                    retries: int = 2, model: str | None = None) -> dict:
    """Blocking call to /v1/chat/completions. Returns parsed JSON response.

    Retries on transient connection errors (the server can briefly reset a
    connection under load); does NOT retry HTTP error responses.

    `model`: mlx_lm.server ignores this field, but mlx_vlm.server REQUIRES it
    and validates it against the loaded model -- so the runner passes the repo
    id for vlm-backed models.
    """
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if model is not None:
        body["model"] = model
    payload = json.dumps(body).encode()
    last_exc = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except (ConnectionResetError, ConnectionError, TimeoutError,
                urllib.error.URLError, OSError) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
    raise last_exc
