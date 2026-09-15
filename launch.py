"""一键启动器：起 Streamlit 服务，然后打开一个「应用窗口」。

双击 start.bat（或桌面快捷方式）就会跑这个文件：

* 服务已经在跑 → 直接开窗口，不重复启动；
* 没在跑 → 先起服务，等它真正就绪再开窗口，避免看到白屏；
* 关掉这个控制台窗口 = 停止服务。

它只负责「起服务 + 开窗口」，不碰任何检索逻辑，所以命令行和 REST API 的
用法完全不受影响。
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8501
URL = f"http://{HOST}:{PORT}"
READY_TIMEOUT = 90

EDGE_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def port_in_use() -> bool:
    """端口上有没有东西在监听 —— 用来判断服务是不是已经起来了。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((HOST, PORT)) == 0


def service_healthy(timeout: float = 1.5) -> bool:
    """问一下 Streamlit 自己的健康检查接口。"""
    try:
        with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def open_window() -> None:
    """优先用 Edge / Chrome 的「应用窗口」模式，观感更像个桌面程序。"""
    for exe in (shutil.which("msedge"), *EDGE_CANDIDATES, shutil.which("chrome")):
        if exe and Path(exe).exists():
            subprocess.Popen([exe, f"--app={URL}", "--new-window"])
            return
    webbrowser.open(URL)


def start_server() -> subprocess.Popen:
    command = [
        sys.executable, "-E", "-m", "streamlit", "run", str(BASE_DIR / "app.py"),
        "--server.address", HOST,
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    return subprocess.Popen(command, cwd=BASE_DIR)


def wait_until_ready(server: subprocess.Popen) -> bool:
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return False
        if service_healthy():
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    if port_in_use():
        if service_healthy():
            print(f"服务已经在 {URL} 跑着，直接开窗口。")
            open_window()
            return 0
        print(f"端口 {PORT} 被别的程序占用了，先把它关掉再试。")
        return 1

    print(f"正在启动 RAG 文档问答助手（{URL}）…")
    server = start_server()
    try:
        if wait_until_ready(server):
            print("已就绪，正在打开窗口。关掉这个窗口即停止服务。")
            open_window()
        elif server.poll() is not None:
            print("启动失败，请看上面的报错。")
            return server.returncode or 1
        else:
            print(f"等了 {READY_TIMEOUT} 秒还没就绪，可以手动打开 {URL}。")
        server.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务…")
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
