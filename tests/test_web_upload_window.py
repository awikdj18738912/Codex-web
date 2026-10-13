"""Browser behavior tests for the file upload window in the actual web page."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
CHROME = ROOT / ".runtime/playwright_browsers/chromium-1243/chrome-linux64/chrome"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


@pytest.fixture
def browser_page():
    handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(ROOT / "system/web"), **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(CHROME), headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        page.set_default_timeout(3000)
        page.route("https://unpkg.com/**", lambda route: route.fulfill(status=200, content_type="application/javascript", body="window.lucide={createIcons(){}};"))
        page.route("**/health", lambda route: route.fulfill(json={"entity_asr_context_enabled": False, "entity_audio_repair_enabled": False}))
        page.add_init_script("""
          window.__fakeSockets = [];
          window.WebSocket = class FakeWebSocket extends EventTarget {
            static CONNECTING=0; static OPEN=1; static CLOSING=2; static CLOSED=3;
            constructor(url) { super(); this.url=url; this.readyState=0; this.sent=[]; window.__fakeSockets.push(this); }
            open() { this.readyState=1; this.onopen?.({}); }
            send(value) { if (this.readyState !== 1) throw new Error('socket not open'); this.sent.push(value); }
            close() { this.readyState=3; this.dispatchEvent(new Event('close')); this.onclose?.({}); }
            ack(index) { this.onmessage?.({data: JSON.stringify({event:'chunk_ack', chunk_index:index})}); }
            ackRaw(value) { this.onmessage?.({data: JSON.stringify({event:'chunk_ack', chunk_index:value})}); }
          };
        """)
        page.goto(f"http://127.0.0.1:{server.server_port}/index.html")
        page.wait_for_function("window.__fakeSockets.length === 0")
        yield page
        browser.close()
    server.shutdown()
    server.server_close()


def start_upload(page, samples):
    page.evaluate("""samples => {
      decodeAudioFile = async () => new Float32Array(samples);
      startFile(new File(['x'], 'fixture.wav'));
    }""", samples)
    page.wait_for_function("window.__fakeSockets.length === 1")
    page.evaluate("window.__fakeSockets[0].open()")
    page.wait_for_function("window.__fakeSockets[0].sent.length > 0")


def binary_frames(page):
    return page.evaluate("window.__fakeSockets[0].sent.filter(x => x instanceof ArrayBuffer).map(x => Array.from(new Float32Array(x)))")


def test_window_orders_exact_bytes_and_waits_for_every_ack(browser_page):
    page = browser_page
    # Four complete 16k sample frames and one partial tail.
    samples = list(range(64003))
    start_upload(page, samples)
    assert page.evaluate("window.__fakeSockets[0].sent.filter(x => x instanceof ArrayBuffer).length") == 4

    # ACKs may arrive out of order. Repeating an ACK must not free another slot.
    page.evaluate("window.__fakeSockets[0].ack(2)")
    page.evaluate("window.__fakeSockets[0].ack(2)")
    page.wait_for_function("window.__fakeSockets[0].sent.filter(x => x instanceof ArrayBuffer).length === 5")
    assert page.evaluate("window.__fakeSockets[0].sent.at(-1) instanceof ArrayBuffer")
    page.evaluate("window.__fakeSockets[0].ack(5)")
    page.evaluate("window.__fakeSockets[0].ack(4)")
    page.evaluate("window.__fakeSockets[0].ack(3)")
    assert page.evaluate("window.__fakeSockets[0].sent.some(x => typeof x === 'string')") is False
    page.evaluate("window.__fakeSockets[0].ack(1)")
    page.wait_for_function("window.__fakeSockets[0].sent.some(x => typeof x === 'string')")

    frames = binary_frames(page)
    assert [len(frame) for frame in frames] == [16000, 16000, 16000, 16000, 3]
    assert [x for frame in frames for x in frame] == samples
    assert page.evaluate("JSON.parse(window.__fakeSockets[0].sent.find(x => typeof x === 'string')).event") == "finish"


@pytest.mark.parametrize("fault", ["close", "invalid_ack", "send_failure"])
def test_upload_transport_fault_reports_error_and_does_not_finish(browser_page, fault):
    page = browser_page
    if fault == "send_failure":
        page.evaluate("""() => { decodeAudioFile=async()=>new Float32Array(20000); startFile(new File(['x'],'fixture.wav')); }""")
        page.wait_for_function("window.__fakeSockets.length === 1")
        page.evaluate("window.__fakeSockets[0].send = () => { throw new Error('synthetic send failure'); }; window.__fakeSockets[0].open()")
    else:
        start_upload(page, list(range(20000)))
    if fault == "close":
        page.evaluate("window.__fakeSockets[0].close()")
    elif fault == "invalid_ack":
        page.evaluate("window.__fakeSockets[0].ackRaw(999)")
    page.wait_for_function("document.querySelector('#status').textContent.includes('失败') || document.querySelector('#status').textContent.includes('关闭')")
    status = page.locator("#status").inner_text()
    assert any(word in status for word in ("文件处理失败", "识别连接已关闭"))
    assert page.evaluate("window.__fakeSockets[0].sent.some(x => typeof x === 'string' && JSON.parse(x).event === 'finish')") is False


def test_new_file_invalidates_pending_old_decode(browser_page):
    page = browser_page
    page.evaluate("""() => {
      let releases=[];
      decodeAudioFile = () => new Promise(resolve => releases.push(() => resolve(new Float32Array(10))));
      startFile(new File(['a'], 'old.wav'));
      startFile(new File(['b'], 'new.wav'));
      window.__releaseDecodes = releases;
    }""")
    page.wait_for_function("window.__releaseDecodes && window.__releaseDecodes.length === 2")
    page.evaluate("window.__releaseDecodes[0]()")
    page.wait_for_timeout(20)
    assert page.evaluate("window.__fakeSockets.length") == 0
    page.evaluate("window.__releaseDecodes[1]()")
    page.wait_for_function("window.__fakeSockets.length === 1")
    page.evaluate("window.__fakeSockets[0].open()")
    page.wait_for_function("window.__fakeSockets[0].sent.some(x => x instanceof ArrayBuffer)")
    assert page.evaluate("window.__fakeSockets[0].sent.filter(x => x instanceof ArrayBuffer).length") == 1
