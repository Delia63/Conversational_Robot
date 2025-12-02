from prometheus_client import Counter, Histogram, make_wsgi_app
from wsgiref.simple_server import make_server, WSGIServer
from socketserver import ThreadingMixIn
from contextlib import contextmanager
import threading, os, time, html

# ---- METRICS DEFINITIONS ----
asr_latency = Histogram("asr_latency_seconds", "ASR transcription latency (seconds)")
llm_latency = Histogram("llm_latency_seconds", "LLM request latency until completion (seconds)")
llm_first_token_latency = Histogram("llm_first_token_latency_seconds", "Latency from LLM request to first token (seconds)")
tts_latency = Histogram("tts_latency_seconds", "TTS blocking speak latency (seconds)")
round_trip = Histogram("round_trip_seconds", "Latency from end of user recording to issuing TTS (seconds)")

wake_triggers = Counter("wake_triggers_total", "Wake phrases successfully detected")
sessions_started = Counter("sessions_started_total", "Conversation sessions started")
sessions_ended = Counter("sessions_ended_total", "Conversation sessions ended")
interactions = Counter("interactions_total", "Turns inside active sessions")
unknown_answer = Counter("unknown_answer_total", "LLM replied unknown/uncertain")
errors_total = Counter("errors_total", "Unhandled errors")
tts_speak_calls = Counter("tts_speak_calls_total", "Number of TTS speak calls")

# ---- HELPERS ----
def _hist_sum_count(hist: Histogram):
    """Returnează (sum, count) pentru un histogram fără etichete."""
    s = c = 0.0
    for metric in hist.collect():
        for sample in metric.samples:
            # sample are câmpuri: name, labels, value, timestamp, exemplar
            if sample.labels:  # ignoră variantele etichetate
                continue
            if sample.name.endswith("_sum"):
                s = float(sample.value)
            elif sample.name.endswith("_count"):
                c = float(sample.value)
    return s, c

def _counter_val(cnt: Counter):
    val = 0.0
    for metric in cnt.collect():
        for sample in metric.samples:
            if sample.labels:
                continue
            if sample.name.endswith("_total"):
                val = float(sample.value)
    return val

def _fmt_ms(avg_s, count):
    if count <= 0:
        return "—"
    return f"{avg_s*1000:.0f} ms (n={int(count)})"


def gather_metrics_snapshot():
    histograms = [
        ("Round-trip", round_trip),
        ("ASR latency", asr_latency),
        ("LLM first token", llm_first_token_latency),
        ("LLM total", llm_latency),
        ("TTS latency", tts_latency),
    ]
    counters = [
        ("Wake triggers", wake_triggers),
        ("Sessions started", sessions_started),
        ("Sessions ended", sessions_ended),
        ("Turns", interactions),
        ("TTS speak calls", tts_speak_calls),
        ("\"Unknown\" replies", unknown_answer),
        ("Errors", errors_total),
    ]
    latencies = []
    for label, hist in histograms:
        s, c = _hist_sum_count(hist)
        avg = (s / c) if c else None
        latencies.append((label, avg, c))
    counter_vals = [(label, int(_counter_val(cnt))) for label, cnt in counters]
    return {"latencies": latencies, "counters": counter_vals}


def log_metrics_snapshot(logger):
    snap = gather_metrics_snapshot()
    logger.info("📊 Metrics snapshot (averages/counters):")
    for label, avg, count in snap["latencies"]:
        if avg is None or count == 0:
            logger.info(f"  • {label}: —")
        else:
            logger.info(f"  • {label}: {avg*1000:.0f} ms (n={int(count)})")
    for label, value in snap["counters"]:
        logger.info(f"  • {label}: {value}")

def _render_vitals_html():
    hs = [
        ("Round-trip", round_trip),
        ("ASR latency", asr_latency),
        ("LLM first token", llm_first_token_latency),
        ("LLM total", llm_latency),
        ("TTS latency", tts_latency),
    ]
    cs = [
        ("Wake triggers", wake_triggers),
        ("Sessions started", sessions_started),
        ("Sessions ended", sessions_ended),
        ("Turns (interactions)", interactions),
        ("TTS speak calls", tts_speak_calls),
        ("\"Unknown\" replies", unknown_answer),
        ("Errors", errors_total),
    ]

    rows_lat = []
    for label, h in hs:
        s, c = _hist_sum_count(h)
        avg = (s / c) if c else 0.0
        rows_lat.append((label, _fmt_ms(avg, c)))

    rows_cnt = [(label, f"{int(_counter_val(cn))}") for label, cn in cs]

    css = """
    <style>
      body { font: 14px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, system-ui, sans-serif; margin: 24px; }
      h1 { margin: 0 0 8px; font-size: 20px; }
      .small { color:#666; margin-bottom:16px }
      table { border-collapse: collapse; margin: 12px 0 24px; min-width: 420px; }
      th, td { padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }
      th { background: #fafafa; }
      .grid { display:flex; gap:32px; flex-wrap: wrap; }
      .card { padding:16px; border:1px solid #eee; border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
      a { color:#3366cc; text-decoration:none; }
      a:hover { text-decoration:underline; }
    </style>
    """
    lat_rows_html = "\n".join(f"<tr><td>{html.escape(k)}</td><td><b>{html.escape(v)}</b></td></tr>" for k,v in rows_lat)
    cnt_rows_html = "\n".join(f"<tr><td>{html.escape(k)}</td><td><b>{html.escape(v)}</b></td></tr>" for k,v in rows_cnt)

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Robot Vitals</title>{css}</head>
<body>
  <h1>Robot Vitals</h1>
  <div class="small">Only the important stuff. Full Prometheus at <a href="/metrics">/metrics</a>.</div>
  <div class="grid">
    <div class="card">
      <h3>Latency (avg)</h3>
      <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{lat_rows_html}</tbody></table>
      <div class="small">Tip: Round-trip = end of user speech → TTS start.</div>
    </div>
    <div class="card">
      <h3>Counters</h3>
      <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{cnt_rows_html}</tbody></table>
    </div>
  </div>
</body></html>"""
    return html_doc.encode("utf-8")

class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True

def _router_app(environ, start_response):
    path = environ.get("PATH_INFO") or "/"
    if path == "/metrics":
        return make_wsgi_app()(environ, start_response)
    if path == "/" or path == "/vitals":
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [_render_vitals_html()]
    start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
    return [b"Not Found"]

def boot_metrics():
    addr = os.getenv("METRICS_ADDR", "127.0.0.1")
    port = int(os.getenv("METRICS_PORT", "9108"))
    httpd = make_server(addr, port, _router_app, server_class=ThreadingWSGIServer)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()

    # Self-test opțional ca să nu vezi 0 la început
    if os.getenv("METRICS_SELFTEST", "0") == "1":
        wake_triggers.inc()
        sessions_started.inc()
        interactions.inc()
        tts_speak_calls.inc()
        with observe_hist(asr_latency): time.sleep(0.12)
        with observe_hist(round_trip): time.sleep(0.22)
        with observe_hist(tts_latency): time.sleep(0.05)
        sessions_ended.inc()
    return addr, port

@contextmanager
def observe_hist(hist: Histogram):
    start = time.perf_counter()
    try:
        yield
    finally:
        hist.observe(time.perf_counter() - start)

def wrap_stream_for_first_token(generator, hist: Histogram):
    first = {"done": False}
    start = time.perf_counter()
    def gen():
        for tok in generator:
            if not first["done"]:
                first["done"] = True
                hist.observe(time.perf_counter() - start)
            yield tok
    return gen()
