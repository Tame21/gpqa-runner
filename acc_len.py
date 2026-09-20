#!/usr/bin/env python3
"""Report cumulative vLLM speculative decoding acceptance length using stdlib."""

import argparse
from datetime import datetime, timezone
from http.client import HTTPException
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DRAFTS = "vllm:spec_decode_num_drafts"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_per_pos"
# Quoted label values may contain spaces, commas, braces and escaped quotes.
SAMPLE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>(?:[^"{}]|"(?:\\.|[^"\\])*")*)\})?'
    r'\s+(?P<value>\S+)(?:\s+.*)?$'
)
LABEL = re.compile(r'(?:^|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"\s*(?=,|$)')


def analyse_metrics(metrics_text, num_speculative_tokens=None):
    """Read only counter samples, excluding Prometheus *_created timestamps."""
    if num_speculative_tokens is not None and (
        isinstance(num_speculative_tokens, bool)
        or not isinstance(num_speculative_tokens, int) or num_speculative_tokens < 1
    ):
        raise ValueError("num_speculative_tokens must be a positive integer")
    drafts = 0.0
    positions = {}
    found_drafts = False
    names = {DRAFTS, DRAFTS + "_total", ACCEPTED, ACCEPTED + "_total"}
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r'[\s{]', line, maxsplit=1)[0]
        if name not in names:
            continue
        sample = SAMPLE.fullmatch(line)
        if sample is None:
            raise ValueError(f"Malformed metric sample: {name}")
        value = float(sample["value"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid counter value for {name}")
        if name in (DRAFTS, DRAFTS + "_total"):
            found_drafts = True
            drafts += value
        else:
            labels = dict(LABEL.findall(sample["labels"] or ""))
            if "position" not in labels or not labels["position"].isdigit():
                raise ValueError("Accepted-token counter needs a nonnegative integer position label")
            position = int(labels["position"])
            positions[position] = positions.get(position, 0.0) + value
    if not found_drafts:
        raise ValueError("No speculative decoding draft metrics found")
    if drafts > 0 and not positions:
        raise ValueError("No per-position accepted-token metrics found; acc_len is unavailable")
    size = num_speculative_tokens or (max(positions, default=-1) + 1)
    if positions and max(positions) >= size:
        raise ValueError("Metric position exceeds --num-spec; use the server's configured speculative token count")
    if not math.isfinite(drafts) or any(not math.isfinite(value) for value in positions.values()):
        raise ValueError("Counter totals must be finite")
    return drafts, [positions.get(position, 0.0) for position in range(size)]


def calculate_acc_len(metrics_text, num_speculative_tokens=None):
    drafts, accepted = analyse_metrics(metrics_text, num_speculative_tokens)
    rates = [value / drafts for value in accepted] if drafts else None
    return {
        "status": "ok" if drafts else "no_draft",
        "num_drafts": drafts,
        "num_accepted_tokens_per_pos": accepted,
        "acceptance_per_pos": rates,
        "acc_len": 1 + sum(rates) if rates is not None else None,
    }


def metrics_url(model, override=""):
    if override:
        url = override
    elif model.get("url"):
        url = urljoin(model["url"].rstrip("/") + "/", "metrics")
    else:
        host = model.get("host_ip", "localhost")
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        protocol = "https" if model.get("enable_ssl", False) else "http"
        url = f"{protocol}://{host}:{model.get('host_port', 8080)}/metrics"
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.fragment:
        raise ValueError("Metrics URL must be an absolute HTTP(S) URL")
    return url


def collect_acc_len(url, num_speculative_tokens=None, timeout=10):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Metrics timeout must be a finite positive number")
    request = Request(url, headers={"Accept": "text/plain"})
    with urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    return calculate_acc_len(text, num_speculative_tokens)


def report_acc_len(model, options, output_dir, context=None):
    """Print and save results; metric failures do not erase a completed benchmark."""
    report = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "scope": "server_cumulative",
        "num_speculative_tokens": options.get("num_speculative_tokens"),
        "context": context or {},
    }
    try:
        report["metrics_url"] = metrics_url(model, options.get("metrics_url", ""))
        report.update(collect_acc_len(report["metrics_url"], report["num_speculative_tokens"],
                                      options.get("timeout", 10)))
    except (OSError, ValueError, HTTPException) as exc:
        report.update(status="unavailable", acc_len=None, error=str(exc))
    if report["status"] == "ok":
        print("acc_len statistics (server cumulative counters):", flush=True)
        for key in ("num_drafts", "num_accepted_tokens_per_pos", "acceptance_per_pos", "acc_len"):
            print(f"{key}={report[key]}", flush=True)
    elif report["status"] == "no_draft":
        print("acc_len: No draft (num_drafts=0); acceptance length is unavailable.", flush=True)
    else:
        print(f"acc_len unavailable: {report['error']}", file=sys.stderr, flush=True)
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = output_dir / f"acc_len_{stamp}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"acc_len report: {path}", flush=True)
    except OSError as exc:
        print(f"Could not save acc_len report: {exc}", file=sys.stderr, flush=True)
    return report


def main(argv=None):
    cli = argparse.ArgumentParser(description="Read cumulative vLLM acc_len from /metrics")
    cli.add_argument("port", type=int, nargs="?", default=8080)
    cli.add_argument("num_spec", type=int, nargs="?", help="Speculative token count; auto-detected when omitted")
    cli.add_argument("--host-ip", default="127.0.0.1")
    cli.add_argument("--metrics-url", default="")
    cli.add_argument("--timeout", type=float, default=10)
    args = cli.parse_args(argv)
    if not 1 <= args.port <= 65535 or (args.num_spec is not None and args.num_spec < 1):
        cli.error("port must be 1..65535 and num_spec must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        cli.error("timeout must be a finite positive number")
    report = report_acc_len(
        {"host_ip": args.host_ip, "host_port": args.port},
        {"metrics_url": args.metrics_url, "num_speculative_tokens": args.num_spec, "timeout": args.timeout},
        ROOT / "outputs" / "acc_len",
    )
    return 0 if report["status"] in ("ok", "no_draft") else 1


if __name__ == "__main__":
    sys.exit(main())
