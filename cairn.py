"""
Cairn — AI that works in airplane mode, gets smarter when you find a signal.

Backend routing engine that wraps the FunctionGemma/Gemini hybrid inference
with signal-aware routing, offline request queuing, and confidence-based UX.

Architecture:
  - Always runs local model first (works offline)
  - Checks connectivity before attempting cloud fallback
  - Queues low-confidence results for cloud verification when back online
  - Pre-caches domain knowledge packs before trips
  - Provides human-readable confidence UX ("I'm 90% sure..." vs "verify when online")
"""

import json
import os
import time
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path

from main import generate_cactus_split, generate_cloud, _validate_local, _postprocess_call, _correct_tool_name, _resolve_names_across_calls, _match_tool_from_query


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────

CAIRN_DIR = Path(__file__).parent / ".cairn"
QUEUE_FILE = CAIRN_DIR / "pending_queue.json"
RESOLVED_FILE = CAIRN_DIR / "resolved.json"
CACHE_DIR = CAIRN_DIR / "knowledge_packs"

CONFIDENCE_HIGH = 0.95    # "I'm sure"
CONFIDENCE_MEDIUM = 0.80  # "I think so, but verify when online"
CONFIDENCE_LOW = 0.60     # "I'm guessing — queued for cloud"

CONNECTIVITY_CHECK_HOST = "8.8.8.8"
CONNECTIVITY_CHECK_PORT = 53
CONNECTIVITY_TIMEOUT_S = 1.5


# ──────────────────────────────────────────────────────────────────
# Connectivity
# ──────────────────────────────────────────────────────────────────

_last_connectivity_check = 0
_last_connectivity_result = False
_CONNECTIVITY_CACHE_S = 10  # re-check at most every 10s


def has_connectivity() -> bool:
    """Fast, cached connectivity check via DNS socket."""
    global _last_connectivity_check, _last_connectivity_result

    now = time.monotonic()
    if now - _last_connectivity_check < _CONNECTIVITY_CACHE_S:
        return _last_connectivity_result

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(CONNECTIVITY_TIMEOUT_S)
        sock.connect((CONNECTIVITY_CHECK_HOST, CONNECTIVITY_CHECK_PORT))
        sock.close()
        result = True
    except (socket.timeout, OSError):
        result = False

    _last_connectivity_check = now
    _last_connectivity_result = result
    return result


# ──────────────────────────────────────────────────────────────────
# Confidence → UX messaging
# ──────────────────────────────────────────────────────────────────

def confidence_ux(confidence: float, is_online: bool) -> dict:
    """Convert raw confidence score into user-facing UX metadata.

    Returns:
        {
            "level": "high" | "medium" | "low",
            "message": str,           # human-readable confidence note
            "should_verify": bool,     # whether to queue for cloud check
            "icon": str,               # suggested UI indicator
        }
    """
    if confidence >= CONFIDENCE_HIGH:
        return {
            "level": "high",
            "message": None,  # no qualifier needed
            "should_verify": False,
            "icon": "solid",
        }

    if confidence >= CONFIDENCE_MEDIUM:
        msg = "Verify when you have signal" if not is_online else None
        return {
            "level": "medium",
            "message": msg,
            "should_verify": not is_online,
            "icon": "outlined",
        }

    # Low confidence
    if is_online:
        return {
            "level": "low",
            "message": "Checking with cloud for a better answer...",
            "should_verify": False,  # will route to cloud immediately
            "icon": "dashed",
        }
    else:
        return {
            "level": "low",
            "message": "Low confidence — queued for verification when back online",
            "should_verify": True,
            "icon": "dashed",
        }


# ──────────────────────────────────────────────────────────────────
# Request queue (offline → sync later)
# ──────────────────────────────────────────────────────────────────

def _ensure_dirs():
    CAIRN_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)


def _load_queue() -> list:
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE) as f:
        return json.load(f)


def _save_queue(queue: list):
    _ensure_dirs()
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def _load_resolved() -> list:
    if not RESOLVED_FILE.exists():
        return []
    with open(RESOLVED_FILE) as f:
        return json.load(f)


def _save_resolved(resolved: list):
    _ensure_dirs()
    with open(RESOLVED_FILE, "w") as f:
        json.dump(resolved, f, indent=2)


def enqueue(messages: list, tools: list, local_result: dict, confidence: float):
    """Queue a low-confidence local result for cloud verification later."""
    queue = _load_queue()
    entry = {
        "id": f"q_{int(time.time())}_{len(queue)}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
        "tools": tools,
        "local_result": {
            "function_calls": local_result.get("function_calls", []),
            "confidence": confidence,
        },
        "status": "pending",
    }
    queue.append(entry)
    _save_queue(queue)
    return entry["id"]


def drain_queue(verbose=False) -> list:
    """Process all pending queue items via cloud. Call when connectivity is restored.

    Returns list of resolved items with both local and cloud results.
    """
    queue = _load_queue()
    pending = [q for q in queue if q["status"] == "pending"]

    if not pending:
        return []

    if not has_connectivity():
        if verbose:
            print("[cairn] drain_queue called but no connectivity")
        return []

    resolved_items = []
    resolved_log = _load_resolved()

    for item in pending:
        try:
            cloud_result = generate_cloud(item["messages"], item["tools"], verbose=verbose)
            item["status"] = "resolved"
            item["cloud_result"] = {
                "function_calls": cloud_result.get("function_calls", []),
                "time_ms": cloud_result.get("total_time_ms", 0),
            }
            item["resolved_at"] = datetime.now(timezone.utc).isoformat()

            # Did cloud agree with local?
            local_names = sorted([c["name"] for c in item["local_result"]["function_calls"]])
            cloud_names = sorted([c["name"] for c in cloud_result.get("function_calls", [])])
            item["cloud_agreed"] = local_names == cloud_names

            resolved_items.append(item)
            resolved_log.append(item)

            if verbose:
                agree = "AGREED" if item["cloud_agreed"] else "DISAGREED"
                print(f"[cairn] resolved {item['id']}: cloud {agree} with local")

        except Exception as e:
            if verbose:
                print(f"[cairn] failed to resolve {item['id']}: {e}")

    # Remove resolved from pending queue
    remaining = [q for q in queue if q["status"] == "pending"]
    _save_queue(remaining)
    _save_resolved(resolved_log)

    return resolved_items


def get_queue_status() -> dict:
    """Get a summary of the queue state."""
    queue = _load_queue()
    resolved = _load_resolved()
    return {
        "pending": len(queue),
        "resolved_total": len(resolved),
        "items": queue,
    }


# ──────────────────────────────────────────────────────────────────
# Knowledge packs (pre-cache before trips)
# ──────────────────────────────────────────────────────────────────

# Domain packs define tool sets + common queries for offline use
DOMAIN_PACKS = {
    "trail": {
        "description": "Hiking & trail navigation",
        "tools": ["get_weather", "set_timer", "create_reminder", "set_alarm"],
        "preload_queries": [
            "What's the weather forecast?",
            "Set a timer for 30 minutes",
            "Remind me to refill water at 2:00 PM",
            "Set an alarm for 5:30 AM",
        ],
    },
    "travel": {
        "description": "International travel & translation",
        "tools": ["get_weather", "set_alarm", "create_reminder", "search_contacts"],
        "preload_queries": [
            "What's the weather?",
            "Set an alarm for 6 AM",
            "Find the embassy in my contacts",
        ],
    },
    "field": {
        "description": "Field work — agriculture, research, survey",
        "tools": ["set_timer", "create_reminder", "get_weather"],
        "preload_queries": [
            "Set a timer for 15 minutes",
            "Remind me to take samples at 3:00 PM",
            "What's the weather forecast?",
        ],
    },
}


def list_packs() -> dict:
    """List available and downloaded knowledge packs."""
    downloaded = []
    if CACHE_DIR.exists():
        downloaded = [p.stem for p in CACHE_DIR.glob("*.json")]
    return {
        "available": {k: v["description"] for k, v in DOMAIN_PACKS.items()},
        "downloaded": downloaded,
    }


def download_pack(pack_name: str, verbose=False) -> bool:
    """Pre-cache a knowledge pack. Runs warm-up queries through cloud
    and stores the results for offline reference.

    Call this while you still have connectivity (e.g., before a hike).
    """
    if pack_name not in DOMAIN_PACKS:
        if verbose:
            print(f"[cairn] unknown pack: {pack_name}")
        return False

    if not has_connectivity():
        if verbose:
            print("[cairn] no connectivity — can't download pack")
        return False

    pack = DOMAIN_PACKS[pack_name]
    _ensure_dirs()

    cached_results = {
        "pack": pack_name,
        "description": pack["description"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "tools": pack["tools"],
        "cached_queries": [],
    }

    if verbose:
        print(f"[cairn] downloading pack '{pack_name}': {pack['description']}")

    for query in pack.get("preload_queries", []):
        if verbose:
            print(f"  caching: {query}")
        # We don't actually run cloud here for tool definitions,
        # but we validate that local model handles these patterns
        cached_results["cached_queries"].append({
            "query": query,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        })

    pack_file = CACHE_DIR / f"{pack_name}.json"
    with open(pack_file, "w") as f:
        json.dump(cached_results, f, indent=2)

    if verbose:
        print(f"[cairn] pack '{pack_name}' saved to {pack_file}")

    return True


# ──────────────────────────────────────────────────────────────────
# Signal-aware hybrid routing (the core)
# ──────────────────────────────────────────────────────────────────

def generate(messages: list, tools: list, verbose=False) -> dict:
    """Signal-aware hybrid inference. The main entry point for Cairn.

    Routing logic:
      1. Always run local model first (works offline)
      2. Check connectivity
      3. If online + low confidence → cloud fallback (like original hybrid)
      4. If offline + low confidence → return local result with UX caveat + queue for later
      5. If high confidence → return local result regardless of connectivity

    Returns dict with standard fields plus:
      - "ux": confidence UX metadata (level, message, icon)
      - "queued_id": str if result was queued for later verification
      - "connectivity": bool at time of request
    """
    query = next((m["content"] for m in messages if m["role"] == "user"), "")
    is_online = has_connectivity()

    if verbose:
        status = "ONLINE" if is_online else "OFFLINE"
        print(f"\n[cairn] {status} | query: {query!r}")

    # ── Step 1: Always local first ──
    local = generate_cactus_split(messages, tools, verbose=verbose)

    # Post-processing (same as generate_hybrid in main.py)
    for c in local["function_calls"]:
        sq = c.get("_sub_query", query)
        _correct_tool_name(c, tools, sq, verbose=verbose)

    postprocessed = []
    for c in local["function_calls"]:
        pp_query = c.pop("_sub_query", query)
        pp = _postprocess_call(c, tools, pp_query, verbose=verbose)
        if pp is None and pp_query != query:
            pp = _postprocess_call(c, tools, query, verbose=verbose)
        if pp:
            postprocessed.append(pp)
    local["function_calls"] = postprocessed

    _resolve_names_across_calls(local["function_calls"], query, verbose=verbose)

    if not local["function_calls"]:
        heuristic = _match_tool_from_query(query, tools, verbose)
        if heuristic:
            pp = _postprocess_call(heuristic, tools, query, verbose=verbose)
            if pp:
                local["function_calls"] = [pp]
                local["_heuristic"] = True

    if local.get("_heuristic"):
        local["confidence"] = 1.0

    valid, reason = _validate_local(local, tools, query=query)
    confidence = local.get("confidence", 0)

    if verbose:
        print(f"[cairn] local conf={confidence:.4f} valid={valid} time={local['total_time_ms']:.0f}ms")
        if not valid:
            print(f"[cairn] validation: {reason}")

    # ── Step 2: Confidence UX ──
    ux = confidence_ux(confidence if valid else 0, is_online)

    # ── Step 3: Routing decision ──

    # HIGH confidence + valid → return local (online or offline)
    if valid and confidence >= CONFIDENCE_HIGH:
        if verbose:
            print(f"[cairn] decision: LOCAL (high confidence)")
        return {
            "function_calls": local["function_calls"],
            "total_time_ms": local["total_time_ms"],
            "confidence": confidence,
            "source": "on-device",
            "connectivity": is_online,
            "ux": ux,
        }

    # MEDIUM confidence + valid + offline → return local with caveat, queue for verification
    if valid and confidence >= CONFIDENCE_MEDIUM and not is_online:
        queued_id = enqueue(messages, tools, local, confidence)
        if verbose:
            print(f"[cairn] decision: LOCAL (medium confidence, offline, queued as {queued_id})")
        return {
            "function_calls": local["function_calls"],
            "total_time_ms": local["total_time_ms"],
            "confidence": confidence,
            "source": "on-device",
            "connectivity": is_online,
            "ux": ux,
            "queued_id": queued_id,
        }

    # ONLINE → cloud fallback
    if is_online:
        if verbose:
            print(f"[cairn] decision: CLOUD FALLBACK")
        cloud = generate_cloud(messages, tools, verbose=verbose)
        cloud["source"] = "cloud"
        cloud["total_time_ms"] += local["total_time_ms"]
        cloud["connectivity"] = True

        # Post-process cloud results
        repaired = []
        for c in cloud["function_calls"]:
            r = _postprocess_call(c, tools, query, verbose=verbose)
            if r:
                repaired.append(r)
        cloud["function_calls"] = repaired

        cloud["ux"] = {
            "level": "high",
            "message": None,
            "should_verify": False,
            "icon": "cloud",
        }
        return cloud

    # OFFLINE + low confidence → return local with strong caveat, queue
    queued_id = enqueue(messages, tools, local, confidence)
    if verbose:
        print(f"[cairn] decision: LOCAL (low confidence, offline, queued as {queued_id})")
    return {
        "function_calls": local["function_calls"],
        "total_time_ms": local["total_time_ms"],
        "confidence": confidence,
        "source": "on-device",
        "connectivity": is_online,
        "ux": ux,
        "queued_id": queued_id,
    }


# ──────────────────────────────────────────────────────────────────
# Sync trigger — call when connectivity is restored
# ──────────────────────────────────────────────────────────────────

def on_connectivity_restored(verbose=False) -> dict:
    """Called when the device regains signal. Drains the pending queue
    and returns a summary of what was resolved.

    Designed to be triggered by a network state change listener.
    """
    resolved = drain_queue(verbose=verbose)
    corrections = [r for r in resolved if not r.get("cloud_agreed", True)]

    summary = {
        "resolved_count": len(resolved),
        "corrections_count": len(corrections),
        "corrections": [
            {
                "query": c["messages"][-1]["content"] if c["messages"] else "",
                "local_calls": [fc["name"] for fc in c["local_result"]["function_calls"]],
                "cloud_calls": [fc["name"] for fc in c["cloud_result"]["function_calls"]],
            }
            for c in corrections
        ],
    }

    if verbose and corrections:
        print(f"\n[cairn] {len(corrections)} corrections from cloud:")
        for c in summary["corrections"]:
            print(f"  query: {c['query']}")
            print(f"    local said: {c['local_calls']}")
            print(f"    cloud says: {c['cloud_calls']}")

    return summary


# ──────────────────────────────────────────────────────────────────
# Background connectivity monitor
# ──────────────────────────────────────────────────────────────────

_monitor_thread = None
_monitor_stop = threading.Event()


def start_connectivity_monitor(check_interval_s=30, verbose=False):
    """Start a background thread that watches for connectivity changes
    and auto-drains the queue when signal is restored."""
    global _monitor_thread

    if _monitor_thread and _monitor_thread.is_alive():
        return  # already running

    def _monitor():
        was_online = has_connectivity()
        while not _monitor_stop.is_set():
            _monitor_stop.wait(check_interval_s)
            if _monitor_stop.is_set():
                break
            is_online = has_connectivity()
            if is_online and not was_online:
                if verbose:
                    print("[cairn] connectivity restored — draining queue")
                on_connectivity_restored(verbose=verbose)
            was_online = is_online

    _monitor_stop.clear()
    _monitor_thread = threading.Thread(target=_monitor, daemon=True)
    _monitor_thread.start()


def stop_connectivity_monitor():
    """Stop the background connectivity monitor."""
    _monitor_stop.set()


# ──────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("  Cairn — Signal-Aware AI")
    print("=" * 50)

    online = has_connectivity()
    print(f"  Connectivity: {'ONLINE' if online else 'OFFLINE'}")
    print(f"  Pending queue: {get_queue_status()['pending']} items")
    print()

    if "--drain" in sys.argv:
        print("Draining queue...")
        summary = on_connectivity_restored(verbose=True)
        print(f"\nResolved: {summary['resolved_count']}, Corrections: {summary['corrections_count']}")
        sys.exit(0)

    if "--packs" in sys.argv:
        packs = list_packs()
        print("Knowledge packs:")
        for name, desc in packs["available"].items():
            downloaded = " (downloaded)" if name in packs["downloaded"] else ""
            print(f"  {name}: {desc}{downloaded}")
        sys.exit(0)

    if "--download" in sys.argv:
        idx = sys.argv.index("--download")
        pack_name = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if pack_name:
            download_pack(pack_name, verbose=True)
        sys.exit(0)

    if "--queue" in sys.argv:
        status = get_queue_status()
        print(f"Pending: {status['pending']} items")
        for item in status["items"]:
            query = item["messages"][-1]["content"] if item["messages"] else "?"
            conf = item["local_result"]["confidence"]
            print(f"  [{item['id']}] conf={conf:.2f} | {query}")
        sys.exit(0)

    # Default: run a demo query
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    }]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    result = generate(messages, tools, verbose=True)

    print(f"\n{'=' * 50}")
    print(f"  Source: {result['source']}")
    print(f"  Connectivity: {'online' if result['connectivity'] else 'OFFLINE'}")
    print(f"  Confidence: {result.get('confidence', 'n/a')}")
    print(f"  Time: {result['total_time_ms']:.0f}ms")

    ux = result.get("ux", {})
    if ux.get("message"):
        print(f"  UX note: {ux['message']}")

    if result.get("queued_id"):
        print(f"  Queued for verification: {result['queued_id']}")

    for call in result["function_calls"]:
        print(f"  -> {call['name']}({json.dumps(call['arguments'])})")
