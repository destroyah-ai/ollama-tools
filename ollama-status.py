#!/usr/bin/env python3
"""
ollama-status - Non-interactive Ollama model status checker

Similar to ollama-top but outputs machine-parseable JSON instead of a TUI.
Queries an Ollama server and reports model status for unload safety checks.

Usage:
    ollama-status [--host HOST] [--model MODEL] [--duration SECONDS] [--interval SECONDS] [--json]
    ollama-status --wait-until-idle [--model MODEL] [--timeout SECONDS] [--poll-interval SECONDS]

Modes:
    Default: Sample model activity for --duration seconds and report status
    --wait-until-idle: Block until model(s) are idle (no connections), then exit

Output (JSON):
    {
        "models": [
            {
                "name": "llama3.2:3b",
                "idle": true,           // 0 if idle, 1 if busy
                "busy": false,
                "connections": 0,       // max concurrent connections during sampling
                "size_bytes": 2017296384,  // model size + context
                "size_human": "1.88 GB",
                "context_bytes": 4096,
                "context_human": "4.0 KB",
                "loaded": true
            }
        ],
        "sampling_duration": 10,
        "timestamp": "2024-01-15T10:30:00Z"
    }

    Wait mode output (when --wait-until-idle):
    {
        "waited": true,
        "models": [...],
        "total_wait_seconds": 5.2,
        "timestamp": "2024-01-15T10:30:05Z"
    }

Exit codes:
    0 - Success (idle in wait mode, or sampling complete)
    1 - Error (connection, parsing, etc.)
    2 - Timeout in wait mode
    3 - Model not found
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


@dataclass
class ModelStatus:
    name: str
    idle: bool
    busy: bool
    connections: int
    size_bytes: int
    size_human: str
    context_bytes: int
    context_human: str
    loaded: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Add numeric idle/busy for backward compatibility
        d["idle_int"] = 1 if self.idle else 0
        d["busy_int"] = 1 if self.busy else 0
        return d


def human_size(bytes_val: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def fetch_json(url: str, timeout: int = 5) -> Optional[Dict]:
    """Fetch JSON from URL, return None on error."""
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
        return None


def get_running_models(host: str) -> List[Dict]:
    """Get currently running/loaded models from /api/ps."""
    data = fetch_json(f"{host.rstrip('/')}/api/ps")
    if data and 'models' in data:
        return data['models']
    return []


def get_all_models(host: str) -> List[Dict]:
    """Get all available models from /api/tags."""
    data = fetch_json(f"{host.rstrip('/')}/api/tags")
    if data and 'models' in data:
        return data['models']
    return []


def expiration_ts(m):
    expires_at = None
    expires_raw = m.get("expires_at", "")
    if expires_raw:
        try:
            expires_at = datetime.fromisoformat(expires_raw)
        except (ValueError, TypeError):
            pass
    return expires_at

def sample_model_activity(host: str, model_name: str, duration: float, interval: float) -> tuple:
    """
    Sample model activity over duration seconds.
    Returns (max_connections, was_busy_any_sample)
    """
    max_connections = 0
    was_busy = False
    start_time = time.time()
    end_time = start_time + duration

    prev_exp = { }
    while time.time() < end_time:
        running = get_running_models(host)
        for m in running:
            if m.get('name') == model_name:
                connections = 1  # Model is loaded = at least 1 connection
                max_connections = max(max_connections, connections)
                # Detect activity: if expires_at changed since last poll, the model
                # is actively being used (Ollama resets the timer on each request).
                expires_raw = m.get("expires_at", "")
                # expiration_ts(m)
                if model_name in prev_exp and expires_raw != prev_exp[model_name]:
                    was_busy = True
                prev_exp[model_name] = expires_raw
                break
        
        sleep_time = min(interval, end_time - time.time())
        if sleep_time > 0:
            time.sleep(sleep_time)

    return max_connections, was_busy


def get_model_size(host: str, model_name: str) -> int:
    """Get model size in bytes from /api/tags."""
    all_models = get_all_models(host)
    for m in all_models:
        if m.get('name') == model_name:
            return m.get('size', 0)
    return 0


def check_model_status(host: str, model_name: str, duration: float, interval: float) -> ModelStatus:
    """Check status of a single model with sampling."""
    model_size = get_model_size(host, model_name)
    
    max_connections, was_busy = sample_model_activity(host, model_name, duration, interval)
    
    running = get_running_models(host)
    context_bytes = 0
    loaded = False
    for m in running:
        if m.get('name') == model_name:
            loaded = True
            context_bytes = m.get('context_size', 0) or m.get('context_length', 0) or 0
            size_bytes = m.get('size', model_size)
            break
    else:
        size_bytes = model_size

    return ModelStatus(
        name=model_name,
        idle=not was_busy,
        busy=was_busy,
        connections=max_connections,
        size_bytes=size_bytes,
        size_human=human_size(size_bytes),
        context_bytes=context_bytes,
        context_human=human_size(context_bytes),
        loaded=loaded
    )


def is_model_idle(host: str, model_name: str) -> tuple:
    """
    Quick check if model is currently idle (no connections).
    Returns (is_idle, connections, loaded)
    """
    running = get_running_models(host)
    for m in running:
        if m.get('name') == model_name:
            # Model is loaded - check if it has active connections
            # In Ollama, a model in /api/ps is loaded but may not have active requests
            # We consider it "idle" if loaded but no active generation
            # For safety, we treat any loaded model as potentially busy
            return False, 1, True
    return True, 0, False


def wait_until_idle(host: str, model_names: List[str], timeout: float, poll_interval: float) -> Dict[str, Any]:
    """
    Block until all specified models are idle (not loaded).
    Returns dict with wait results.
    """
    start_time = time.time()
    end_time = start_time + timeout if timeout > 0 else float('inf')
    
    # Initial check
    all_idle = True
    model_statuses = {}
    
    while time.time() < end_time:
        all_idle = True
        model_statuses = {}
        
        for model_name in model_names:
            is_idle, connections, loaded = is_model_idle(host, model_name)
            model_statuses[model_name] = {
                "idle": is_idle,
                "connections": connections,
                "loaded": loaded
            }
            if not is_idle:
                all_idle = False
        
        if all_idle:
            break
            
        time.sleep(poll_interval)
    
    total_wait = time.time() - start_time
    timed_out = not all_idle and timeout > 0
    
    # Get final detailed status for each model
    final_models = []
    for model_name in model_names:
        status = model_statuses.get(model_name, {})
        # Get full status for output
        full_status = check_model_status(host, model_name, 0.1, 0.1)  # Quick sample
        final_models.append(full_status.to_dict())
    
    return {
        "waited": True,
        "models": final_models,
        "total_wait_seconds": round(total_wait, 2),
        "timed_out": timed_out,
        "all_idle": all_idle,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def main():
    parser = argparse.ArgumentParser(
        description='Non-interactive Ollama model status checker for unload safety',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    parser.add_argument('--wait-until-idle', action='store_true',
                        help='Block until model(s) are idle (not loaded), then exit')
    
    # Connection options
    parser.add_argument('--host', default=os.environ.get('OLLAMA_HOST', 'http://localhost:11434'),
                        help='Ollama host URL (default: $OLLAMA_HOST or http://localhost:11434)')
    
    # Model selection
    parser.add_argument('--model', help='Specific model to check (default: all loaded models)')
    parser.add_argument('--all', action='store_true',
                        help='Check all available models, not just loaded ones')
    
    # Sampling mode options
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Sampling duration in seconds (default: 10)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (default: 1)')
    
    # Wait mode options
    parser.add_argument('--timeout', type=float, default=300.0,
                        help='Max wait time in seconds for --wait-until-idle (default: 300, 0=infinite)')
    parser.add_argument('--poll-interval', type=float, default=1.0,
                        help='Poll interval in seconds for --wait-until-idle (default: 1)')
    
    # Output options
    parser.add_argument('--json', action='store_true', default=True,
                        help='Output JSON (default: true)')
    parser.add_argument('--no-json', action='store_false', dest='json',
                        help='Output simple text format')
    
    args = parser.parse_args()

    # Validate host
    host = args.host.rstrip('/')
    
    # Test connection
    test = fetch_json(f"{host}/api/tags")
    if test is None:
        print(json.dumps({"error": f"Cannot connect to Ollama at {host}"}), file=sys.stderr)
        sys.exit(1)

    # Determine which models to check
    if args.model:
        models_to_check = [args.model]
    elif args.all:
        models_to_check = [m['name'] for m in get_all_models(host)]
    else:
        # Default: check only currently loaded models
        models_to_check = [m['name'] for m in get_running_models(host)]

    if not models_to_check:
        output = {
            "models": [],
            "sampling_duration": args.duration,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if args.wait_until_idle:
            output = {
                "waited": True,
                "models": [],
                "total_wait_seconds": 0,
                "all_idle": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print("No models to check")
        sys.exit(0)

    # WAIT MODE
    if args.wait_until_idle:
        result = wait_until_idle(host, models_to_check, args.timeout, args.poll_interval)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["all_idle"]:
                print(f"All models idle after {result['total_wait_seconds']}s")
            else:
                print(f"Timeout after {result['total_wait_seconds']}s - models still busy")
                for m in result["models"]:
                    state = "IDLE" if m.get("idle") else "BUSY"
                    print(f"  {m['name']}: {state}")
        
        # Exit codes for wait mode
        if result.get("timed_out"):
            sys.exit(2)  # Timeout
        sys.exit(0)  # Success - all idle

    # SAMPLING MODE (default)
    results = []
    for model_name in models_to_check:
        try:
            status = check_model_status(host, model_name, args.duration, args.interval)
            results.append(status.to_dict())
        except Exception as e:
            results.append({
                "name": model_name,
                "error": str(e),
                "idle": True,
                "busy": False,
                "connections": 0,
                "size_bytes": 0,
                "size_human": "0 B",
                "context_bytes": 0,
                "context_human": "0 B",
                "loaded": False
            })

    output = {
        "models": results,
        "sampling_duration": args.duration,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        for m in results:
            if 'error' in m:
                print(f"{m['name']}: ERROR - {m['error']}")
            else:
                state = "BUSY" if m['busy'] else "IDLE"
                print(f"{m['name']}: {state} | connections={m['connections']} | size={m['size_human']} | ctx={m['context_human']} | loaded={m['loaded']}")

    sys.exit(0)


if __name__ == '__main__':
    main()

"""
import aiohttp
import psutil
    def get_system(self) -> SystemInfo:
        #Get CPU and RAM usage via psutil.
        vm = psutil.virtual_memory()
        return SystemInfo(
            cpu_pct=psutil.cpu_percent(interval=None),
            ram_used=vm.used,
            ram_total=vm.total,
        )
"""
