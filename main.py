
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"
qwen3_path = "cactus/weights/qwen3-1.7b"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


############## Post-Processing Helpers ##############

def _repair_call(call, tools):
    """Auto-repair a function call. Returns repaired call dict, or None if unrecoverable."""
    tool_map = {t["name"]: t for t in tools}
    name = call.get("name", "")

    if name not in tool_map:
        return None  # unknown tool — unrecoverable

    tool = tool_map[name]
    props = tool.get("parameters", {}).get("properties", {})
    required = set(tool.get("parameters", {}).get("required", []))
    args = dict(call.get("arguments", {}))

    for param_name, param_schema in props.items():
        if param_name not in args:
            continue
        val = args[param_name]
        param_type = param_schema.get("type", "string")

        if param_type == "integer":
            if isinstance(val, str):
                try:
                    val = int(float(val))
                except (ValueError, TypeError):
                    val = 0
            if isinstance(val, (int, float)) and val < 0:
                val = abs(int(val))
            if isinstance(val, float):
                val = int(val)
            args[param_name] = val

        elif param_type == "string" and isinstance(val, str):
            val = val.strip().strip('.,!?;:"\'-')
            val = re.sub(r'^(the|a|an)\s+', '', val, flags=re.IGNORECASE)
            # Strip action-word prefixes (e.g., "Reminder about meeting" → "meeting")
            val = re.sub(
                r'^(?:reminder|alarm|timer|search|note|message)\s+(?:about|for|to|of)\s+',
                '', val, flags=re.IGNORECASE
            )
            args[param_name] = val

    # Remove args not in schema
    valid_params = set(props.keys())
    args = {k: v for k, v in args.items() if k in valid_params}

    # Reject if any required param is missing or empty
    for r in required:
        if r not in args or args[r] is None or args[r] == "":
            return None

    return {"name": name, "arguments": args}


def _extract_string_value(query, param_name, param_desc):
    """Extract a string param value from the user's query.

    Uses preposition-based splitting and param hints to find the most
    likely value. Fully generic — no tool-specific logic.
    """
    q = query.strip().rstrip(".!?")

    # Preposition-based extraction: split on common preps and take segments
    prep_patterns = [
        (r'\bsaying\s+', None),       # "saying good morning" → message
        (r'\bthat says\s+', None),     # "that says hello"
        (r'\bcalled\s+', None),        # "called meeting"
        (r'\bnamed\s+', None),         # "named Bob"
        (r'\babout\s+', None),         # "about the meeting"
        (r'\bfor\s+', None),           # "for 5 minutes"
        (r'\bin\s+', None),            # "in Paris"
        (r'\bat\s+', None),            # "at 3:00 PM"
        (r'\bto\s+', None),            # "to Alice"
    ]

    # Collect all candidate segments from preposition splits
    candidates = []
    for prep_re, _ in prep_patterns:
        parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            segment = parts[1].strip().rstrip(".!?,;")
            sub = re.split(r'\s+(?:in|at|to|for|about|saying|and)\s+', segment, maxsplit=1, flags=re.IGNORECASE)
            candidates.append(sub[0].strip())

    # Also extract proper nouns (capitalized words not at sentence start)
    words = q.split()
    proper_nouns = [w.strip(".,!?;:'\"") for w in words[1:] if w[0].isupper()] if len(words) > 1 else []

    # Extract time patterns
    time_patterns = re.findall(r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?', q)

    # Score candidates based on param name/description hints
    param_hint = (param_name + " " + (param_desc or "")).lower()

    if any(h in param_hint for h in ["location", "city", "place", "where"]):
        for prep_re in [r'\bin\s+', r'\bat\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = re.split(r'\s+(?:and|then|,|;)\s+', parts[1], maxsplit=1, flags=re.IGNORECASE)[0]
                val = val.strip().rstrip(".!?,;")
                if val:
                    return val
        if proper_nouns:
            return " ".join(proper_nouns)

    if any(h in param_hint for h in ["recipient", "contact", "name", "person", "who"]):
        if proper_nouns:
            return proper_nouns[0]
        for prep_re in [r'\bto\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = parts[1].split()[0].strip(".,!?;:'\"") if parts[1].strip() else ""
                if val:
                    return val

    if any(h in param_hint for h in ["message", "text", "body", "content"]):
        for prep_re in [r'\bsaying\s+', r'\bthat says\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = parts[1].strip().rstrip(".!?,;")
                if val:
                    return val

    if any(h in param_hint for h in ["song", "music", "playlist", "track"]):
        for prep_re in [r'\bplay\s+', r'\blisten to\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = parts[1].strip().rstrip(".!?,;")
                val = re.sub(r'^(some|a|an)\s+', '', val, flags=re.IGNORECASE)
                # Strip trailing generic words like "music", "song"
                val = re.sub(r'\s+(?:music|song|playlist|track)s?\s*$', '', val, flags=re.IGNORECASE)
                if val:
                    return val

    if any(h in param_hint for h in ["time", "when", "schedule"]):
        if time_patterns:
            return time_patterns[0]
        for prep_re in [r'\bat\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = re.split(r'\s+(?:and|then|,|;)\s+', parts[1], maxsplit=1, flags=re.IGNORECASE)[0]
                val = val.strip().rstrip(".!?,;")
                if val and re.search(r'\d', val):
                    return val

    if any(h in param_hint for h in ["title", "subject", "topic", "reminder"]):
        for prep_re in [r'\babout\s+', r'\bto\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = re.split(r'\s+(?:at|in|on|by|for)\s+', parts[1], maxsplit=1, flags=re.IGNORECASE)[0]
                val = val.strip().rstrip(".!?,;")
                val = re.sub(r'^(the|a|an)\s+', '', val, flags=re.IGNORECASE)
                if val:
                    return val

    if any(h in param_hint for h in ["query", "search", "look", "find"]):
        if proper_nouns:
            return proper_nouns[0]
        for prep_re in [r'\b(?:find|search|look up|look for)\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = re.split(r'\s+(?:in|from|on|and)\s+', parts[1], maxsplit=1, flags=re.IGNORECASE)[0]
                val = val.strip().rstrip(".!?,;")
                if val:
                    return val

    # Generic fallback: return the longest candidate
    if candidates:
        return max(candidates, key=len)

    return None


def _extract_integer_value(query, param_name, param_desc):
    """Extract an integer param value from the user's query."""
    q = query.strip()
    param_hint = (param_name + " " + (param_desc or "")).lower()

    # For hour/minute, parse time expressions
    if any(h in param_hint for h in ["hour"]):
        m = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', q)
        if m:
            hour = int(m.group(1))
            ampm = (m.group(3) or "").upper()
            if ampm == "PM" and hour < 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0
            return hour
        m = re.search(r'(\d{1,2})\s*(AM|PM|am|pm)', q)
        if m:
            hour = int(m.group(1))
            ampm = m.group(2).upper()
            if ampm == "PM" and hour < 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0
            return hour

    if any(h in param_hint for h in ["minute"]):
        m = re.search(r'(\d{1,2}):(\d{2})', q)
        if m:
            return int(m.group(2))
        return 0

    if any(h in param_hint for h in ["duration", "minutes", "timer", "length"]):
        m = re.search(r'(\d+)\s*(?:minute|min|hour|hr)', q, re.IGNORECASE)
        if m:
            return int(m.group(1))

    # Generic: find all numbers, return the first one
    nums = re.findall(r'\b(\d+)\b', q)
    if nums:
        return int(nums[0])

    return None


def _postprocess_call(call, tools, query, verbose=False):
    """Post-process a function call: fill missing/hallucinated args from the query.

    Returns the improved call dict, or None if unrecoverable.
    """
    tool_map = {t["name"]: t for t in tools}
    name = call.get("name", "")

    if name not in tool_map:
        return None

    tool = tool_map[name]
    props = tool.get("parameters", {}).get("properties", {})
    required = set(tool.get("parameters", {}).get("required", []))
    args = dict(call.get("arguments", {}))

    for param_name, param_schema in props.items():
        param_type = param_schema.get("type", "string")
        param_desc = param_schema.get("description", "")
        current_val = args.get(param_name)

        needs_fill = False

        # Case 1: Missing or empty
        if param_name not in args or current_val is None or current_val == "":
            needs_fill = True

        # Case 1b: String param got a non-string value (e.g., number)
        elif param_type == "string" and not isinstance(current_val, str):
            if verbose:
                print(f"  [postprocess] {param_name}={current_val!r} is wrong type for string param")
            needs_fill = True

        # Case 2: String value looks hallucinated (not in query)
        elif param_type == "string" and isinstance(current_val, str):
            query_lower = query.lower()
            val_lower = current_val.lower().strip()

            val_words = set(re.findall(r'[a-zA-Z]{2,}', val_lower))
            stop = {"the", "a", "an", "is", "in", "at", "to", "for", "of", "and", "or", "my", "me", "i",
                    "reminder", "about", "set", "create", "get", "send", "play", "find", "search"}
            meaningful_words = val_words - stop

            is_iso_datetime = bool(re.match(r'\d{4}-\d{2}-\d{2}', val_lower))

            if is_iso_datetime and val_lower not in query_lower:
                if verbose:
                    print(f"  [postprocess] {param_name}={current_val!r} is hallucinated ISO datetime")
                needs_fill = True
            elif meaningful_words and not any(w in query_lower for w in meaningful_words):
                if verbose:
                    print(f"  [postprocess] {param_name}={current_val!r} looks hallucinated (not in query)")
                needs_fill = True
            elif not meaningful_words and len(val_lower) > 0:
                # No meaningful alpha words — could be a numeric/time-like hallucination
                if val_lower not in query_lower:
                    if verbose:
                        print(f"  [postprocess] {param_name}={current_val!r} is non-alpha and not in query")
                    needs_fill = True

        # Case 3: Integer value that doesn't match any number in the query
        elif param_type == "integer" and isinstance(current_val, (int, float)):
            query_nums = set(int(n) for n in re.findall(r'\b(\d+)\b', query))
            if query_nums and abs(int(current_val)) not in query_nums:
                if verbose:
                    print(f"  [postprocess] {param_name}={current_val} not found in query numbers {query_nums}")
                needs_fill = True

        if needs_fill:
            if param_type == "string":
                extracted = _extract_string_value(query, param_name, param_desc)
                if extracted:
                    if verbose:
                        print(f"  [postprocess] filled {param_name}: {current_val!r} -> {extracted!r}")
                    args[param_name] = extracted
            elif param_type == "integer":
                extracted = _extract_integer_value(query, param_name, param_desc)
                if extracted is not None:
                    if verbose:
                        print(f"  [postprocess] filled {param_name}: {current_val!r} -> {extracted}")
                    args[param_name] = extracted

    # Now run through _repair_call logic for final cleanup
    call_out = {"name": name, "arguments": args}
    return _repair_call(call_out, tools)


############## Model Singletons ##############

_fgemma = None
_qwen3 = None

def _get_fgemma():
    global _fgemma
    if _fgemma is None:
        _fgemma = cactus_init(functiongemma_path)
    return _fgemma

def _get_qwen3():
    global _qwen3
    if _qwen3 is None:
        _qwen3 = cactus_init(qwen3_path)
    return _qwen3


############## FunctionGemma Output Recovery ##############

def _recover_function_calls(raw_text):
    """Extract function calls from raw/malformed FunctionGemma output."""
    if not raw_text:
        return []
    text = raw_text.replace('\uff1a', ':')
    calls = []

    for match in re.finditer(r'call:(\w+)\{([^}]*)\}', text):
        name, args_str = match.group(1), match.group(2)
        args = {}
        for am in re.finditer(r'(\w+):<escape>(.*?)<escape>', args_str):
            k, v = am.group(1), am.group(2)
            try: v = int(v)
            except ValueError:
                try: v = float(v)
                except ValueError: pass
            args[k] = v
        if not args:
            for pair in args_str.split(','):
                if ':' in pair:
                    k, v = pair.split(':', 1)
                    k, v = k.strip(), v.strip()
                    if not k: continue
                    try: v = int(v)
                    except ValueError:
                        try: v = float(v)
                        except ValueError: pass
                    args[k] = v
        if name:
            calls.append({"name": name, "arguments": args})

    if not calls:
        for match in re.finditer(r'"name"\s*:\s*"(\w+)"', text):
            name = match.group(1)
            rest = text[match.end():]
            args = {}
            for am in re.finditer(r'(\w+)[:\uff1a]<escape>(.*?)<escape>', rest):
                k, v = am.group(1), am.group(2)
                try: v = int(v)
                except ValueError:
                    try: v = float(v)
                    except ValueError: pass
                args[k] = v
                break
            if name:
                calls.append({"name": name, "arguments": args})

    return calls


def _fix_arguments(function_calls):
    """abs() for negative numbers, strip strings, flatten nested dicts."""
    for call in function_calls:
        args = call.get("arguments", {})
        for key in list(args):
            val = args[key]
            if isinstance(val, dict):
                if key in val:
                    args[key] = val[key]
                elif len(val) == 1:
                    args[key] = next(iter(val.values()))
            elif isinstance(val, (int, float)) and val < 0:
                args[key] = abs(int(val)) if isinstance(val, int) else abs(val)
            elif isinstance(val, str):
                args[key] = val.strip()
    return function_calls


############## Query Splitting (Qwen3-1.7B) ##############

_SPLIT_PROMPT = 'Split this into separate actions. Return ONLY a JSON array of strings like ["action 1", "action 2"]. No objects, no thinking.\n\nRequest: '


def split_query_qwen(query, verbose=False):
    """Use Qwen3-1.7B to split a multi-intent query into sub-queries."""
    model = _get_qwen3()
    cactus_reset(model)

    start = time.time()
    raw = cactus_complete(
        model,
        [{"role": "user", "content": _SPLIT_PROMPT + f'"{query}"'}],
        tools=[],
        max_tokens=256,
        temperature=0,
        stop_sequences=["<|im_end|>"],
    )
    split_time = (time.time() - start) * 1000

    try:
        parsed = json.loads(raw)
        response = parsed.get("response", "")
    except json.JSONDecodeError:
        response = raw

    # Strip Qwen3 thinking blocks (closed or unclosed)
    response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    response = re.sub(r'<think>.*', '', response, flags=re.DOTALL)  # unclosed

    try:
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            parts = json.loads(match.group())
            if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
                parts = [p.strip() for p in parts if p.strip()]
                if verbose:
                    print(f"  [split] qwen3 ({split_time:.0f}ms): {json.dumps(parts)}")
                return parts if parts else [query]
    except (json.JSONDecodeError, ValueError):
        pass

    if verbose:
        print(f"  [split] qwen3 ({split_time:.0f}ms) parse failed, raw: {response!r}")
    return [query]


def _might_be_multi_intent(query):
    """Fast check: does the query look like it could have multiple actions?"""
    q = query.lower()
    return ' and ' in q or ', ' in q


def generate_cactus_split(messages, tools, verbose=False):
    """Use Qwen3 to split multi-intent queries, then run each through FunctionGemma."""
    query = next((m["content"] for m in messages if m["role"] == "user"), "")

    if not _might_be_multi_intent(query):
        if verbose:
            print(f"  [split] single intent, skipping qwen3")
        return generate_cactus(messages, tools, verbose)

    sub_queries = split_query_qwen(query, verbose)

    if len(sub_queries) <= 1:
        return generate_cactus(messages, tools, verbose)

    all_calls = []
    total_time = 0
    min_confidence = 1.0
    cloud_handoff = False
    parse_failed = False

    for i, sq in enumerate(sub_queries, 1):
        if verbose:
            print(f"  [split] sub-query {i}/{len(sub_queries)}: {sq!r}")
        sub_result = generate_cactus([{"role": "user", "content": sq}], tools, verbose)
        all_calls.extend(sub_result["function_calls"])
        total_time += sub_result["total_time_ms"]
        min_confidence = min(min_confidence, sub_result["confidence"])
        if sub_result.get("_cloud_handoff"):
            cloud_handoff = True
        if sub_result.get("_parse_failed"):
            parse_failed = True

    if verbose:
        names = [c["name"] for c in all_calls]
        print(f"  [split] merged: {len(all_calls)} calls {names} | conf={min_confidence:.4f} | {total_time:.0f}ms")

    merged = {
        "function_calls": all_calls,
        "total_time_ms": total_time,
        "confidence": min_confidence,
    }
    if cloud_handoff:
        merged["_cloud_handoff"] = True
    if parse_failed:
        merged["_parse_failed"] = True
    return merged


############## On-Device Inference (FunctionGemma) ##############

_FGEMMA_SYSTEM = "Always return at least one tool. Never respond with text or questions. Use exact words from the user's message as argument values. Do not add emails, URLs, or extra text."

def generate_cactus(messages, tools, verbose=False):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_fgemma()
    cactus_reset(model)

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": _FGEMMA_SYSTEM}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        temperature=0,
        tool_rag_top_k=0,
    )

    if verbose:
        print(f"  [cactus] raw output: {raw_str!r}")

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        recovered = _fix_arguments(_recover_function_calls(raw_str))
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw_str)
        recovered_conf = float(conf_match.group(1)) if conf_match and recovered else 0
        if verbose:
            print(f"      JSON PARSE FAILED — recovered {len(recovered)} call(s) conf={recovered_conf:.4f}")
            for fc in recovered:
                print(f"      -> {fc['name']}({json.dumps(fc.get('arguments', {}))})")
        return {
            "function_calls": recovered,
            "total_time_ms": 0,
            "confidence": recovered_conf,
            "_parse_failed": not recovered,
        }

    function_calls = raw.get("function_calls", [])
    if verbose:
        conf = raw.get("confidence", 0)
        total = raw.get("total_time_ms", 0)
        resp = raw.get("response", "")
        print(f"      conf={conf:.4f}  total={total:.0f}ms  prefill={raw.get('prefill_tokens',0)}  decode={raw.get('decode_tokens',0)}  tps={raw.get('decode_tps',0):.1f}")
        if raw.get("cloud_handoff"):
            print(f"      CLOUD_HANDOFF")
        if resp:
            print(f"      response: {resp!r}")
        for fc in function_calls:
            print(f"      -> {fc['name']}({json.dumps(fc.get('arguments', {}))})")
        if not function_calls:
            print(f"      -> (no calls)")

    # Recover from response text if C++ parser found nothing
    if not function_calls:
        function_calls = _recover_function_calls(raw.get("response", ""))
        if verbose and function_calls:
            print(f"      RECOVERED {len(function_calls)} call(s) from response text")
            for fc in function_calls:
                print(f"      -> {fc['name']}({json.dumps(fc.get('arguments', {}))})")

    result = {
        "function_calls": _fix_arguments(function_calls),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }
    if raw.get("cloud_handoff"):
        result["_cloud_handoff"] = True
    return result


############## Cloud Inference (Gemini) ##############

_CLOUD_SYSTEM = "ALWAYS call ALL relevant tools for every action in the query. If the query has multiple actions, return multiple function calls. Use the shortest accurate argument values. Strip trailing punctuation. For music/song args, use only the genre or title keyword (e.g. 'jazz' not 'jazz music'). For reminder titles, preserve the action phrase as-is from the query."

def generate_cloud(messages, tools, verbose=False):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [_CLOUD_SYSTEM] + [m["content"] for m in messages if m["role"] == "user"]
    start_time = time.time()

    system_instruction = (
        "You are a precise function calling assistant. "
        "Extract exact values from the user's message. "
        "Do NOT add articles (the, a, an), punctuation, or modify the user's words. "
        "For messages, use the exact words the user specified in lowercase. "
        "When multiple actions are requested, make ALL function calls. "
        "For reminder titles, use only the core noun/phrase without articles."
    )

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=system_instruction,
        ),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    if verbose:
        print(f"  [cloud] {total_time_ms:.0f}ms")
        for fc in function_calls:
            print(f"  [cloud] -> {fc['name']}({json.dumps(fc.get('arguments', {}))})")
        if not function_calls:
            print(f"  [cloud] -> (no calls)")

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


############## Validation & Hybrid Routing ##############

def _validate_local(local, tools, query=""):
    """Check if local result is trustworthy: has calls, known tools, required args, no hallucinations."""
    calls = local.get("function_calls", [])
    if not calls:
        return False, "no function calls"
    tool_map = {t["name"]: t for t in tools}
    query_lower = query.lower()
    for fc in calls:
        tool_def = tool_map.get(fc["name"])
        if not tool_def:
            return False, f"unknown tool {fc['name']}"
        required = tool_def.get("parameters", {}).get("required", [])
        args = fc.get("arguments", {})
        for req in required:
            if req not in args or args[req] == "" or args[req] is None:
                return False, f"missing required arg '{req}' in {fc['name']}"
        if query_lower:
            for k, v in args.items():
                if isinstance(v, dict) or isinstance(v, list):
                    return False, f"non-primitive arg {k}={v!r}"
                if isinstance(v, str) and v.strip():
                    if v.lower() not in query_lower:
                        return False, f"hallucinated arg {k}={v!r}"
                elif isinstance(v, (int, float)) and v != 0:
                    if str(int(v)) not in query:
                        return False, f"hallucinated arg {k}={v}"
    return True, None


def generate_hybrid(messages, tools, confidence_threshold=0.95, verbose=False):
    """Hybrid: on-device first (with Qwen3 splitting + postprocessing), cloud fallback if untrusted."""
    query = next((m["content"] for m in messages if m["role"] == "user"), "")
    local = generate_cactus_split(messages, tools, verbose=verbose)

    # Apply postprocessing to fix missing/hallucinated args from on-device
    postprocessed = []
    for c in local["function_calls"]:
        pp = _postprocess_call(c, tools, query, verbose=verbose)
        if pp:
            postprocessed.append(pp)
    local["function_calls"] = postprocessed

    valid, reason = _validate_local(local, tools, query=query)

    if verbose:
        print(f"\n  [hybrid] query: {query!r}")
        print(f"  [hybrid] tools: {[t['name'] for t in tools]}")
        print(f"  [hybrid] conf={local['confidence']:.4f}  valid={valid}  time={local['total_time_ms']:.0f}ms")
        for fc in local["function_calls"]:
            print(f"  [hybrid] local -> {fc['name']}({json.dumps(fc.get('arguments', {}))})")
        if not local["function_calls"]:
            print(f"  [hybrid] local -> (no calls)")
        if not valid:
            print(f"  [hybrid] validation: {reason}")

    if valid and local["confidence"] >= confidence_threshold:
        if verbose:
            print(f"  [hybrid] decision: ON-DEVICE")
        local["source"] = "on-device"
        return local

    if verbose:
        print(f"  [hybrid] decision: CLOUD FALLBACK")

    cloud = generate_cloud(messages, tools, verbose=verbose)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += local["total_time_ms"]

    # Apply postprocessing to cloud results too
    repaired_cloud = []
    for c in cloud["function_calls"]:
        r = _postprocess_call(c, tools, query, verbose=verbose)
        if r:
            repaired_cloud.append(r)
    cloud["function_calls"] = repaired_cloud

    return cloud


############## Example usage ##############

if __name__ == "__main__":
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

    result = generate_hybrid(messages, tools, verbose=True)
    print(f"\n=== Result ===")
    print(f"Source: {result.get('source')}")
    print(f"Time: {result['total_time_ms']:.0f}ms")
    for call in result["function_calls"]:
        print(f"  {call['name']}({json.dumps(call['arguments'])})")
