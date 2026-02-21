
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re, math
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

DEFAULT_SYSTEM_PROMPT = (
    "You are a function-calling assistant. "
    "Return function calls only. "
    "For multi-intent requests, return one call per intent in the same order as the user. "
    "Use exact argument values from the user text (no paraphrasing, no invented values). "
    "Use lowercase for text arguments. "
    "Do not add punctuation or articles. "
    "Include all required parameters for each call. "
    "If a required value is missing or unclear, return no call for that intent."
)
SYSTEM_PROMPT = os.environ.get("FUNCTIONGEMMA_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


def set_system_prompt(prompt):
    """Override the active system prompt used for on-device calls."""
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = (prompt or DEFAULT_SYSTEM_PROMPT).strip()


def get_system_prompt():
    """Get the active system prompt used for on-device calls."""
    return SYSTEM_PROMPT


_ACTION_VERBS = {
    "set", "get", "check", "send", "text", "play", "find", "look", "search",
    "remind", "create", "call", "wake", "tell", "show", "open", "start",
    "stop", "turn", "book", "order", "buy", "schedule", "cancel", "delete",
    "message", "put", "make", "add", "read", "write", "run", "track",
    "navigate", "translate", "convert", "calculate", "launch", "toggle",
}


def _split_intents(query):
    """Split a multi-intent query using commas/then and `and` before action verbs."""
    text = (query or "").strip()
    if not text:
        return []

    first_pass = re.split(r"\s*(?:,|;|\bthen\b)\s*", text, flags=re.IGNORECASE)
    verb_pattern = "|".join(sorted(_ACTION_VERBS, key=len, reverse=True))

    intents = []
    for chunk in first_pass:
        chunk = chunk.strip()
        if not chunk:
            continue
        split_on_and = re.split(
            rf"\s+and\s+(?=(?:{verb_pattern})\b)",
            chunk,
            flags=re.IGNORECASE,
        )
        intents.extend([part.strip(" ,") for part in split_on_and if part.strip(" ,")])

    return intents if intents else [text]


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


def _simplify_query(query):
    """Strip filler words for a cleaner prompt to the model."""
    q = re.sub(
        r'\b(please|could you|can you|i want to|i need to|i\'d like to)\b',
        '', query, flags=re.IGNORECASE
    )
    q = re.sub(r'\s+', ' ', q).strip()
    return q if q else query


def _extract_string_value(query, param_name, param_desc):
    """Extract a string param value from the user's query.

    Uses preposition-based splitting and param hints to find the most
    likely value. Fully generic — no tool-specific logic.
    """
    q = query.strip().rstrip(".!?")

    # Preposition-based extraction: split on common preps and take segments
    # Map of prepositions → what typically follows them
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
            # Remove trailing prepositional phrases (e.g., "Paris at 3:00 PM" → "Paris")
            # Only keep text before the next preposition
            sub = re.split(r'\s+(?:in|at|to|for|about|saying|and)\s+', segment, maxsplit=1, flags=re.IGNORECASE)
            candidates.append(sub[0].strip())

    # Also extract proper nouns (capitalized words not at sentence start)
    words = q.split()
    proper_nouns = [w.strip(".,!?;:'\"") for w in words[1:] if w[0].isupper()] if len(words) > 1 else []

    # Extract numbers
    numbers_in_query = re.findall(r'\b\d+\b', q)

    # Extract time patterns
    time_patterns = re.findall(r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?', q)

    # Score candidates based on param name/description hints
    param_hint = (param_name + " " + (param_desc or "")).lower()

    # Direct hint matching
    if any(h in param_hint for h in ["location", "city", "place", "where"]):
        # Look for text after "in" or "at"
        for prep_re in [r'\bin\s+', r'\bat\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = re.split(r'\s+(?:and|then|,|;)\s+', parts[1], maxsplit=1, flags=re.IGNORECASE)[0]
                val = val.strip().rstrip(".!?,;")
                if val:
                    return val
        # Fallback: proper nouns
        if proper_nouns:
            return " ".join(proper_nouns)

    if any(h in param_hint for h in ["recipient", "contact", "name", "person", "who"]):
        # Look for proper nouns
        if proper_nouns:
            return proper_nouns[0]
        # After "to"
        for prep_re in [r'\bto\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = parts[1].split()[0].strip(".,!?;:'\"") if parts[1].strip() else ""
                if val:
                    return val

    if any(h in param_hint for h in ["message", "text", "body", "content"]):
        # After "saying" or "that says"
        for prep_re in [r'\bsaying\s+', r'\bthat says\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = parts[1].strip().rstrip(".!?,;")
                if val:
                    return val

    if any(h in param_hint for h in ["song", "music", "playlist", "track"]):
        # After "play" or "listen to"
        for prep_re in [r'\bplay\s+', r'\blisten to\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = parts[1].strip().rstrip(".!?,;")
                # Remove filler like "some" or "a"
                val = re.sub(r'^(some|a|an)\s+', '', val, flags=re.IGNORECASE)
                # Strip trailing generic words like "music", "song"
                val = re.sub(r'\s+(?:music|song|playlist|track)s?\s*$', '', val, flags=re.IGNORECASE)
                if val:
                    return val

    if any(h in param_hint for h in ["time", "when", "schedule"]):
        if time_patterns:
            return time_patterns[0]
        # After "at" that looks like a time
        for prep_re in [r'\bat\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = re.split(r'\s+(?:and|then|,|;)\s+', parts[1], maxsplit=1, flags=re.IGNORECASE)[0]
                val = val.strip().rstrip(".!?,;")
                if val and re.search(r'\d', val):
                    return val

    if any(h in param_hint for h in ["title", "subject", "topic", "reminder"]):
        # After "about" or "to" (for reminders like "remind me to X")
        for prep_re in [r'\babout\s+', r'\bto\s+']:
            parts = re.split(prep_re, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                val = re.split(r'\s+(?:at|in|on|by|for)\s+', parts[1], maxsplit=1, flags=re.IGNORECASE)[0]
                val = val.strip().rstrip(".!?,;")
                val = re.sub(r'^(the|a|an)\s+', '', val, flags=re.IGNORECASE)
                if val:
                    return val

    if any(h in param_hint for h in ["query", "search", "look", "find"]):
        # Proper nouns or text after action verb
        if proper_nouns:
            return proper_nouns[0]
        # After the action verb, take the first noun-like word
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
        # Look for "H:MM AM/PM" or "H AM/PM"
        m = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', q)
        if m:
            hour = int(m.group(1))
            ampm = (m.group(3) or "").upper()
            if ampm == "PM" and hour < 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0
            return hour
        # "H AM/PM" without minutes
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
        # If no colon, assume 0 minutes (e.g., "10 AM" → minute=0)
        return 0

    if any(h in param_hint for h in ["duration", "minutes", "timer", "length"]):
        # Find the number closest to time-related words
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

    Takes a call that has already been through _repair_call (or raw from model),
    checks for missing or suspicious values, and tries to fill them from the query.
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

            # Check if the value (or a core part of it) appears in the query
            val_words = set(re.findall(r'[a-zA-Z]{2,}', val_lower))
            stop = {"the", "a", "an", "is", "in", "at", "to", "for", "of", "and", "or", "my", "me", "i",
                    "reminder", "about", "set", "create", "get", "send", "play", "find", "search"}
            meaningful_words = val_words - stop

            # Also check if it looks like a hallucinated datetime (ISO format not in query)
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
                # e.g., "3200" instead of "3:00 PM", "07:00" instead of "7:00 AM"
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


def generate_cactus(messages, tools, verbose=False):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    if verbose:
        print(f"  [cactus] raw output: {raw_str!r}")

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        if verbose:
            print(f"  [cactus] JSON parse FAILED — raw was: {raw_str!r}")
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "_parse_failed": True,
        }

    if verbose:
        print(f"  [cactus] success={raw.get('success')}  cloud_handoff={raw.get('cloud_handoff')}")
        print(f"  [cactus] response text: {raw.get('response')!r}")
        print(f"  [cactus] confidence={raw.get('confidence')}  ttft={raw.get('time_to_first_token_ms')}ms  total={raw.get('total_time_ms')}ms")
        print(f"  [cactus] prefill_tokens={raw.get('prefill_tokens')}  decode_tokens={raw.get('decode_tokens')}  decode_tps={raw.get('decode_tps')}")
        if raw.get("function_calls"):
            for fc in raw["function_calls"]:
                print(f"  [cactus] extracted call: {fc['name']}({json.dumps(fc.get('arguments', {}))})")
        else:
            print(f"  [cactus] extracted calls: (none)")

    result = {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }
    if raw.get("cloud_handoff"):
        result["_cloud_handoff"] = True
    return result


def generate_cloud(messages, tools):
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

    contents = [m["content"] for m in messages if m["role"] == "user"]

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

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def _extract_calls_from_response(response_text, tools):
    """Try to extract function calls from malformed response text.

    FunctionGemma sometimes outputs calls in non-standard formats like:
      call:get_weather{location:"Paris"}
      <start_function_declaration>call:create_reminder{}
    This parser tries to recover those calls generically.
    Also handles truncated calls where the value is missing (e.g., call:get_weather{location:})
    """
    if not response_text:
        return []

    # Normalize fullwidth colons (U+FF1A) to regular colons
    response_text = response_text.replace('\uff1a', ':')

    tool_names = {t["name"] for t in tools}
    calls = []

    # Pattern: call:function_name{...} or call:function_name({...})
    # Also match truncated calls with empty/missing values
    for m in re.finditer(r'call:(\w+)\s*\{([^}]*)\}?', response_text):
        fname = m.group(1)
        if fname not in tool_names:
            continue
        args_str = m.group(2).strip()
        args = {}
        if args_str:
            # First, replace <escape> tags with quotes for easier parsing
            clean_args = args_str.replace('<escape>', '"')
            # Parse key:value or key:"value" pairs
            for kv in re.finditer(r'(\w+)\s*:\s*"([^"]*)"', clean_args):
                key = kv.group(1)
                val = kv.group(2).strip()
                # Try to convert to int if it looks numeric
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                args[key] = val
            # Also try unquoted values (integers, etc.) for keys not yet found
            for kv in re.finditer(r'(\w+)\s*:\s*(-?\d+(?:\.\d+)?)', clean_args):
                key = kv.group(1)
                if key not in args:  # Don't override quoted values
                    val = kv.group(2)
                    try:
                        val = int(val)
                    except ValueError:
                        val = float(val)
                    args[key] = val
        calls.append({"name": fname, "arguments": args})

    # Also try to find truncated calls from raw token output
    # Pattern: function names that appear after "call:" even without proper braces
    if not calls:
        for m in re.finditer(r'call:(\w+)', response_text):
            fname = m.group(1)
            if fname in tool_names:
                calls.append({"name": fname, "arguments": {}})
                break  # only take the first one

    return calls


def _run_on_device(user_content, tools, verbose=False):
    """Run a single on-device inference. Returns (calls_list, time_ms, parse_failed).

    Applies post-processing to fill missing/hallucinated args from the query.
    """
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    # Collect tokens for debugging
    _tokens = []
    def _token_callback(token_text, token_id, user_data):
        _tokens.append({"id": token_id, "text": token_text})

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_content}],
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=5,
        callback=_token_callback,
    )
    cactus_destroy(model)

    if verbose:
        print(f"  [on-device] input: {user_content!r}")
        print(f"  [on-device] raw: {raw_str!r}")
        # Show decoded token stream
        token_texts = [t["text"].decode("utf-8", errors="replace") if isinstance(t["text"], bytes) else str(t["text"]) for t in _tokens]
        print(f"  [on-device] token_count={len(_tokens)} token_stream: {''.join(token_texts)}")
        print(f"  [on-device] token_ids: {[t['id'] for t in _tokens]}")

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        if verbose:
            print(f"  [on-device] JSON parse FAILED, trying raw extraction")
        # Try to extract calls from the malformed raw string
        extracted = _extract_calls_from_response(raw_str, tools)
        # Also try from the reconstructed token stream (has the raw call:func{} format)
        if not extracted and _tokens:
            token_text = "".join(
                t["text"].decode("utf-8", errors="replace") if isinstance(t["text"], bytes) else str(t["text"])
                for t in _tokens
            )
            extracted = _extract_calls_from_response(token_text, tools)
            if extracted and verbose:
                print(f"  [on-device] rescued {len(extracted)} call(s) from token stream")
        elif extracted and verbose:
            print(f"  [on-device] rescued {len(extracted)} call(s) from raw string")
        if extracted:
            # Post-process to fill missing args from query
            postprocessed = []
            for c in extracted:
                pp = _postprocess_call(c, tools, user_content, verbose=verbose)
                if pp:
                    postprocessed.append(pp)
            if verbose and postprocessed:
                for fc in postprocessed:
                    print(f"  [on-device] postprocessed call: {fc['name']}({json.dumps(fc.get('arguments', {}))})")
            return postprocessed, 0, not bool(postprocessed)
        return [], 0, True

    time_ms = raw.get("total_time_ms", 0)
    calls = raw.get("function_calls", [])

    if verbose:
        print(f"  [on-device] confidence={raw.get('confidence')} time={time_ms}ms calls={len(calls)}")
        print(f"  [on-device] cloud_handoff={raw.get('cloud_handoff')} success={raw.get('success')}")
        print(f"  [on-device] prefill_tokens={raw.get('prefill_tokens')} decode_tokens={raw.get('decode_tokens')}")
        print(f"  [on-device] response_text={raw.get('response')!r}")
        for fc in calls:
            print(f"  [on-device] raw_call: {fc['name']}({json.dumps(fc.get('arguments', {}))})")

    # Fallback: if no formal calls, try to extract from response text
    if not calls:
        response_text = raw.get("response", "")
        extracted = _extract_calls_from_response(response_text, tools)
        if extracted:
            calls = extracted
            if verbose:
                print(f"  [on-device] extracted {len(extracted)} call(s) from response text")

    # Post-process ALL calls to fill missing/hallucinated args
    if calls:
        postprocessed = []
        for c in calls:
            pp = _postprocess_call(c, tools, user_content, verbose=verbose)
            if pp:
                postprocessed.append(pp)
        if verbose:
            for fc in postprocessed:
                print(f"  [on-device] final_call: {fc['name']}({json.dumps(fc.get('arguments', {}))})")
        return postprocessed, time_ms, False

    return calls, time_ms, False


def generate_hybrid(messages, tools, confidence_threshold=0.99, verbose=False):
    """
    Aggressive on-device strategy:
    1. Try full query on-device (handles both single and multi-call)
    2. If that fails, split into sub-queries and try each on-device
    3. Retry with simplified query
    4. Cloud fallback only when on-device produces nothing usable
    """
    user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
    total_time = 0

    if verbose:
        tool_names = [t["name"] for t in tools]
        print(f"\n  [hybrid] query: {user_content!r}")
        print(f"  [hybrid] tools: {tool_names}")

    # Pre-compute sub-queries to detect multi-intent
    sub_queries = _split_intents(user_content)
    expected_intents = len(sub_queries)

    # === Attempt 1: Full query, no splitting ===
    if verbose:
        print(f"  [hybrid] attempt 1: full query (expected intents: {expected_intents})")
    calls, t, parse_failed = _run_on_device(user_content, tools, verbose=verbose)
    total_time += t

    # _run_on_device already applies _postprocess_call (which includes _repair_call)
    if not parse_failed and calls:
        # Accept attempt 1 only if we got enough calls for the detected intents
        if len(calls) >= expected_intents:
            if verbose:
                print(f"  [hybrid] decision: ON-DEVICE (attempt 1, {len(calls)} calls)")
            return {
                "function_calls": calls,
                "total_time_ms": total_time,
                "confidence": 1.0,
                "source": "on-device",
            }
        elif verbose:
            print(f"  [hybrid] attempt 1 partial: got {len(calls)} calls but expected {expected_intents}, trying split")

    # === Attempt 2: Split into sub-queries ===
    if expected_intents > 1:
        if verbose:
            print(f"  [hybrid] attempt 2: split into {len(sub_queries)} sub-queries: {sub_queries}")

        all_calls = []
        for sq in sub_queries:
            sq_calls, t, _ = _run_on_device(sq, tools, verbose=verbose)
            total_time += t
            all_calls.extend(sq_calls)

        # Accept split only if we recovered enough calls (>= 60% of expected)
        min_required = max(1, math.ceil(expected_intents * 0.6))
        if len(all_calls) >= min_required:
            if verbose:
                print(f"  [hybrid] decision: ON-DEVICE (attempt 2 split, {len(all_calls)}/{expected_intents} calls, min={min_required})")
            return {
                "function_calls": all_calls,
                "total_time_ms": total_time,
                "confidence": 1.0,
                "source": "on-device",
            }
        elif all_calls and verbose:
            print(f"  [hybrid] attempt 2 insufficient: got {len(all_calls)} calls but need {min_required}/{expected_intents}")

    # === Attempt 3: Simplified query ===
    simplified = _simplify_query(user_content)
    if simplified != user_content:
        if verbose:
            print(f"  [hybrid] attempt 3: simplified query: {simplified!r}")
        calls, t, _ = _run_on_device(simplified, tools, verbose=verbose)
        total_time += t

        if calls:
            if verbose:
                print(f"  [hybrid] decision: ON-DEVICE (attempt 3 simplified, {len(calls)} calls)")
            return {
                "function_calls": calls,
                "total_time_ms": total_time,
                "confidence": 1.0,
                "source": "on-device",
            }

    # === Attempt 4: Cloud fallback (last resort) ===
    if verbose:
        print(f"  [hybrid] decision: CLOUD FALLBACK (all on-device attempts failed)")

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += total_time

    # Apply postprocessing to cloud results too
    repaired_cloud = []
    for c in cloud["function_calls"]:
        r = _postprocess_call(c, tools, user_content, verbose=verbose)
        if r:
            repaired_cloud.append(r)
    cloud["function_calls"] = repaired_cloud

    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
