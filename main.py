
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_reset, cactus_destroy
from google import genai
from google.genai import types


# ============================================================
# Model caching — avoid repeated init/destroy per call
# ============================================================
_cached_model = None


def _get_model():
    """Return a cached FunctionGemma model handle, initializing on first call."""
    global _cached_model
    if _cached_model is None:
        _cached_model = cactus_init(functiongemma_path)
    return _cached_model


def _reset_model():
    """Clear KV cache between unrelated calls (reuse the model handle)."""
    global _cached_model
    if _cached_model is not None:
        cactus_reset(_cached_model)


# ============================================================
# Task complexity classifier (zeroclaw hint-routing concept)
# ============================================================
_ACTION_VERBS = {
    "send", "text", "message", "get", "check", "set", "play",
    "find", "search", "look", "remind", "create", "call",
    "wake", "tell", "ask", "start", "stop", "turn", "open",
    "read", "write", "make", "timer", "alarm", "weather",
}

_MULTI_ACTION_PATTERNS = [
    " and ", " also ", " then ", " plus ",
    " as well as ", " along with ", " additionally ",
]


def classify_task_complexity(messages, tools):
    """Classify request complexity: 'easy', 'medium', or 'hard'.

    Runs in pure Python with zero latency — no model calls.
    Ported from zeroclaw's hint-based routing concept.
    """
    user_msg = " ".join(m["content"] for m in messages if m["role"] == "user").strip().lower()
    num_tools = len(tools)

    # Count multi-action conjunctions
    conjunction_count = sum(user_msg.count(p) for p in _MULTI_ACTION_PATTERNS)

    # Count comma-separated segments that contain action verbs
    segments = user_msg.split(",")
    verb_segments = sum(
        1 for seg in segments
        if any(v in seg.lower().split() for v in _ACTION_VERBS)
    )

    estimated_calls = max(1, conjunction_count + 1, verb_segments)

    if estimated_calls >= 2:
        return "hard"
    if num_tools == 1:
        return "easy"
    return "medium"


# ============================================================
# Local execution with 4-gate validation (zeroclaw reliable provider pattern)
# ============================================================

def _run_single_local(messages, tools, conf_threshold, max_tokens=256, system_prompt=None):
    """Single FunctionGemma call with structural validation.

    Returns (result_dict, raw_calls) if valid, (None, []) if should fall back.
    Uses cached model + reset to avoid init/destroy overhead (~50-100ms savings).
    """
    model = _get_model()
    _reset_model()  # Clear KV cache to prevent sequential bleed

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    system_msg = {
        "role": "system",
        "content": system_prompt or (
            "You are a function-calling assistant. "
            "Always respond with exactly the required function call(s). "
            "Match parameter names and types precisely."
        ),
    }

    raw_str = cactus_complete(
        model,
        [system_msg] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=1,
        tool_rag_top_k=0,  # Show ALL tools (default 2 could miss correct one)
        confidence_threshold=conf_threshold,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return None, []

    function_calls = raw.get("function_calls", [])
    confidence = raw.get("confidence", 0)
    total_time = raw.get("total_time_ms", 0)
    cloud_handoff = raw.get("cloud_handoff", False)

    # Gate 1: Cloud handoff signal
    if cloud_handoff:
        return None, []

    # Gate 2: Confidence threshold
    if confidence < conf_threshold:
        return None, []

    # Gate 3: Non-empty function calls
    if not function_calls:
        return None, []

    # Gate 4: Schema validation + type coercion
    valid_tool_names = {t["name"] for t in tools}
    for call in function_calls:
        if call.get("name") not in valid_tool_names:
            return None, []

        tool_def = next((t for t in tools if t["name"] == call["name"]), None)
        if tool_def:
            required_params = tool_def["parameters"].get("required", [])
            call_args = call.get("arguments", {})
            props = tool_def["parameters"].get("properties", {})
            for param in required_params:
                if param not in call_args:
                    return None, []
                val = call_args[param]
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    return None, []

            # Type coercion + sign fix: FunctionGemma sometimes returns
            # strings for int/number params, or negatives (e.g. -5 for 5 minutes)
            user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
            for param, pinfo in props.items():
                if param not in call_args:
                    continue
                val = call_args[param]
                expected_type = pinfo.get("type", "").lower()
                if expected_type == "integer":
                    if isinstance(val, str):
                        try:
                            val = int(val)
                        except (ValueError, TypeError):
                            continue
                    if isinstance(val, (int, float)):
                        call_args[param] = abs(int(val))
                elif expected_type == "number":
                    if isinstance(val, str):
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            continue
                    if isinstance(val, (int, float)):
                        call_args[param] = abs(float(val))

            # AM/PM post-processing for hour params:
            # FunctionGemma may return hour=10 for "10:15 PM" (should be 22)
            for param, pinfo in props.items():
                if param not in call_args:
                    continue
                if pinfo.get("type", "").lower() == "integer" and "hour" in param.lower():
                    hour_val = call_args[param]
                    if isinstance(hour_val, int):
                        if "pm" in user_text and 1 <= hour_val <= 11:
                            call_args[param] = hour_val + 12
                        elif "am" in user_text and hour_val == 12:
                            call_args[param] = 0

    result = {
        "function_calls": function_calls,
        "total_time_ms": total_time,
        "confidence": confidence,
    }
    return result, function_calls


def _calls_agree(calls_a, calls_b):
    """Check if two sets of function calls agree on names and key argument values."""
    if len(calls_a) != len(calls_b):
        return False
    for a, b in zip(calls_a, calls_b):
        if a.get("name") != b.get("name"):
            return False
        # Check that main argument values match
        args_a = a.get("arguments", {})
        args_b = b.get("arguments", {})
        for key in args_a:
            if key in args_b:
                va = str(args_a[key]).strip().lower()
                vb = str(args_b[key]).strip().lower()
                if va != vb:
                    return False
    return True


def _semantic_check(messages, calls, tools):
    """Lightweight semantic validation of argument values.

    Catches cases where the model is structurally correct but semantically wrong
    (e.g., user says '3:00 PM' but model returns 'time': '30 minutes').
    """
    user_msg = " ".join(m["content"] for m in messages if m["role"] == "user").lower()

    for call in calls:
        tool = next((t for t in tools if t["name"] == call["name"]), None)
        if not tool:
            return False
        args = call.get("arguments", {})
        props = tool["parameters"].get("properties", {})

        # Extract all numbers from user message for validation
        user_nums = set(int(n) for n in re.findall(r'\b\d+\b', user_msg))
        # Extract content words from user message (for string value validation)
        _stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                       "being", "have", "has", "had", "do", "does", "did", "will",
                       "would", "could", "should", "may", "might", "shall", "can",
                       "for", "and", "nor", "but", "or", "yet", "so", "at", "by",
                       "in", "of", "on", "to", "up", "it", "its", "my", "me", "we",
                       "our", "you", "your", "he", "she", "his", "her", "they",
                       "them", "their", "this", "that", "what", "which", "who",
                       "how", "when", "where", "why", "not", "no", "yes", "all",
                       "some", "any", "each", "from", "with", "about", "into",
                       "set", "get", "send", "play", "find", "check", "make",
                       "saying", "said", "tell", "ask", "let", "know"}
        user_content_words = {w for w in re.findall(r'[a-z]+', user_msg)
                              if len(w) > 2 and w not in _stop_words}

        for param, pinfo in props.items():
            if param not in args:
                continue
            val = args[param]
            ptype = pinfo.get("type", "").lower()

            # Check 1: Time-related string params should match user's time expressions
            if ptype == "string" and isinstance(val, str) and "time" in param.lower():
                time_pats = re.findall(r'\d{1,2}:\d{2}\s*[APap][Mm]|\d{1,2}\s*[APap][Mm]', user_msg)
                if time_pats:
                    val_lower = val.lower()
                    if not any(tp.lower().replace(" ", "") in val_lower.replace(" ", "") for tp in time_pats):
                        return False

            # Check 2: String params — at least one content word should appear in user message
            # Catches: wrong recipient names, wrong song titles, wrong message content
            if ptype == "string" and isinstance(val, str) and len(val.strip()) > 0:
                val_words = {w for w in re.findall(r'[a-z]+', val.lower())
                             if len(w) > 2 and w not in _stop_words}
                if val_words and user_content_words:
                    if not val_words & user_content_words:
                        return False

            # Check 3: Integer params should match numbers from user message
            # Catches: minute=120 when user said "8:15", minutes=5 when user said "10"
            if ptype == "integer" and isinstance(val, (int, float)):
                int_val = abs(int(val))
                # 0 is a common default (e.g., minute=0 for "10 AM"), allow it
                if int_val != 0 and user_nums and int_val not in user_nums:
                    return False

                # Check 3b: Range validation for common param patterns
                param_lower = param.lower()
                if "hour" in param_lower and not (0 <= int_val <= 23):
                    return False
                if "minute" in param_lower and not (0 <= int_val <= 59):
                    return False

    return True


def run_local_with_validation(messages, tools, conf_threshold, max_tokens=256, retries=1):
    """Run FunctionGemma locally with validation and retry-on-failure.

    If the first attempt fails semantic/structural validation, reset the KV cache
    and retry with a slightly different prompt. This lightweight "agentic loop"
    catches FunctionGemma's non-deterministic failures without cloud cost.
    """
    # Alternate system prompts for retries (variation can shake loose correct answer)
    system_prompts = [
        "You are a function-calling assistant. Always respond with exactly the required function call(s). Match parameter names and types precisely.",
        "You MUST call one of the available functions. Extract all parameter values directly from the user's message. Do not ask for clarification. Do not refuse.",
    ]

    for attempt in range(1 + retries):
        prompt = system_prompts[attempt % len(system_prompts)]
        result, calls = _run_single_local(
            messages, tools, conf_threshold, max_tokens, system_prompt=prompt,
        )
        if result is None:
            continue

        # Semantic check: catch structurally valid but semantically wrong results
        if not _semantic_check(messages, calls, tools):
            continue

        return result

    return None


# ============================================================
# Request decomposition for multi-tool (hard) tasks
# ============================================================

def _tool_relevance_score(segment, tool):
    """Score how relevant a tool is to a text segment by keyword overlap."""
    score = 0
    seg_lower = segment.lower()

    # Tool name keywords (high weight)
    for word in tool["name"].replace("_", " ").lower().split():
        if word in seg_lower:
            score += 3

    # Description keywords
    for word in tool["description"].lower().split():
        if len(word) > 3 and word in seg_lower:
            score += 1

    # Parameter description keywords
    for param_info in tool["parameters"].get("properties", {}).values():
        for word in param_info.get("description", "").lower().split():
            if len(word) > 3 and word in seg_lower:
                score += 1

    return score


def decompose_request(user_message, tools):
    """Split a multi-action request into individual sub-requests.

    Each sub-request gets matched to its most relevant tool(s).
    """
    msg_lower = user_message.lower()

    # Split on conjunctions and commas
    parts = re.split(r'\band\b|,', msg_lower)
    segments = [p.strip().strip('.') for p in parts if len(p.strip()) > 3]

    if not segments:
        segments = [msg_lower]

    sub_requests = []
    for seg in segments:
        scored = [(tool, _tool_relevance_score(seg, tool)) for tool in tools]
        scored.sort(key=lambda x: x[1], reverse=True)

        if scored and scored[0][1] > 0:
            sub_requests.append({
                "message": seg,
                "likely_tools": [scored[0][0]],
                "all_tools": tools,
            })

    if not sub_requests:
        return [{"message": user_message, "likely_tools": tools, "all_tools": tools}]

    return sub_requests


def merge_results(local_results, local_time_ms):
    """Merge locally-resolved function calls from sub-requests."""
    all_calls = []
    for lr in local_results:
        all_calls.extend(lr["function_calls"])

    return {
        "function_calls": all_calls,
        "total_time_ms": local_time_ms,
        "confidence": min(lr["confidence"] for lr in local_results) if local_results else 0,
        "source": "on-device",
    }


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    try:
        return _generate_cloud_inner(messages, tools)
    except Exception as e:
        return {
            "function_calls": [],
            "total_time_ms": 0,
        }


def _generate_cloud_inner(messages, tools):
    """Inner cloud call — separated for error handling."""
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

    # System instruction for concise, precise function calls
    system_instruction = (
        "You are a function-calling assistant. "
        "If the user requests multiple actions, call ALL relevant functions in a single response. "
        "Extract parameter values directly from the user's request. "
        "Do not add trailing punctuation to string values. "
        "For time parameters, use format like '2:00 PM' with colon and minutes."
    )

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=system_instruction,
            temperature=0.0,
        ),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                args = dict(part.function_call.args)
                # Post-process: strip trailing punctuation from string args
                for k, v in args.items():
                    if isinstance(v, str):
                        args[k] = v.strip().rstrip(".,!?;:")
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": args,
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """CellClaw hybrid routing — task-aware adaptive routing between on-device and cloud.

    Ported from zeroclaw's multi-model router architecture:
    - Hint-based routing → task complexity classification
    - ReliableProvider pattern → 4-gate result validation
    - Adaptive thresholds per difficulty level
    - Request decomposition for multi-tool scenarios
    """
    complexity = classify_task_complexity(messages, tools)

    # ── EASY PATH ──────────────────────────────────────────────
    # 1 tool, direct request. FunctionGemma excels here.
    # Single run + semantic check → maximize on-device ratio.
    if complexity == "easy":
        local = run_local_with_validation(
            messages, tools, conf_threshold=0.50, retries=0,
        )
        if local is not None:
            local["source"] = "on-device"
            return local

        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        return cloud

    # ── MEDIUM PATH ────────────────────────────────────────────
    # 2-5 tools: try local with validation. FunctionGemma sometimes
    # picks wrong tool, so validation catches most errors.
    if complexity == "medium":
        local = run_local_with_validation(
            messages, tools, conf_threshold=0.40, retries=0,
        )
        if local is not None:
            local["source"] = "on-device"
            return local

        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        return cloud

    # ── HARD PATH ──────────────────────────────────────────────
    # Multi-call: try decomposition locally first.
    # If any sub-request fails, fall back to cloud for entire request.
    user_msg = " ".join(m["content"] for m in messages if m["role"] == "user").strip()
    sub_requests = decompose_request(user_msg, tools)

    local_results = []
    any_failed = False
    local_time = 0

    for sub in sub_requests:
        sub_messages = [{"role": "user", "content": sub["message"]}]

        # Try with matched tool first
        result = run_local_with_validation(
            sub_messages, sub["likely_tools"], conf_threshold=0.60,
        )

        # Retry with all tools if single-tool attempt failed
        if result is None and sub["likely_tools"] != sub["all_tools"]:
            result = run_local_with_validation(
                sub_messages, sub["all_tools"], conf_threshold=0.60,
            )

        if result is not None:
            local_results.append(result)
            local_time += result["total_time_ms"]
        else:
            any_failed = True
            break  # Any failure → cloud for the whole request

    if not any_failed and local_results:
        return merge_results(local_results, local_time)

    # Cloud fallback for entire original request
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += local_time
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
