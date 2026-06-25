"""Model registry + prompting + output parsing for the rename benchmark.

Two protocols:
  * "json" (instruct models): show the whole masked file, ask for a JSON map
    placeholder -> identifier. One answer per identifier => site-consistent.
  * "fim"  (base models): native fill-in-the-middle at each identifier's first
    masked site. Consistency across sites is measured, not enforced.
"""
import json
import re

# ---------------------------------------------------------------------------
# Registry. Add your own entries freely; keys are what --models accepts.
# ---------------------------------------------------------------------------
MODELS = {
    # DeepSeek-Coder ---------------------------------------------------------
    "deepseek-1.3b-instruct": {
        "id": "deepseek-ai/deepseek-coder-1.3b-instruct", "protocol": "json"},
    "deepseek-6.7b-instruct": {
        "id": "deepseek-ai/deepseek-coder-6.7b-instruct", "protocol": "json"},
    "deepseek-1.3b-base": {
        "id": "deepseek-ai/deepseek-coder-1.3b-base", "protocol": "fim",
        "fim": ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>")},
    "deepseek-6.7b-base": {
        "id": "deepseek-ai/deepseek-coder-6.7b-base", "protocol": "fim",
        "fim": ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>")},
    # Qwen2.5-Coder ----------------------------------------------------------
    "qwen2.5-0.5b-instruct": {
        "id": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "protocol": "json"},
    "qwen2.5-1.5b-instruct": {
        "id": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "protocol": "json"},
    "qwen2.5-3b-instruct": {
        "id": "Qwen/Qwen2.5-Coder-3B-Instruct", "protocol": "json"},
    "qwen2.5-7b-instruct": {
        "id": "Qwen/Qwen2.5-Coder-7B-Instruct", "protocol": "json"},
    "qwen2.5-1.5b-base": {
        "id": "Qwen/Qwen2.5-Coder-1.5B", "protocol": "fim",
        "fim": ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")},
    "qwen2.5-7b-base": {
        "id": "Qwen/Qwen2.5-Coder-7B", "protocol": "fim",
        "fim": ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")},
}

IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
PLACEHOLDER_RE = re.compile(r"IDENT_\d+")


def list_models():
    return sorted(MODELS)


def clean_identifier(text):
    """Pull the first plausible Java identifier out of a noisy model string."""
    if not text:
        return ""
    text = text.strip().strip("`\"' \t")
    text = text.split("\n")[0]
    m = IDENT_RE.search(text)
    return m.group(0) if m else ""


def build_json_messages(record):
    placeholders = sorted(record["ground_truth"].keys(),
                          key=lambda p: int(p.split("_")[1]))
    example = "{" + ", ".join(f'"{p}": "..."' for p in placeholders) + "}"
    system = ("You are an expert Java developer performing a rename refactoring. "
              "You infer descriptive, idiomatic identifier names from code context.")
    user = (
        f"In the Java code below, {len(placeholders)} identifiers were replaced by "
        f"the placeholders {', '.join(placeholders)}. Each placeholder consistently "
        "replaces *all* occurrences of one original identifier (a local variable or "
        "parameter). Recover the single most likely original name for each placeholder.\n\n"
        "Respond with ONLY a JSON object on one line, mapping each placeholder to "
        f"your predicted identifier, e.g. {example}\n\n"
        "```java\n" + record["masked_code"] + "\n```"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def parse_json_prediction(text, placeholders):
    """Robustly extract {placeholder: identifier} from a model response."""
    out = {}
    # 1) try a real JSON object slice
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            for ph in placeholders:
                                if ph in obj:
                                    out[ph] = clean_identifier(str(obj[ph]))
                    except Exception:
                        pass
                    break
    # 2) fall back to per-placeholder regex for anything still missing
    for ph in placeholders:
        if out.get(ph):
            continue
        m = re.search(re.escape(ph) + r'"?\s*[:=]\s*"?([A-Za-z_$][A-Za-z0-9_$]*)', text)
        if m:
            out[ph] = m.group(1)
    return out


class ModelRunner:
    def __init__(self, key, max_new_tokens=160, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if key not in MODELS:
            raise KeyError(f"unknown model '{key}'. Known: {list_models()}")
        self.key = key
        self.cfg = MODELS[key]
        self.protocol = self.cfg["protocol"]
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {key} ({self.cfg['id']}) ...")
        self.tok = AutoTokenizer.from_pretrained(self.cfg["id"], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["id"],
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    # -- generation helpers --------------------------------------------------
    def _generate(self, prompt, max_new_tokens):
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True)

    # -- protocols -----------------------------------------------------------
    def _predict_json(self, record):
        placeholders = sorted(record["ground_truth"].keys(),
                              key=lambda p: int(p.split("_")[1]))
        messages = build_json_messages(record)
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        raw = self._generate(prompt, self.max_new_tokens)
        pred = parse_json_prediction(raw, placeholders)
        return pred, raw

    def _predict_fim(self, record):
        begin, hole, end = self.cfg["fim"]
        code = record["masked_code"]
        placeholders = sorted(record["ground_truth"].keys(),
                              key=lambda p: int(p.split("_")[1]))
        pred, raws = {}, []
        for ph in placeholders:
            idx = code.find(ph)
            if idx == -1:
                pred[ph] = ""
                continue
            prefix = code[:idx]
            suffix = code[idx + len(ph):]
            prompt = f"{begin}{prefix}{hole}{suffix}{end}"
            raw = self._generate(prompt, 16)
            raws.append(f"{ph}={raw!r}")
            pred[ph] = clean_identifier(raw)
        return pred, " | ".join(raws)

    def predict(self, record):
        if self.protocol == "json":
            return self._predict_json(record)
        return self._predict_fim(record)

    def close(self):
        import gc
        del self.model
        del self.tok
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
