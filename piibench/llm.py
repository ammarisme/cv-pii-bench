"""LLM stages: (1) span extractor (LLM-only stack) and (2) adjudicator over detector candidates.
The LLM never rewrites the CV: it returns verbatim strings, which code maps back to offsets."""
import json, os, re, hashlib
from functools import lru_cache
from pathlib import Path
from .pipeline import S, section_at

CACHE = Path(__file__).resolve().parent.parent / "detector_outputs" / "llm"; CACHE.mkdir(parents=True, exist_ok=True)
MODELS = os.environ.get("PII_MODELS", "./models")

POLICY = """Masking policy for CVs (GDPR-style anonymisation before recruiter review):
MASK: personal names of the candidate AND of any other person (referees, supervisors, co-authors, family),
emails, phone numbers, street addresses and postcodes, the candidate's home city/region/country, personal URLs
(LinkedIn, GitHub profile, personal website), social handles, ID/account/licence/registration numbers, date of birth or age,
nationality/citizenship, marital status or children, gender/pronouns, and names of universities, colleges and schools.
KEEP VISIBLE: employer and client company names (even if they contain a person or place name), job titles, skills, tools,
programming languages and products (e.g. Jenkins, Julia, Kafka, Phoenix, Django, Ruby), certification names and issuers,
cities mentioned only as work/project locations, employment/education date ranges, metrics, journal/conference names,
company or documentation URLs."""

LABELS = ["NAME", "EMAIL", "PHONE", "ADDRESS", "LOCATION", "URL", "HANDLE", "ID", "DOB", "NATIONALITY", "MARITAL", "GENDER", "EDU_ORG"]


class Backend:
    def __init__(self, kind, model):
        """Hold the backend kind and model; the cache key uses only the model's basename.

        Example:
          kind     = "llama"
          model    = "/opt/models/qwen2.5-3b-instruct-q4_k_m.gguf"
          returns  = Backend with key = "llama:qwen2.5-3b-instruct-q4_k_m.gguf"   # install path dropped
        """
        self.kind, self.model = kind, model
        self.key = f"{kind}:{os.path.basename(model)}"  # cache key independent of install path

    @lru_cache(None)
    def _llama(self):
        """Load the local GGUF model with llama.cpp (once per Backend, via lru_cache).

        Example (illustrative, not executed):
          self     = Backend("llama", "./models/qwen2.5-3b-instruct-q4_k_m.gguf")
          returns  = Llama(model_path=..., n_ctx=8192, seed=0)   # loads the model into memory
        """
        from llama_cpp import Llama
        return Llama(model_path=self.model, n_ctx=8192, n_threads=os.cpu_count(), verbose=False, seed=0)

    def chat(self, system, user, max_tokens=1500):
        """Return the model's JSON-mode reply, reading/writing the on-disk cache keyed by md5 of the request.

        Example (illustrative, not executed):
          self       = Backend("openai", "gpt-4o-mini")
          system     = EXTRACT_SYS
          user       = "CV:\\n<<<\\nAna Ruiz\\nLeeds, UK\\n>>>"
          max_tokens = 1500
          returns    = '{"pii": [{"text": "Ana Ruiz", "label": "NAME"}, ...]}'   # cache hit -> no API call
        """
        h = hashlib.md5(json.dumps([self.key, system, user, max_tokens]).encode()).hexdigest()
        f = CACHE / f"{h}.json"
        if f.exists():
            return json.loads(f.read_text())["out"]
        if self.kind == "llama":
            r = self._llama().create_chat_completion(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                                     temperature=0, max_tokens=max_tokens, response_format={"type": "json_object"})
            out = r["choices"][0]["message"]["content"]
        elif self.kind == "openai":
            import requests
            r = requests.post("https://api.openai.com/v1/chat/completions", timeout=120,
                              headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                              json={"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status(); out = r.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError(self.kind)
        f.write_text(json.dumps({"out": out}))
        return out


def _parse(out):
    """Parse an LLM reply as JSON, falling back to the first {...} block, else {}.

    Example:
      out      = 'Sure! {"pii": [{"text": "Ana"}]} done'
      returns  = {"pii": [{"text": "Ana"}]}           # surrounding chatter stripped
      out      = "no json"
      returns  = {}
    """
    try:
        return json.loads(out)
    except Exception:
        m = re.search(r"\{.*\}", out, re.S)
        try:
            return json.loads(m.group()) if m else {}
        except Exception:
            return {}


def _locate(text, s):
    """All (start, end) offsets of the stripped string s in text; [] if s is shorter than 2 chars.

    Example:
      text     = "Ana Ruiz, ana@x.es. Ana"
      s        = " Ana "                              # stripped -> "Ana"; case-sensitive, so "ana@" not matched
      returns  = [(0, 3), (20, 23)]
    """
    s = s.strip()
    if len(s) < 2: return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(s), text)]


EXTRACT_SYS = POLICY + """
Task: list every string in the CV that must be MASKED. Copy each string EXACTLY as it appears (same spelling, case,
spacing). One entry per distinct string. Answer only JSON: {"pii": [{"text": "...", "label": "NAME|EMAIL|PHONE|ADDRESS|LOCATION|URL|HANDLE|ID|DOB|NATIONALITY|MARITAL|GENDER|EDU_ORG"}]}"""


def llm_extract_detector(backend):
    """Build a detector fn(text) -> spans that asks the LLM to list PII strings and maps them back to offsets.

    Example (illustrative, not executed):
      backend  = Backend("openai", "gpt-4o-mini")
      returns  = fn                                  # see fn below
    """
    def fn(text):
        """Extract PII spans from text via backend.chat(EXTRACT_SYS, ...).

        Example (illustrative, not executed; LLM reply '{"pii": [{"text": "Ana Ruiz", "label": "name"}, {"text": "Leeds", "label": "CITY"}]}'):
          text     = "Ana Ruiz\\nLeeds, UK"
          returns  = [S(0, 8, "NAME", "llm:gpt-4o-mini", 0.9),
                      S(9, 14, "OTHER", "llm:gpt-4o-mini", 0.9)]   # "CITY" not in LABELS -> OTHER
        """
        out = _parse(backend.chat(EXTRACT_SYS, "CV:\n<<<\n" + text + "\n>>>"))
        spans = []
        for it in out.get("pii", []) if isinstance(out, dict) else []:
            if not isinstance(it, dict): continue
            lab = str(it.get("label", "OTHER")).upper()
            for a, b in _locate(text, str(it.get("text", ""))):
                spans.append(S(a, b, lab if lab in LABELS else "OTHER", f"llm:{backend.model.split('/')[-1]}", 0.9))
        return spans
    return fn


ADJ_SYS = POLICY + """
You review candidate detections made by automatic detectors on a CV. For each numbered candidate decide
"mask" or "keep_visible" according to the policy, using the surrounding context. Then list any PII the detectors MISSED,
copying the exact text. Answer only JSON:
{"decisions": {"<id>": "mask"|"keep_visible", ...}, "missed": [{"text": "...", "label": "..."}]}"""

TRUSTED = ("re", "re-kv", "net")  # high-precision regex sources bypass adjudication


def make_adjudicator(backend, review_labels=("NAME", "LOCATION", "ADDRESS", "ORG", "EDU_ORG", "NATIONALITY", "DATE", "OTHER", "URL"), add_missed=True, mode="full", drop_labels=None):
    """Build an adjudicator adj(text, cand, sections, cfg) where the LLM keeps/drops candidates and adds missed PII.

    Example (illustrative, not executed):
      backend  = Backend("openai", "gpt-4o-mini")
      mode     = "add_only"                           # LLM may add spans but never drop candidates
      returns  = adj                                  # see adj below
    """
    def adj(text, cand, sections, cfg):
        """Keep trusted/unreviewed spans, apply the LLM's mask/keep_visible decisions, add its missed strings.

        Example (illustrative, not executed; mode="full"; LLM decides 0 mask, 1 and 2 keep_visible, missed "Ruiz"):
          text     = "Ana Ruiz\\nana@x.es\\n\\nEXPERIENCE\\nEngineer at Siemens, Berlin"
          sections = section_of_lines(text)
          cand     = [S(0, 8, "NAME", "gliner"), S(9, 17, "EMAIL", "re"),   # "re" is TRUSTED -> not sent to LLM
                      S(42, 49, "ORG", "gliner"), S(51, 57, "LOCATION", "gliner")]
          returns  = [S(9, 17, "EMAIL", "re"), S(0, 8, "NAME", "gliner+adj"),
                      S(4, 8, "NAME", "re-llm-add", 0.9)]   # "Siemens"/"Berlin" dropped as keep_visible
        """
        fixed, review = [], []
        for sp in cand:
            if sp["src"] in TRUSTED or sp["label"] not in review_labels:
                fixed.append(sp)
            else:
                review.append(sp)
        # dedupe review candidates by surface string
        groups = {}
        for sp in review:
            groups.setdefault((text[sp["start"]:sp["end"]].strip(), sp["label"]), []).append(sp)
        items = list(groups.items())
        lines = []
        for i, ((t, lab), sps) in enumerate(items):
            _, ln, _ = section_at(sections, sps[0]["start"])
            lines.append(f'{i}. "{t}" (detector label {lab}; line: "{ln.strip()[:140]}")')
        user = "CV:\n<<<\n" + text + "\n>>>\n\nCandidates:\n" + ("\n".join(lines) if lines else "(none)")
        out = _parse(backend.chat(ADJ_SYS, user, max_tokens=2000))
        dec = out.get("decisions", {}) if isinstance(out, dict) else {}
        kept = list(fixed)
        for i, ((t, lab), sps) in enumerate(items):
            d = str(dec.get(str(i), dec.get(i, "mask"))).lower()
            may_drop = mode in ("full", "drop_only") and (drop_labels is None or lab in drop_labels)
            if "keep" not in d or not may_drop:
                for sp in sps: kept.append(dict(sp, src=sp["src"] + "+adj"))
        if add_missed and mode in ("full", "add_only") and isinstance(out, dict):
            for it in out.get("missed", []) or []:
                if not isinstance(it, dict): continue
                lab = str(it.get("label", "OTHER")).upper()
                for a, b in _locate(text, str(it.get("text", ""))):
                    kept.append(S(a, b, lab if lab in LABELS else "OTHER", "re-llm-add", 0.9))
        return kept
    return adj
