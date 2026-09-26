"""Model detector adapters. Each factory returns a callable text -> list of spans (dict start,end,label,src,score)."""
import os, re
from functools import lru_cache

MODELS = os.environ.get("PII_MODELS", "./models")


def _S(s, e, label, src, score):
    """Build a span dict, coercing offsets to int and score to float.

    Example:
      s        = 0
      e        = 10
      label    = "NAME"
      src      = "gliner:urchade"
      score    = 0.91234
      returns  = {"start": 0, "end": 10, "label": "NAME", "src": "gliner:urchade", "score": 0.91234}
    """
    return {"start": int(s), "end": int(e), "label": label, "src": src, "score": float(score)}

# ------------------------------------------------------------ spaCy
SPACY_MAP = {"PERSON": "NAME", "PER": "NAME", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "ADDRESS", "ORG": "ORG",
             "NORP": "NATIONALITY", "DATE": "DATE"}


@lru_cache(None)
def _spacy(name):
    """Load (and cache) a spaCy pipeline by package name.

    Example (illustrative, not executed):
      name     = "en_core_web_lg"
      returns  = <spacy.lang.en.English object>   # same object on repeat calls (lru_cache)
    """
    import spacy
    return spacy.load(name)


def spacy_detector(model="en_core_web_lg"):
    """Return a callable text -> spans using spaCy NER, labels mapped via SPACY_MAP, fixed score 0.85.

    Example (illustrative, not executed):
      model    = "en_core_web_lg"
      fn       = spacy_detector(model)
      text     = "John Smith, john@x.com, Leeds"
      returns  = fn(text)
               = [{"start": 0, "end": 10, "label": "NAME", "src": "spacy:en_core_web_lg", "score": 0.85},
                  {"start": 24, "end": 29, "label": "LOCATION", "src": "spacy:en_core_web_lg", "score": 0.85}]
                                                    # email not an NER entity -> no span
    """
    nlp = _spacy(model)
    def fn(text):
        doc = nlp(text)
        return [_S(e.start_char, e.end_char, SPACY_MAP[e.label_], f"spacy:{model}", 0.85)
                for e in doc.ents if e.label_ in SPACY_MAP]
    return fn

# ------------------------------------------------------------ Presidio
PRESIDIO_MAP = {"PERSON": "NAME", "LOCATION": "LOCATION", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "URL": "URL",
                "DATE_TIME": "DATE", "NRP": "NATIONALITY", "IBAN_CODE": "ID", "CREDIT_CARD": "ID", "UK_NHS": "ID", "US_SSN": "ID",
                "US_PASSPORT": "ID", "US_DRIVER_LICENSE": "ID", "US_BANK_NUMBER": "ID", "IP_ADDRESS": "ID", "MEDICAL_LICENSE": "ID",
                "UK_NINO": "ID", "ORGANIZATION": "ORG", "ES_NIF": "ID", "IT_FISCAL_CODE": "ID", "FI_PERSONAL_IDENTITY_CODE": "ID"}


@lru_cache(None)
def _presidio(model):
    """Build (and cache) an English Presidio AnalyzerEngine backed by the given spaCy model.

    Example (illustrative, not executed):
      model    = "en_core_web_lg"
      returns  = <presidio_analyzer.AnalyzerEngine object>   # supported_languages=["en"]
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    prov = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model}]})
    return AnalyzerEngine(nlp_engine=prov.create_engine(), supported_languages=["en"])


def presidio_detector(model="en_core_web_lg", threshold=0.0):
    """Return a callable text -> spans using Presidio, entity types mapped via PRESIDIO_MAP (else "OTHER").

    Example (illustrative, not executed):
      model     = "en_core_web_lg"
      threshold = 0.0
      fn        = presidio_detector(model, threshold)
      text      = "John Smith, john@x.com, Leeds"
      returns   = fn(text)
                = [{"start": 12, "end": 22, "label": "EMAIL", "src": "presidio:en_core_web_lg", "score": 1.0},
                   {"start": 0, "end": 10, "label": "NAME", "src": "presidio:en_core_web_lg", "score": 0.85},
                   {"start": 24, "end": 29, "label": "LOCATION", "src": "presidio:en_core_web_lg", "score": 0.85},
                   {"start": 17, "end": 22, "label": "URL", ...}]   # "x.com" inside the email also hits URL
    """
    eng = _presidio(model)
    def fn(text):
        res = eng.analyze(text=text, language="en", score_threshold=threshold)
        return [_S(r.start, r.end, PRESIDIO_MAP.get(r.entity_type, "OTHER"), f"presidio:{model}", r.score) for r in res]
    return fn

# ------------------------------------------------------------ GLiNER (v1 family: knowledgator, nvidia, urchade)
GLINER_LABELS = {
 "person name": "NAME", "email address": "EMAIL", "phone number": "PHONE", "street address": "ADDRESS",
 "city": "LOCATION", "country": "LOCATION", "university": "EDU_ORG", "organization": "ORG",
 "date of birth": "DOB", "nationality": "NATIONALITY", "national id number": "ID", "passport number": "ID",
 "bank account number": "ID", "iban": "ID", "tax id": "ID", "url": "URL", "username": "HANDLE", "marital status": "MARITAL",
}


@lru_cache(None)
def _gliner(path):
    """Load (and cache) a GLiNER v1 model from a local directory (no network).

    Example (illustrative, not executed):
      path     = "./models/urchade/gliner_multi_pii-v1"
      returns  = <gliner.GLiNER object>   # raises if not present locally (local_files_only=True)
    """
    from gliner import GLiNER
    return GLiNER.from_pretrained(path, local_files_only=True)


def _chunks(text, max_chars=1200):
    """Line-aligned chunks with offsets (GLiNER v1 has a ~384-token window).

    Example:
      text      = "John Smith\\njohn@x.com\\nLeeds, UK\\n"
      max_chars = 20                        # each line fits, two lines don't -> one chunk per line
      returns   = [(0, "John Smith\\n"), (11, "john@x.com\\n"), (22, "Leeds, UK\\n")]
      # with default max_chars=1200 -> [(0, "John Smith\\njohn@x.com\\nLeeds, UK\\n")]
    """
    out, start, buf = [], 0, ""
    pos = 0
    for ln in text.splitlines(keepends=True):
        if len(buf) + len(ln) > max_chars and buf:
            out.append((start, buf)); start = pos; buf = ""
        buf += ln; pos += len(ln)
    if buf: out.append((start, buf))
    return out


def gliner_detector(repo, threshold=0.3, labels=None):
    """Return a callable text -> spans running a local GLiNER v1 model chunk-by-chunk with GLINER_LABELS.

    Example (illustrative, not executed):
      repo      = "urchade/gliner_multi_pii-v1"   # loaded from $PII_MODELS/<repo>
      threshold = 0.3
      labels    = None                            # -> GLINER_LABELS
      fn        = gliner_detector(repo, threshold, labels)
      text      = "John Smith, john@x.com, Leeds"
      returns   = fn(text)
                = [{"start": 0, "end": 10, "label": "NAME", "src": "gliner:urchade", "score": 0.97},
                   {"start": 12, "end": 22, "label": "EMAIL", "src": "gliner:urchade", "score": 0.99},
                   {"start": 24, "end": 29, "label": "LOCATION", "src": "gliner:urchade", "score": 0.88}]
    """
    model = _gliner(os.path.join(MODELS, repo))
    labels = labels or GLINER_LABELS
    def fn(text):
        out = []
        for off, chunk in _chunks(text):
            for e in model.predict_entities(chunk, list(labels), threshold=threshold, flat_ner=True):
                out.append(_S(off + e["start"], off + e["end"], labels[e["label"]], f"gliner:{repo.split('/')[0]}", e["score"]))
        return out
    return fn

# ------------------------------------------------------------ GLiNER2-PII
GLINER2_LABELS = {
 "person": "NAME", "full_name": "NAME", "first_name": "NAME", "last_name": "NAME", "email": "EMAIL", "phone_number": "PHONE",
 "street_address": "ADDRESS", "address": "ADDRESS", "city": "LOCATION", "state_or_region": "LOCATION", "postal_code": "ADDRESS",
 "country": "LOCATION", "date_of_birth": "DOB", "national_id_number": "ID", "passport_number": "ID", "tax_id": "ID",
 "iban": "ID", "bank_account": "ID", "account_number": "ID", "license_number": "ID", "username": "HANDLE", "government_id": "ID",
}


GLINER2_CV_LABELS = dict(GLINER2_LABELS, **{"university": "EDU_ORG", "school": "EDU_ORG", "nationality": "NATIONALITY",
                                              "marital_status": "MARITAL", "personal_website": "URL"})


@lru_cache(None)
def _gliner2(path):
    """Load (and cache) a GLiNER2 model from a local directory.

    Example (illustrative, not executed):
      path     = "./models/fastino/gliner2-privacy-filter-PII-multi"
      returns  = <gliner2.GLiNER2 object>
    """
    from gliner2 import GLiNER2
    return GLiNER2.from_pretrained(path)


def gliner2_detector(repo="fastino/gliner2-privacy-filter-PII-multi", threshold=0.5, labels=None):
    """Return a callable text -> spans running GLiNER2 entity extraction over 1500-char chunks.

    Example (illustrative, not executed):
      repo      = "fastino/gliner2-privacy-filter-PII-multi"
      threshold = 0.5
      labels    = None                            # -> GLINER2_LABELS (unknown labels -> "OTHER")
      fn        = gliner2_detector(repo, threshold, labels)
      text      = "John Smith, john@x.com, Leeds"
      returns   = fn(text)
                = [{"start": 0, "end": 10, "label": "NAME", "src": "gliner2", "score": 0.96},
                   {"start": 12, "end": 22, "label": "EMAIL", "src": "gliner2", "score": 0.99},
                   {"start": 24, "end": 29, "label": "LOCATION", "src": "gliner2", "score": 0.91}]
    """
    model = _gliner2(os.path.join(MODELS, repo))
    labels = labels or GLINER2_LABELS
    def fn(text):
        out = []
        for off, chunk in _chunks(text, 1500):
            res = model.extract_entities(chunk, list(labels), threshold=threshold, include_confidence=True, include_spans=True)
            ents = res.get("entities", res)
            for lab, items in ents.items():
                for it in items:
                    if isinstance(it, dict) and "start" in it:
                        out.append(_S(off + it["start"], off + it["end"], labels.get(lab, "OTHER"), "gliner2", it.get("confidence", 1.0)))
        return out
    return fn

# ------------------------------------------------------------ OpenAI Privacy Filter (and OpenMed multilingual fine-tune)
PF_MAP = {"private_person": "NAME", "private_address": "ADDRESS", "private_email": "EMAIL", "private_phone": "PHONE",
          "private_url": "URL", "private_date": "DATE", "account_number": "ID", "secret": "ID"}


@lru_cache(None)
def _pf(path):
    """Build (and cache) a CPU HF token-classification pipeline with simple aggregation.

    Example (illustrative, not executed):
      path     = "./models/openai/privacy-filter"
      returns  = <transformers.TokenClassificationPipeline object>   # device=-1 (CPU)
    """
    from transformers import pipeline
    return pipeline("token-classification", model=path, aggregation_strategy="simple", device=-1, trust_remote_code=True)


def privacy_filter_detector(repo="openai/privacy-filter", threshold=0.5):
    """Return a callable text -> spans from the HF pipeline, labels mapped via PF_MAP then _openmed_map.

    Example (illustrative, not executed):
      repo      = "openai/privacy-filter"
      threshold = 0.5
      fn        = privacy_filter_detector(repo, threshold)
      text      = "John Smith, john@x.com, Leeds"
      returns   = fn(text)
                = [{"start": 0, "end": 10, "label": "NAME", "src": "pf:openai", "score": 0.98},    # private_person
                   {"start": 12, "end": 22, "label": "EMAIL", "src": "pf:openai", "score": 0.99}]  # private_email
                                                    # bare city "Leeds" below threshold -> dropped
    """
    pipe = _pf(os.path.join(MODELS, repo))
    def fn(text):
        out = []
        for e in pipe(text):
            lab = e["entity_group"].lower()
            lab = re.sub(r"^[bieos]-", "", lab)
            mapped = PF_MAP.get(lab) or _openmed_map(lab)
            if e["score"] >= threshold:
                out.append(_S(e["start"], e["end"], mapped, f"pf:{repo.split('/')[0]}", e["score"]))
        return out
    return fn


OPENMED_MAP = {"organization": "ORG", "accountname": "HANDLE", "currencyname": "OTHER", "age": "DOB", "bitcoinaddress": "OTHER",
               "ethereumaddress": "OTHER", "litecoinaddress": "OTHER", "ipaddress": "ID", "macaddress": "OTHER", "useragent": "OTHER",
               "creditcardissuer": "OTHER", "jobtitle": "OTHER", "jobdepartment": "OTHER", "occupation": "OTHER", "prefix": "OTHER"}


def _openmed_map(lab):
    """Map an OpenMed/free-form PII label to our label set: exact OPENMED_MAP hit, else first substring rule, else "OTHER".

    Example:
      lab      = "CITY"            # lower-cased first
      returns  = "LOCATION"
      lab      = "firstname"       # substring "name"
      returns  = "NAME"
      lab      = "zipcode"         # substring "zip"
      returns  = "ADDRESS"
      lab      = "jobtitle"        # exact OPENMED_MAP hit
      returns  = "OTHER"
    """
    l = lab.lower()
    if l in OPENMED_MAP: return OPENMED_MAP[l]
    for k, v in [("email", "EMAIL"), ("phone", "PHONE"), ("url", "URL"), ("user", "HANDLE"), ("birth", "DOB"), ("dob", "DOB"),
                 ("street", "ADDRESS"), ("address", "ADDRESS"), ("zip", "ADDRESS"), ("postcode", "ADDRESS"), ("building", "ADDRESS"),
                 ("city", "LOCATION"), ("state", "LOCATION"), ("country", "LOCATION"), ("county", "LOCATION"),
                 ("name", "NAME"), ("person", "NAME"), ("surname", "NAME"), ("title", "OTHER"),
                 ("iban", "ID"), ("account", "ID"), ("card", "ID"), ("passport", "ID"), ("ssn", "ID"), ("id", "ID"), ("tax", "ID"),
                 ("license", "ID"), ("number", "ID"), ("date", "DATE"), ("age", "AGE"), ("gender", "GENDER"), ("sex", "GENDER")]:
        if k in l: return v
    return "OTHER"

# ------------------------------------------------------------ language routing for spaCy
_LANG_MODEL = {"ENGLISH": "en_core_web_lg", "GERMAN": "de_core_news_lg", "FRENCH": "fr_core_news_lg", "SPANISH": "es_core_news_lg",
               "DUTCH": "nl_core_news_lg", "SWEDISH": "sv_core_news_lg", "FINNISH": "fi_core_news_lg"}


@lru_cache(None)
def _lid():
    """Build (and cache) a lingua language detector restricted to the languages in _LANG_MODEL.

    Example (illustrative, not executed):
      returns  = <lingua.LanguageDetector object>   # candidates: ENGLISH, GERMAN, FRENCH, SPANISH, DUTCH, SWEDISH, FINNISH
    """
    from lingua import LanguageDetectorBuilder, Language
    return LanguageDetectorBuilder.from_languages(*[getattr(Language, k) for k in _LANG_MODEL]).build()


def detect_lang(text):
    """Return the detected language name (a key of _LANG_MODEL), defaulting to "ENGLISH".

    Example (illustrative, not executed):
      text     = "Berufserfahrung: Softwareentwickler bei Siemens AG, München"
      returns  = "GERMAN"
      text     = ""                # lingua returns None -> fallback
      returns  = "ENGLISH"
    """
    l = _lid().detect_language_of(text)
    return l.name if l else "ENGLISH"


def spacy_auto_detector():
    """Return a callable text -> spans that picks the spaCy model per text via detect_lang.

    Example (illustrative, not executed):
      fn       = spacy_auto_detector()
      text     = "Hans Müller, hans@x.de, München"   # detected GERMAN -> de_core_news_lg
      returns  = fn(text)
               = [{"start": 0, "end": 11, "label": "NAME", "src": "spacy:de_core_news_lg", "score": 0.85},
                  {"start": 24, "end": 31, "label": "LOCATION", "src": "spacy:de_core_news_lg", "score": 0.85}]
    """
    def fn(text):
        return spacy_detector(_LANG_MODEL[detect_lang(text)])(text)
    return fn


# ------------------------------------------------------------ Privacy Filter with our own BIOES decoding
@lru_cache(None)
def _pf_model(path):
    """Load (and cache) the tokenizer and bfloat16 token-classification model in eval mode.

    Example (illustrative, not executed):
      path     = "./models/openai/privacy-filter"
      returns  = (<transformers tokenizer>, <AutoModelForTokenClassification, eval, bfloat16>)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path, dtype=torch.bfloat16)
    model.eval()
    return tok, model


def privacy_filter_tokens_detector(repo="openai/privacy-filter", min_prob=0.05):
    """Span score = min over its tokens of P(not O). Threshold sweeps filter on that score.

    Example (illustrative, not executed):
      repo     = "openai/privacy-filter"
      min_prob = 0.05
      fn       = privacy_filter_tokens_detector(repo, min_prob)
      text     = "John Smith, john@x.com, Leeds"
      returns  = fn(text)
               = [{"start": 0, "end": 10, "label": "NAME", "src": "pf:openai", "score": 0.97},     # "John"+" Smith" merged, min score
                  {"start": 12, "end": 22, "label": "EMAIL", "src": "pf:openai", "score": 0.99},
                  {"start": 24, "end": 29, "label": "ADDRESS", "src": "pf:openai", "score": 0.12}]  # low P(not O) kept, filtered later
    """
    import torch
    tok, model = _pf_model(os.path.join(MODELS, repo))
    id2 = model.config.id2label
    def fn(text):
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=8192)
        offs = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = model(**enc).logits[0].float()
        probs = logits.softmax(-1)
        o_idx = [i for i, l in id2.items() if l == "O"][0]
        spans, cur = [], None
        for ti, (a, b) in enumerate(offs):
            if a == b:
                continue
            p_pii = 1.0 - probs[ti, int(o_idx)].item()
            if p_pii >= min_prob:
                pr = probs[ti].clone(); pr[int(o_idx)] = -1
                lab = re.sub(r"^[BIES]-", "", id2[int(pr.argmax())])
                mapped = PF_MAP.get(lab) or _openmed_map(lab)
                if cur and cur["label"] == mapped and a - cur["end"] <= 1:
                    cur["end"] = b; cur["score"] = min(cur["score"], p_pii)
                    continue
                if cur: spans.append(cur)
                cur = _S(a, b, mapped, f"pf:{repo.split('/')[0]}", p_pii)
            else:
                if cur: spans.append(cur); cur = None
        if cur: spans.append(cur)
        for sp in spans:  # trim leading whitespace from BPE offsets
            while sp["start"] < sp["end"] and text[sp["start"]].isspace(): sp["start"] += 1
        return spans
    return fn


GLINER2_FT_LABELS = {"person": "NAME", "email": "EMAIL", "phone_number": "PHONE", "street_address": "ADDRESS", "city": "LOCATION",
                     "personal_website": "URL", "username": "HANDLE", "national_id_number": "ID", "date_of_birth": "DOB",
                     "nationality": "NATIONALITY", "marital_status": "MARITAL", "university": "EDU_ORG", "gender": "GENDER"}

KNOWLEDGATOR_LABELS = {"name": "NAME", "email address": "EMAIL", "phone number": "PHONE", "location address": "ADDRESS",
    "location street": "ADDRESS", "location city": "LOCATION", "location country": "LOCATION", "location zip": "ADDRESS",
    "url": "URL", "username": "HANDLE", "dob": "DOB", "age": "DOB", "gender": "GENDER", "marital status": "MARITAL",
    "account number": "ID", "bank account": "ID", "ssn": "ID", "passport number": "ID", "driver license": "ID",
    "university": "EDU_ORG", "nationality": "NATIONALITY"}
