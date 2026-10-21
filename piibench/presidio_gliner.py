"""Presidio as the host: GLiNER (v1 and GLiNER2) recognizers inside a Presidio AnalyzerEngine.

Every configuration is a dict of switches (PG_CONFIGS). A config builds one AnalyzerEngine with
  - an NLP engine (spaCy model, or Presidio's no-op engine when no NLP artefacts are needed),
  - Presidio's predefined pattern recognizers (all, CV-relevant subset, or none), phone regions,
  - GLiNERRecognizer (gliner v1 models) and/or a GLiNER2 recognizer written against Presidio's LocalRecognizer API,
  - a text chunker (Presidio's CharacterBasedTextChunker, or LineChunker below),
  - an optional allow_list passed to analyze(), and optional context words for Presidio's LemmaContextAwareEnhancer.
The detector returns Presidio's final results (after its own duplicate removal) mapped to the bench's labels.
Starting point: https://presidio.dataprivacystack.org/samples/python/gliner/ (config "doc-urchade").
"""
import os
from functools import lru_cache
from .detectors import MODELS, _S, PRESIDIO_MAP
from .lexicons import TECH, LANGUAGE_NAMES, FIELD_LABELS, KV_KEYS
try:
    from presidio_analyzer.chunkers import BaseTextChunker
    from presidio_analyzer.chunkers.base_chunker import TextChunk
except ImportError:  # presidio only needed for live runs; configs stay importable
    BaseTextChunker = object

# Presidio entity type -> bench label. Presidio accepts any entity string, so CV-specific types get their own names.
PG_MAP = dict(PRESIDIO_MAP, **{"STREET_ADDRESS": "ADDRESS", "POSTAL_CODE": "ADDRESS", "EDUCATION_ORG": "EDU_ORG",
                               "DATE_OF_BIRTH": "DOB", "AGE": "DOB", "MARITAL_STATUS": "MARITAL", "GENDER": "GENDER",
                               "USERNAME": "HANDLE", "ID_NUMBER": "ID", "NATIONALITY": "NATIONALITY"})

# GLiNER label -> Presidio entity type
DOC_LABELS = {"person": "PERSON", "name": "PERSON", "organization": "ORGANIZATION", "location": "LOCATION"}  # as in the Presidio sample
PII_LABELS = {"person": "PERSON", "email": "EMAIL_ADDRESS", "phone number": "PHONE_NUMBER", "address": "STREET_ADDRESS",
              "location": "LOCATION", "date of birth": "DATE_OF_BIRTH", "passport number": "ID_NUMBER", "social security number": "ID_NUMBER",
              "bank account number": "ID_NUMBER", "username": "USERNAME", "organization": "ORGANIZATION"}   # generic PII list
CV_LABELS = {"person name": "PERSON", "email address": "EMAIL_ADDRESS", "phone number": "PHONE_NUMBER",
             "street address": "STREET_ADDRESS", "postal code": "POSTAL_CODE", "city": "LOCATION", "country": "LOCATION",
             "university": "EDUCATION_ORG", "school": "EDUCATION_ORG", "date of birth": "DATE_OF_BIRTH", "age": "AGE",
             "nationality": "NATIONALITY", "marital status": "MARITAL_STATUS", "gender": "GENDER",
             "personal website": "URL", "username": "USERNAME", "national id number": "ID_NUMBER", "passport number": "ID_NUMBER",
             "bank account number": "ID_NUMBER", "driver license number": "ID_NUMBER"}
KN_LABELS = {"name": "PERSON", "email address": "EMAIL_ADDRESS", "phone number": "PHONE_NUMBER", "location address": "STREET_ADDRESS",
             "location street": "STREET_ADDRESS", "location city": "LOCATION", "location country": "LOCATION", "location zip": "POSTAL_CODE",
             "url": "URL", "username": "USERNAME", "dob": "DATE_OF_BIRTH", "age": "AGE", "gender": "GENDER", "marital status": "MARITAL_STATUS",
             "account number": "ID_NUMBER", "bank account": "ID_NUMBER", "ssn": "ID_NUMBER", "passport number": "ID_NUMBER",
             "driver license": "ID_NUMBER", "university": "EDUCATION_ORG", "nationality": "NATIONALITY"}   # knowledgator's native label names
G2_LABELS = {"person": "PERSON", "email": "EMAIL_ADDRESS", "phone_number": "PHONE_NUMBER", "street_address": "STREET_ADDRESS",
             "postal_code": "POSTAL_CODE", "city": "LOCATION", "state_or_region": "LOCATION", "country": "LOCATION",
             "date_of_birth": "DATE_OF_BIRTH", "national_id_number": "ID_NUMBER", "passport_number": "ID_NUMBER", "tax_id": "ID_NUMBER",
             "iban": "ID_NUMBER", "bank_account": "ID_NUMBER", "license_number": "ID_NUMBER", "username": "USERNAME",
             "university": "EDUCATION_ORG", "school": "EDUCATION_ORG", "nationality": "NATIONALITY", "marital_status": "MARITAL_STATUS",
             "personal_website": "URL", "gender": "GENDER"}   # GLiNER2-PII snake_case labels + CV labels
# "organization" lets the CV layer turn schools in education sections into EDU_ORG (employers elsewhere stay visible)
CVORG_LABELS = dict(CV_LABELS, organization="ORGANIZATION")
KNORG_LABELS = dict(KN_LABELS, organization="ORGANIZATION")
LABEL_SETS = {"doc": DOC_LABELS, "pii": PII_LABELS, "cv": CV_LABELS, "cvorg": CVORG_LABELS, "kn": KN_LABELS, "knorg": KNORG_LABELS,
              "g2": G2_LABELS}

# Presidio's predefined pattern recognizers that can fire on CV content; the rest are US/IN/AU-specific IDs.
CV_PATTERNS = ["EmailRecognizer", "PhoneRecognizer", "IbanRecognizer", "CreditCardRecognizer", "UrlRecognizer", "IpRecognizer"]
EU_REGIONS = ("US", "GB", "DE", "FR", "ES", "IT", "NL", "BE", "SE", "FI", "DK", "NO", "PL", "PT", "CH", "AT", "IE", "IN", "CA", "BR")

# allow_list for analyze(): tool names and language names that NER models confuse with people/places
ALLOW = sorted({w for w in TECH | LANGUAGE_NAMES if len(w) > 1} | {w.capitalize() for w in TECH | LANGUAGE_NAMES if len(w) > 1})

# context words for Presidio's LemmaContextAwareEnhancer: the bench's multilingual field keys ("Name:", "Nationalität:", ...)
CONTEXT = sorted(FIELD_LABELS | {k for v in KV_KEYS.values() for k in v.split("|") if k.isalpha()})


class LineChunker(BaseTextChunker):
    """Presidio text chunker that packs whole lines up to max_chars and repeats `overlap_lines` lines between chunks.
    Chunks never start mid-word or mid-line, unlike CharacterBasedTextChunker whose overlap starts at a character offset.

    Example:
      LineChunker(25, 1).chunk("John Smith\\njohn@x.com\\nLeeds, UK\\n")
      returns  = [TextChunk("John Smith\\njohn@x.com\\n", 0, 22), TextChunk("john@x.com\\nLeeds, UK\\n", 11, 32)]
    """
    def __init__(self, max_chars=1000, overlap_lines=2):
        self.max_chars, self.overlap_lines = max_chars, overlap_lines

    def chunk(self, text):
        lines, pos = [], 0
        for ln in text.splitlines(keepends=True):
            lines.append((pos, ln)); pos += len(ln)
        out, i = [], 0
        while i < len(lines):
            j, size = i, 0
            while j < len(lines) and (j == i or size + len(lines[j][1]) <= self.max_chars):
                size += len(lines[j][1]); j += 1
            s = lines[i][0]; e = lines[j - 1][0] + len(lines[j - 1][1])
            out.append(TextChunk(text=text[s:e], start=s, end=e))
            if j >= len(lines): break
            i = max(i + 1, j - self.overlap_lines)
        return out


def _chunker(spec):
    """("char", size, overlap) -> Presidio CharacterBasedTextChunker; ("line", max_chars, overlap_lines) -> LineChunker."""
    kind, a, b = spec
    if kind == "char":
        from presidio_analyzer.chunkers import CharacterBasedTextChunker
        return CharacterBasedTextChunker(chunk_size=a, chunk_overlap=b)
    return LineChunker(a, b)


def gliner2_recognizer(repo, labels, threshold, chunker, context=None):
    """A Presidio LocalRecognizer that runs a GLiNER2 model (the gliner2 library; Presidio only ships a GLiNER v1 recognizer)."""
    from presidio_analyzer import LocalRecognizer, RecognizerResult, AnalysisExplanation
    from gliner2 import GLiNER2

    class GLiNER2Recognizer(LocalRecognizer):
        def load(self):
            self.model = GLiNER2.from_pretrained(os.path.join(MODELS, repo))

        def analyze(self, text, entities, nlp_artifacts=None):
            def predict(chunk):
                res = self.model.extract_entities(chunk, list(labels), threshold=threshold, include_confidence=True, include_spans=True)
                out = []
                for lab, items in res.get("entities", res).items():
                    for it in items:
                        if isinstance(it, dict) and "start" in it:
                            ent = labels.get(lab, lab)
                            if entities and ent not in entities: continue
                            out.append(RecognizerResult(ent, it["start"], it["end"], it.get("confidence", 1.0),
                                                        AnalysisExplanation(self.name, it.get("confidence", 1.0), textual_explanation=f"GLiNER2 {lab}")))
                return out
            return chunker.predict_with_chunking(text, predict)

    return GLiNER2Recognizer(supported_entities=sorted(set(labels.values())), name="GLiNER2Recognizer", supported_language="en",
                             context=context)


@lru_cache(None)
def _engine(name):
    """Build (and cache) the AnalyzerEngine for PG_CONFIGS[name]."""
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import GLiNERRecognizer, PhoneRecognizer
    c = PG_CONFIGS[name]
    nlp_model = c.get("nlp", "no_op")
    nlp = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "no_op" if nlp_model == "no_op" else "spacy",
                                               "models": [{"lang_code": "en", "model_name": nlp_model}]}).create_engine()
    reg = RecognizerRegistry(supported_languages=["en"])
    reg.load_predefined_recognizers(languages=["en"], nlp_engine=nlp)
    patterns = c.get("patterns", "all")
    keep_ner = c.get("keep_spacy", False)
    for r in list(reg.recognizers):
        is_ner = r.name in ("SpacyRecognizer", "StanzaRecognizer", "TransformersRecognizer")
        if is_ner and not keep_ner or not is_ner and (patterns == "none" or patterns == "cv" and r.name not in CV_PATTERNS) \
                or r.name in c.get("drop_recognizers", ()):
            reg.recognizers.remove(r)
    if c.get("phone_regions"):
        reg.recognizers = [r for r in reg.recognizers if r.name != "PhoneRecognizer"]
        reg.add_recognizer(PhoneRecognizer(supported_regions=c["phone_regions"]))
    ctx = CONTEXT if c.get("context") else None   # needs a spaCy NLP engine for lemmas
    for g in c["gliner"]:
        labels = LABEL_SETS[g.get("labels", "cv")]
        chunker = _chunker(g.get("chunker", ("char", 250, 50)))
        if g.get("kind") == "gliner2":
            reg.add_recognizer(gliner2_recognizer(g["model"], labels, g.get("threshold", 0.05), chunker, ctx))
        else:
            reg.add_recognizer(GLiNERRecognizer(model_name=os.path.join(MODELS, g["model"]), entity_mapping=labels,
                                                flat_ner=g.get("flat_ner", True), multi_label=g.get("multi_label", False),
                                                threshold=g.get("threshold", 0.05), map_location="cpu", text_chunker=chunker,
                                                name=f"GLiNER:{g['model'].split('/')[-1]}", context=ctx))
    # whole-word matching: the default substring mode lets "age" match "language" and "sex" match "Sussex"
    enh = LemmaContextAwareEnhancer(context_matching_mode="whole_word")
    return AnalyzerEngine(registry=reg, nlp_engine=nlp, supported_languages=["en"], context_aware_enhancer=enh)


def presidio_gliner_detector(name):
    """Return a callable text -> spans for Presidio config `name`. Scores are Presidio's (GLiNER confidence / pattern score).

    Example (illustrative, not executed):
      fn       = presidio_gliner_detector("doc-urchade")
      text     = "Hello, my name is Rafi Mor, I'm from Binyamina and I work at Microsoft."
      returns  = fn(text)
               = [{"start": 18, "end": 26, "label": "NAME", "src": "pg:doc-urchade", "score": 0.97},
                  {"start": 37, "end": 46, "label": "LOCATION", ...}, {"start": 60, "end": 69, "label": "ORG", ...}]
    """
    eng = _engine(name)
    allow = ALLOW if PG_CONFIGS[name].get("allow_list") else None
    def fn(text):
        res = eng.analyze(text=text, language="en", score_threshold=0.0, allow_list=allow)
        return [_S(r.start, r.end, PG_MAP.get(r.entity_type, "OTHER"), f"pg:{name}", r.score) for r in res]
    return fn


U, NV, KB, KL, GL, G2 = ("urchade/gliner_multi_pii-v1", "nvidia/gliner-PII", "knowledgator/gliner-pii-base-v1.0",
                         "knowledgator/gliner-pii-large-v1.0", "gliner-community/gliner_large-v2.5", "fastino/gliner2-privacy-filter-PII-multi")
LINE = ("line", 1000, 2)
# All GLiNER thresholds are 0.05 so the saved output can be swept; the sample's 0.30 default is applied at scoring time.
PG_CONFIGS = {
    # 1. the Presidio sample as published: en_core_web_sm, SpacyRecognizer removed, 4 labels, nested multi-label, default chunker
    "doc-urchade": {"nlp": "en_core_web_sm", "gliner": [{"model": U, "labels": "doc", "flat_ner": False, "multi_label": True}]},
    # 2. one change at a time on top of the sample
    "urchade-cv": {"nlp": "en_core_web_sm", "gliner": [{"model": U, "labels": "cv", "flat_ner": False, "multi_label": True}]},
    "urchade-cv-flat": {"gliner": [{"model": U, "labels": "cv"}]},
    "urchade-cv-flat-line": {"gliner": [{"model": U, "labels": "cv", "chunker": LINE}]},
    "urchade-pii-flat-line": {"gliner": [{"model": U, "labels": "pii", "chunker": LINE}]},
    "urchade-cvorg-flat-line": {"gliner": [{"model": U, "labels": "cvorg", "chunker": LINE}]},
    "urchade-pii-nested-line": {"gliner": [{"model": U, "labels": "pii", "flat_ner": False, "multi_label": True, "chunker": LINE}]},
    "urchade-pii-flat-line600": {"gliner": [{"model": U, "labels": "pii", "chunker": ("line", 600, 2)}]},
    "urchade-pii-flat-whole": {"gliner": [{"model": U, "labels": "pii", "chunker": ("line", 5000, 0)}]},
    # 3. other GLiNER checkpoints, best settings from 2
    "nvidia-cv-line": {"gliner": [{"model": NV, "labels": "cv", "chunker": LINE}]},
    "kn-base-kn-line": {"gliner": [{"model": KB, "labels": "kn", "chunker": LINE}]},
    "kn-base-kn-char": {"gliner": [{"model": KB, "labels": "kn"}]},
    "kn-large-kn-line": {"gliner": [{"model": KL, "labels": "kn", "chunker": LINE}]},
    "kn-large-knorg-line": {"gliner": [{"model": KL, "labels": "knorg", "chunker": LINE}]},
    "gliner-large-cv-line": {"gliner": [{"model": GL, "labels": "cv", "chunker": LINE}]},
    "gliner2-g2-line": {"gliner": [{"model": G2, "kind": "gliner2", "labels": "g2", "chunker": ("line", 1500, 2)}]},
    # 4. Presidio-side switches. kn-large-tuned turns all on at once (EU phone regions turn year ranges into PHONE);
    #    the kn-base-* rows switch them one at a time on the cheaper model
    "kn-large-tuned": {"patterns": "cv", "phone_regions": EU_REGIONS, "allow_list": True, "drop_recognizers": ("UrlRecognizer",),
                       "gliner": [{"model": KL, "labels": "kn", "chunker": LINE}]},
    "kn-large+spacy-sm": {"nlp": "en_core_web_sm", "keep_spacy": True, "patterns": "cv", "phone_regions": EU_REGIONS, "allow_list": True,
                          "drop_recognizers": ("UrlRecognizer",), "gliner": [{"model": KL, "labels": "kn", "chunker": LINE}]},
    "kn-base-knorg-line": {"gliner": [{"model": KB, "labels": "knorg", "chunker": LINE}]},
    "kn-base-knorg-line-cvpat": {"patterns": "cv", "gliner": [{"model": KB, "labels": "knorg", "chunker": LINE}]},
    "kn-base-knorg-line-nopat": {"patterns": "none", "gliner": [{"model": KB, "labels": "knorg", "chunker": LINE}]},
    "kn-base-knorg-line-allow": {"allow_list": True, "gliner": [{"model": KB, "labels": "knorg", "chunker": LINE}]},
    "kn-base-knorg-line-euphone": {"phone_regions": EU_REGIONS, "gliner": [{"model": KB, "labels": "knorg", "chunker": LINE}]},
    "kn-base-knorg-line-nourl": {"drop_recognizers": ("UrlRecognizer",), "gliner": [{"model": KB, "labels": "knorg", "chunker": LINE}]},
    # 5. an ensemble hosted in one Presidio engine: GLiNER2 + Knowledgator base, merged by Presidio's duplicate removal
    "ens-gliner2+kn-base": {"gliner": [{"model": G2, "kind": "gliner2", "labels": "g2", "chunker": ("line", 1500, 2)},
                                       {"model": KB, "labels": "knorg", "chunker": LINE}]},
    # 6. the findings combined: CV-relevant patterns only, allow_list, line chunks; optionally Presidio context enhancement
    "best-g2": {"patterns": "cv", "allow_list": True, "gliner": [{"model": G2, "kind": "gliner2", "labels": "g2", "chunker": ("line", 1500, 2)}]},
    "best-g2-ctx": {"nlp": "en_core_web_sm", "context": True, "patterns": "cv", "allow_list": True,
                    "gliner": [{"model": G2, "kind": "gliner2", "labels": "g2", "chunker": ("line", 1500, 2)}]},
    "best-ens": {"patterns": "cv", "allow_list": True, "gliner": [{"model": G2, "kind": "gliner2", "labels": "g2", "chunker": ("line", 1500, 2)},
                                                                    {"model": KB, "labels": "knorg", "chunker": LINE}]},
    "best-ens-ctx": {"nlp": "en_core_web_sm", "context": True, "patterns": "cv", "allow_list": True,
                     "gliner": [{"model": G2, "kind": "gliner2", "labels": "g2", "chunker": ("line", 1500, 2)},
                                {"model": KB, "labels": "knorg", "chunker": LINE}]},
}
