"""CV PII masking pipeline: regex + key-value rules + pluggable model detectors + CV policy layer
+ identity propagation + safety net. Every stage returns character spans; text is never regenerated."""
import re
from functools import lru_cache
from .lexicons import (TECH, MONTHS, COMMON_WORDS, SECTION_WORDS, EDU_KEYWORDS, EDU_ACRONYMS, EDU_EXCLUDE, COUNTRIES,
                       KV_KEYS, FIELD_LABELS, PHONE_KEYS, HANDLE_KEYS, MARITAL_WORDS, LOC_KEYS, LANGUAGE_NAMES)

LABELS = ["NAME", "EMAIL", "PHONE", "ADDRESS", "LOCATION", "URL", "HANDLE", "ID", "DOB", "NATIONALITY", "MARITAL",
          "EDU_ORG", "ORG", "DATE", "GENDER", "AGE", "RELIGION", "OTHER"]


def S(start, end, label, src, score=1.0):
    """Build a span dict.

    Example:
      start, end, label, src = 0, 10, "NAME", "re"
      returns  = {"start": 0, "end": 10, "label": "NAME", "src": "re", "score": 1.0}
    """
    return {"start": start, "end": end, "label": label, "src": src, "score": score}


def is_edu(t):
    """True if the text names an educational institution (keyword/acronym, not excluded).

    Example:
      t        = "University of Leeds"
      returns  = True
      t        = "Acme Ltd"
      returns  = False
    """
    # pad with spaces so edge-anchored lexicon entries (e.g. "chu ") also match at the ends
    low = " " + t.lower() + " "
    # exclusions (hospital, nhs, trust, ...) win over edu keywords
    if any(x in low for x in EDU_EXCLUDE):
        return False
    # keywords match as substrings; acronyms must be whole tokens and case-sensitive
    return any(k in low for k in EDU_KEYWORDS) or any(re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", t) for a in EDU_ACRONYMS)


@lru_cache(None)
def cities():
    """Cached set of city names (population >= 15k, plus alternate names) from geonamescache.

    Example:
      (no inputs)
      returns  = set()                          # geonamescache not installed in venv -> empty set
                                                # with it installed: {"Leeds", "Berlin", ...}
    """
    try:
        import geonamescache
        out = set()
        for v in geonamescache.GeonamesCache().get_cities().values():
            # small towns are skipped to limit false hits on ordinary capitalised words
            if v["population"] >= 15000:
                out.add(v["name"])
                # add capitalised alternate names longer than 3 chars (local spellings, transliterations)
                out.update(a for a in v["alternatenames"] if a and len(a) > 3 and a[0].isupper() and " " not in a[:1])
        return out
    except Exception:
        return set()

# ---------------------------------------------------------------- structure
ALL_SECTION_WORDS = sorted(set(sum(SECTION_WORDS.values(), [])), key=len, reverse=True)
TITLE_WORDS = re.compile(r"(?i)\b(?:engineer|developer|manager|analyst|architect|consultant|nurse|designer|director|lead|officer|specialist|"
                         r"scientist|administrator|accountant|contractor|summary|curriculum|vitae|resume|résumé|cv|profile|lebenslauf|"
                         r"ansioluettelo|intern|head|president|founder|owner|teacher|professor|student|technician|coordinator|executive|"
                         r"ingenieur|ingénieur|ingeniero|ingegnere|entwickler|développeur|desarrollador|sviluppatore|berater|chef de|jefe|"
                         r"responsable|kehittäjä|insinööri|konsult|utvecklare|physio|physiotherapist|pharmacist|lawyer|attorney|doctor|physician)\b")
YEARISH = re.compile(r"(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present|now|today|heute|actuel|presente|nu|nyt|nå|nykyinen|oggi|atual)|\b(?:19|20)\d{2}\b.*\b(?:19|20)\d{2}\b", re.I)


def is_section_title(line):
    """True if the line is a CV section heading (e.g. EXPERIENCE, Education:).

    Example:
      line     = "EXPERIENCE"
      returns  = True
      line     = "Work Experience:"
      returns  = True
      line     = "Engineer at Acme Ltd"
      returns  = False
    """
    # strip markdown marks and a trailing colon: "## Education:" -> "Education"
    raw = line.strip().strip("#*_ ").rstrip(":").strip()
    t = raw.lower()
    # long lines are content, not headings
    if not t or len(t) > 45:
        return False
    # must start with a known section word ("skills", "skills & tools", "skills / languages")
    hit = any(t == w or t.startswith(w + " ") or t.startswith(w + " &") or t.startswith(w + " /") for w in ALL_SECTION_WORDS)
    # ...and be formatted like a heading: ALL CAPS, trailing colon, markdown #, exact word or <= 4 words
    return hit and (raw.isupper() or line.strip().endswith(":") or line.strip().startswith("#") or t in ALL_SECTION_WORDS or len(t.split()) <= 4)


def line_index(text):
    """Split text into (start, end, line) tuples with absolute character offsets.

    Example:
      text     = "ab\\ncd"
      returns  = [(0, 2, "ab"), (3, 5, "cd")]
    """
    lines, pos = [], 0
    for ln in text.split("\n"):
        # +1 skips the "\n" so offsets index into the original text
        lines.append((pos, pos + len(ln), ln)); pos += len(ln) + 1
    return lines


def section_of_lines(text):
    """Return list of (start, end, line, section). 'header' = top zone before the first section title or
    the first dated/long content line.

    Example:
      text     = "John Smith\\njohn@x.com | Leeds, UK\\n\\nEXPERIENCE\\nEngineer at Acme Ltd, Leeds 2019-2023"
      returns  = [(0, 10, "John Smith", "header"), (11, 33, "john@x.com | Leeds, UK", "header"), (34, 34, "", "header"),
                  (35, 45, "EXPERIENCE", "experience"),   # section title switches the section
                  (46, 83, "Engineer at Acme Ltd, Leeds 2019-2023", "experience")]
    """
    out, cur = [], "header"
    for s, e, ln in line_index(text):
        if is_section_title(ln):
            # map the heading to its section key; unknown headings become "other"
            t = ln.strip().strip("#*_ ").rstrip(":").lower()
            for sec, words in SECTION_WORDS.items():
                if any(t == w or t.startswith(w) for w in words):
                    cur = sec; break
            else:
                cur = "other"
        # header ends early at the first dated or long line (e.g. a job entry with no heading above it);
        # a year on a DOB line does not count
        elif cur == "header" and (YEARISH.search(ln) or len(ln.split()) > 14 or (re.search(r"\b(?:19|20)\d{2}\b", ln) and len(ln.split()) > 4 and not re.search(rf"(?i){KV_KEYS['DOB']}", ln))):
            cur = "other"
        # a horizontal rule starts the footer (sign-off / contact block)
        if ln.strip() in ("---", "___", "***"):
            cur = "footer"
        out.append((s, e, ln, cur))
    return out


def section_at(sections, pos):
    """Return (section, line, line_start) for the line containing character position pos.

    Example:
      sections = section_of_lines("John Smith\\njohn@x.com | Leeds, UK\\n\\nEXPERIENCE\\nEngineer at Acme Ltd, Leeds 2019-2023")
      pos      = 24
      returns  = ("header", "john@x.com | Leeds, UK", 11)
      pos      = 70
      returns  = ("experience", "Engineer at Acme Ltd, Leeds 2019-2023", 46)
    """
    for s, e, ln, sec in sections:
        # inclusive end so a position on the "\n" still maps to its line
        if s <= pos <= e:
            return sec, ln, s
    # pos outside the text
    return "other", "", 0


def is_contact_line(line):
    """True if the line holds an email, an international phone number or a phone key (Tel:, Mobile: ...).

    Example:
      line     = "john@x.com | Leeds, UK"
      returns  = True
      line     = "Tel: 0113 496 0000"
      returns  = True
      line     = "Leeds, UK"
      returns  = False
    """
    # email, or "+CC ..." number, or a phone key followed by ":" or "."
    return bool(EMAIL_RE.search(line) or re.search(r"\+\d{1,3}[\s\d().-]{6,}", line) or re.search(rf"(?i)(?:^|\W)(?:{PHONE_KEYS})\.?\s*[:.]", line))


def contact_zone(sec, line):
    """True if the line is in a header/footer/personal section or is itself a contact line.

    Example:
      sec, line = "header", "Leeds"
      returns   = True
      sec, line = "experience", "john@x.com"
      returns   = True                          # contact line, any section
      sec, line = "experience", "Leeds"
      returns   = False
    """
    return sec in ("header", "footer", "personal") or is_contact_line(line)

# ---------------------------------------------------------------- regex detectors
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_AT = r"(?:at|arobase|arroba|chiocciola|snabel-?a|kropka|ät|miukumauku)"
_DOT = r"(?:dot|punt|point|punto|ponto|piste|prick|punkt|kropka|pkt)"
OBF_EMAIL_RE = re.compile(rf"(?i)\b\w[\w.-]*(?:\s*[\[({{]{_DOT}[\])}}]\s*\w+)*\s*[\[({{]{_AT}[\])}}]\s*[\w-]+(?:\s*(?:[\[({{]{_DOT}[\])}}]|\.)\s*[\w-]+)+")
OBF_EMAIL_PLAIN_RE = re.compile(rf"(?i)\b[a-z][\w.]*(?: {_DOT} \w+)* {_AT} [a-z][\w-]* {_DOT} (?:com|org|net|io|de|fr|es|nl|se|fi|uk|co|eu|me|dev|it|pl|pt|dk|no|be|ch|at)\b")
PHONE_RE = re.compile(r"(?<![\w+.,/-])(?:\+\d{1,3}[ .-]?)?(?:\(0?\d{1,4}\)[ .-]?)?\d{1,5}(?:(?:[ .-]| / |/)\d{1,8}){1,5}(?:\s*(?:ext\.?|x|poste|durchwahl|anknytning)\s*\d{1,5})?(?![\w]|[.,]\d)")
PHONE_ONE_RE = re.compile(r"(?<![\w+.,/-])(?:\+\d{8,14}|0\d{8,11})(?![\w]|[.,]\d)")
DIGIT_WORDS = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|plus)"
SPELLED_PHONE_RE = re.compile(rf"(?i)\b{DIGIT_WORDS}(?:[\s,-]+{DIGIT_WORDS}){{6,}}\b")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s,;|)\]>]+|\b(?:[\w-]+\.)?(?:linkedin\.com|github\.com|gitlab\.com|twitter\.com|x\.com|medium\.com|behance\.net|dribbble\.com|stackoverflow\.com|xing\.com|about\.me|t\.me|kaggle\.com|scholar\.google\.com|orcid\.org|researchgate\.net)/[^\s,;|)\]>]+")
BARE_DOMAIN_RE = re.compile(r"(?i)(?<![\w@./-])(?:[\w-]+\.)+(?:com|io|dev|me|net|org|se|de|fr|es|nl|fi|it|pl|pt|dk|no|eu|co|ai|app|page|site|xyz|uk|ch|at|be|info|tech|design|studio|blog)(?:/[^\s,;|)\]>]*)?(?![\w@])")
PERSONAL_HOSTS = ("linkedin.com", "twitter.com", "x.com/", "medium.com/@", "behance.net", "dribbble.com", "stackoverflow.com/users",
                  "about.me", "t.me/", "xing.com", "kaggle.com", "scholar.google", "orcid.org", "researchgate.net/profile")
HANDLE_RE = re.compile(r"(?<![\w@.])@[A-Za-z0-9_.]{3,}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
SPACED_NAME_RE = re.compile(r"^[ \t]*(?:[A-ZÀ-ÖØ-Þ] ){2,}[A-ZÀ-ÖØ-Þ](?:\s{2,}(?:[A-ZÀ-ÖØ-Þ] )*[A-ZÀ-ÖØ-Þ])+[ \t]*$", re.M)
ID_KEYS = (r"validation(?: no\.?| number| code)?|learner(?: no\.?| number)|candidate(?: no\.?| number| id)|permit(?: no\.?| number)?|num[ée]ro de permis|n\.?\s?º|n\.?\s?°|nº|certificate(?: id| no\.?| number)|cert\.? (?:no\.?|id)|licence id|account(?: no\.?| number| nr\.?)?|acct|sort code|registration(?: no\.?| number)?|reg\.? no\.?|reg\.? nr\.?|credential(?: id)?|"
           r"licen[cs]e(?: no\.?| number| id)?|driving licen[cs]e(?: no\.?| number)?|driver'?s licen[cs]e(?: no\.?| number)?|führerschein(?:nummer)?|permis(?: de conduire)?|carn[eé](?: de conducir)?|patente|"
           r"passport(?: no\.?| number)?|pass(?:nummer|port-nr\.?)|reisepass|passeport|pasaporte|passaporto|paszport|"
           r"national id(?: number| no\.?)?|id number|id no\.?|id card|identity card|personalausweis|carte d'identité|cni|dni|nie|nif|nif/nie|"
           r"ni number|nino|national insurance(?: number| no\.?)?|ssn|social security(?: number| no\.?)?|"
           r"vat(?: no\.?| number)?|tax id|tin|tax number|steuer-?id|steuernummer|steueridentifikationsnummer|codice fiscale|c\.f\.|"
           r"pesel|nip|regon|cpr(?:-nr\.?| number)?|fødselsnummer|personnummer|personal id|henkilötunnus|hetu|sotu|bsn|burgerservicenummer|"
           r"n° de sécurité sociale|numéro de sécurité sociale|sécurité sociale|nss|nuss|curp|rfc|cnp|cpf|nif|"
           r"employee id|staff id|staff no\.?|member(?:ship)? (?:no\.?|number|id)|mitgliedsnummer|gmc(?: no\.?| number)?|nmc(?: pin)?|hcpc(?: no\.?)?|"
           r"bar(?: no\.?| number)|big-?nummer|rpps|colegiad[oa](?: n[ºo°]\.?)?|n[ºo°] de colegiad[oa]|iban|bic|swift|orcid")
ID_KV_RE = re.compile(rf"(?i)(?<![\w-])(?:{ID_KEYS})(?![\w])\s*(?:no\.?|nr\.?|number|#|n[ºo°]\.?)?\s*[:#|\t]?\s*(?-i:([A-Z0-9][A-Z0-9 ./-]{{2,34}}[A-Z0-9]))")
DATE_RE = (r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\.?\s*(?:de\s+)?[A-Za-zäöüéèåæøûôçñ]+\.?\s+(?:de\s+)?\d{4}|"
           r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}|(?:19|20)\d{2})")
YEAR_RANGE_RE = re.compile(r"^(?:19|20)\d{2}\s*[-–]\s*(?:(?:19|20)\d{2}|present)$", re.I)
KV_SEP = r"(?:\s*[:|\t]\s*|\s{2,}|\s+[-–]\s+)"
EXPERIENCE_WORDS = r"(?:years?|yrs|jahre|ans|años|anos|anni|jaar|år|vuotta|lat)\s+(?:of\s+)?(?:experience|erfahrung|berufserfahrung|d'expérience|de experiencia|de experiência|di esperienza|ervaring|erfarenhet|erfaring|kokemus|doświadczenia)"


def _trim(text, a, b):
    """Shrink [a, b) past surrounding whitespace, quotes, markdown marks and trailing punctuation.

    Example:
      text, a, b = '  "Leeds",  ', 0, 12
      returns    = (3, 8)                       # text[3:8] == "Leeds"
    """
    # left: whitespace, markdown emphasis and quotes
    while a < b and text[a] in " \t*_`\"'": a += 1
    # right: same plus trailing punctuation
    while b > a and text[b - 1] in " .,;\t*_`\"'": b -= 1
    return a, b


def regex_detect(text, sections, cfg):
    """Rule-based detectors (email, phone, IDs, URLs, key-value fields, DOB, schools, ...) -> candidate spans.

    Example (default cfg):
      text     = "Jane Doe\\njane@x.com | +44 7700 900123\\nDate of Birth: 12/03/1990\\n\\nEXPERIENCE\\n"
                 "Support line 2019-2023, ref 1234 5678\\nBSc, University of Leeds 2015"
      sections = section_of_lines(text)
      cfg      = {}
      returns  = [S(9, 19, "EMAIL", "re"),            # "jane@x.com"
                  S(22, 37, "PHONE", "re"),           # "+44 7700 900123"  leading "+" -> strong phone
                  S(53, 63, "DOB", "re-kv"),          # "12/03/1990"       (emitted twice: KV rule + inline DOB rule)
                  S(53, 63, "DOB", "re-kv"),
                  S(119, 138, "EDU_ORG", "re-edu")]   # "University of Leeds"
                  # "1234 5678" not a phone: weak number on an experience line without a phone key
    """
    out = []
    # emails: plain, then obfuscated ("jane [at] x [dot] com", "jane at x dot com")
    for m in EMAIL_RE.finditer(text): out.append(S(m.start(), m.end(), "EMAIL", "re"))
    for m in list(OBF_EMAIL_RE.finditer(text)) + list(OBF_EMAIL_PLAIN_RE.finditer(text)):
        out.append(S(m.start(), m.end(), "EMAIL", "re"))
    # phones: a weak number only counts on a line that is phone-keyed, in a contact zone, or a contact line
    phone_line_ok = {}
    for s, e, ln, sec in sections:
        phone_line_ok[s] = bool(re.search(rf"(?i)(?:^|[\W_])(?:{PHONE_KEYS})(?:[\W_]|$)", ln)) or sec in ("header", "footer", "personal") or is_contact_line(ln)
    for m in list(PHONE_RE.finditer(text)) + list(PHONE_ONE_RE.finditer(text)):
        g = m.group(); digits = re.sub(r"\D", "", g.split("ext")[0])
        # reject by digit count (E.164 max 15), year ranges, dates and ORCIDs
        if not (7 <= len(digits) <= 15) or YEAR_RANGE_RE.match(g.strip()):
            continue
        if re.fullmatch(r"(?:19|20)\d{2}\s*[-–/]\s*(?:19|20)\d{2}", g) or re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", g) or ORCID_RE.fullmatch(g):
            continue
        _, ln, ls = section_at(sections, m.start())
        # "+44 ...", "(212) ..." or a long 0-prefixed number is a phone anywhere
        strong = g.startswith("+") or g.startswith("(") or (g.startswith("0") and len(digits) >= 9)
        if not strong and not phone_line_ok.get(ls, False):
            continue
        out.append(S(m.start(), m.end(), "PHONE", "re"))
    # spelled-out digits ("zero seven seven ...") are always phones
    for m in SPELLED_PHONE_RE.finditer(text): out.append(S(m.start(), m.end(), "PHONE", "re"))
    # handles: bare @name, then keyed ("Twitter: jdoe")
    for m in HANDLE_RE.finditer(text): out.append(S(m.start(), m.end(), "HANDLE", "re"))
    for m in re.finditer(rf"(?im)(?:^|[|•·;,]\s*|\s{{2,}})(?:{HANDLE_KEYS})(?:\s*(?:id|handle|name))?\s*[:|]\s*([^\s|,;]+)", text):
        out.append(S(m.start(1), m.end(1), "HANDLE", "re-kv"))
    # ids: IBAN (>= 10 digits, filters out uppercase words), ORCID, "#123456"
    for m in IBAN_RE.finditer(text):
        if sum(c.isdigit() for c in m.group()) >= 10: out.append(S(m.start(), m.end(), "ID", "re"))
    for m in ORCID_RE.finditer(text): out.append(S(m.start(), m.end(), "ID", "re"))
    for m in re.finditer(r"(?<![\w&])#\s?(\d{6,})\b", text): out.append(S(m.start(1), m.end(1), "ID", "re"))
    # sign-off place/date lines ("Berlin, im Mai 2025", "Napoli, 12/03/2025")
    for m in re.finditer(r"(?m)^[ \t]*([A-ZÀ-Ý][\w'’-]+(?: [A-ZÀ-Ý][\w'’-]+)?),\s*(?:(?:den|le|il|am|im|el|em|d\.)\s+)?(?:\d{1,2}[./]\s?\d{1,2}[./]\s?\d{2,4}|\d{1,2}\.?\s+(?:de\s+)?[A-Za-zäöüéèûç]+\.?\s+(?:de\s+)?\d{4}|[A-Za-zäöüéèûç]+\s+\d{4})\b", text):
        if not TITLE_WORDS.search(m.group(1)) and not is_tech(m.group(1)):
            out.append(S(m.start(1), m.end(1), "LOCATION", "re-signoff"))
    # birthplace in prose ("born in Porto", "née à Lyon")
    for m in re.finditer(r"\b(?:n[ée]e?|nat[oa]|born|geboren|nacid[oa]|född|født|syntynyt)\s+(?:à|a|in|en|em|i)\s+([A-ZÀ-Ý][\w'’-]+(?: [A-ZÀ-Ý][\w'’-]+)?)", text, re.I):
        if m.group(1)[0].isupper(): out.append(S(m.start(1), m.end(1), "LOCATION", "re-kv"))
    # keyed ids ("Passport No: N4419827"); cut the value at a double space or the next capitalised word
    for m in ID_KV_RE.finditer(text):
        v = m.group(1)
        v = re.split(r"\s{2,}|\s(?=[A-Z][a-z])", v)[0]
        if sum(c.isdigit() for c in v) >= 4:
            out.append(S(m.start(1), m.start(1) + len(v), "ID", "re-kv"))
    # letter-spaced names in the header ("R O B E R T   H A Y E S")
    for m in SPACED_NAME_RE.finditer(text):
        sec, _, _ = section_at(sections, m.start())
        if sec in ("header", "personal", "footer"):
            a, b = _trim(text, m.start(), m.end())
            out.append(S(a, b, "NAME", "re-spaced"))
    # urls: explicit (http/www/known hosts) always; bare domains only in contact/links zones or on a web/portfolio line
    for m in URL_RE.finditer(text):
        out.append(S(m.start(), m.end(), "URL", "re", 1.0))
    for m in BARE_DOMAIN_RE.finditer(text):
        sec, ln, ls = section_at(sections, m.start())
        if sec in ("header", "footer", "personal", "links") or is_contact_line(ln) or re.search(r"(?i)\b(?:web|website|site|portfolio|homepage|blog)\b", ln):
            out.append(S(m.start(), m.end(), "URL", "re-bare"))
    # key-value personal fields; value runs to end of line or next field separator
    for label, keys in KV_KEYS.items():
        pat = rf"(?im)(?:^|[|•·;]\s*|\t|\s{{3,}})[ \t*_\-]*(?:{keys})(?:_\w+)?(?:\(s\))?(?:\s*/\s*[\w ]+?(?:\(s\))?)?[*_]*{KV_SEP}([^\n|\t]+?)(?=\s{{3,}}|\s*[|\t]|\s*$)"
        for m in re.finditer(pat, text):
            v0, v1 = _trim(text, m.start(1), m.end(1))
            val = text[v0:v1]
            if not val: continue
            # per-label sanity checks on the value
            if label == "DOB" and not re.search(r"\d", val): continue
            if label == "AGE" and not re.search(r"\d", val): continue
            if label == "NAME" and (re.search(r"\d|@", val) or len(val.split()) > 6): continue
            if label == "ADDRESS" and cfg.get("kv_address", True) is False: continue
            if label == "ADDRESS" and re.fullmatch(r"(?i)remote|hybrid|on-?site|anywhere|flexible|n/?a", val): continue
            # age is masked under DOB
            lab = "DOB" if label == "AGE" else label
            out.append(S(v0, v1, lab, "re-kv"))
    # DOB / age inline
    for m in re.finditer(rf"(?i)\b(?:{KV_KEYS['DOB']})\s*[:,]?\s*(?:on\s+|le\s+|el\s+|il\s+|den\s+)?({DATE_RE})", text):
        out.append(S(m.start(1), m.end(1), "DOB", "re-kv"))
    # keyed age ("Age: 34")
    for m in re.finditer(r"(?i)\b(?:age|âge|edad|idade|età|alter|ålder|alder|ikä|wiek)\s*[:\-]?\s*(\d{2})\b", text):
        out.append(S(m.start(1), m.end(1), "DOB", "re-age"))
    # "34 years old" / "34 anos", but not "10 years of experience"
    for m in re.finditer(rf"(?i)\b(\d{{2}}\s*(?:ans|años|anos|anni|jahre(?: alt)?|år|jaar|vuotta|lat|years old|yo|y/o))\b(?!\s+(?:of\s+)?(?:experience|erfahrung|d'expérience|de experiencia|de experiência|di esperienza|ervaring|erfarenhet|erfaring|kokemus|doświadczenia|in|im|dans|en|na|i|di|de|w))", text):
        _, ln, _ = section_at(sections, m.start())
        if not re.search(EXPERIENCE_WORDS, ln, re.I):
            out.append(S(m.start(1), m.end(1), "DOB", "re-age"))
    # residence in prose
    for m in re.finditer(r"(?i)\b(?:live in|living in|based in|reside in|residing in|resident (?:of|in)|relocat\w+ to|wohne in|wohnhaft in|lebe in|j'habite à|j'habite en|domicilié à|vivo en|resido en|woon in|bor i|bor på|asun|abito a|vivo a|moro em|resido em|mieszkam w)\s+([A-ZÀ-ÝŁŚŻ][^\n.;()]{1,60})", text):
        a, b = _trim(text, m.start(1), m.end(1))
        # stop the place at a conjunction or "since" ("based in Leeds and ...")
        cut = re.search(r"\s+(?:and|und|et|y|e|en|och|og|ja|i|since|seit|depuis|desde|sinds|sedan|siden|with|mit|avec)\s", text[a:b])
        if cut: b = a + cut.start()
        out.append(S(a, b, "LOCATION", "re-prose"))
    # inline nationality / marital / gender keyed with a single space ("Nationalité française")
    for lab in ("NATIONALITY",):
        for m in re.finditer(rf"(?i)\b(?:{KV_KEYS[lab]})\s+(?-i:([a-zà-ÿ]+|[A-ZÀ-Ý][a-zà-ÿ]+(?:\s*/\s*[A-ZÀ-Ý][a-zà-ÿ]+)?))\b", text):
            out.append(S(m.start(1), m.end(1), lab, "re-kv"))
    # "Irish citizen", "British/Irish national"
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s*/\s*[A-Z][a-z]+)?)\s+(?:citizen|national|citizenship|passport holder)\b", text):
        out.append(S(m.start(1), m.end(1), "NATIONALITY", "re-kv"))
    # pronouns -> GENDER
    for m in re.finditer(r"\((?:he|she|they|xe|ze)/(?:him|her|them|they|hers|theirs|xem|zir)\)|\b(?:he|she|they)/(?:him|her|them)\b", text):
        out.append(S(m.start(), m.end(), "GENDER", "re"))
    # marital words standing alone in header/personal zones ("Brasileiro, casado, 34 anos")
    for s_, e_, ln, sec in sections:
        if sec in ("header", "personal", "footer"):
            for m in re.finditer(r"(?<![\w-])([A-Za-zÀ-ÿŻżŁł]+)(?![\w-])", ln):
                if m.group(1).lower() in MARITAL_WORDS:
                    out.append(S(s_ + m.start(1), s_ + m.end(1), "MARITAL", "re-marital"))
    # birthplace
    for m in re.finditer(rf"(?i)\b(?:{LOC_KEYS})\s*[:|\t]?\s*(?-i:([A-ZÀ-Ý][\w'’-]+(?:[ ,]+[A-ZÀ-Ý(][\w'’)-]+){{0,3}}))", text):
        out.append(S(m.start(1), m.end(1), "LOCATION", "re-kv"))
    # "born in Porto, 12/03/1990" -> place and DOB together
    for m in re.finditer(rf"(?i)\b(?:nat[oa]|née?|nacid[oa]|geboren|född|født|born)\s+(?:a|à|en|in|i|em)\s+(?-i:([A-ZÀ-Ý][\w'’-]+))(?:\s*\([A-Z]{{2}}\))?[,\s]+(?:(?:il|le|el|am|den|on|em)\s+)?({DATE_RE})", text):
        out.append(S(m.start(1), m.end(1), "LOCATION", "re-kv")); out.append(S(m.start(2), m.end(2), "DOB", "re-kv"))
    # "Tokyo-based" in header / profile
    for m in re.finditer(r"(?<![\w-])([A-ZÀ-Ý][a-zà-ÿ]+(?:[ -][A-ZÀ-Ý][a-zà-ÿ]+)?)-based\b", text):
        sec, _, _ = section_at(sections, m.start())
        if sec not in ("experience", "education", "skills"):
            out.append(S(m.start(1), m.end(1), "LOCATION", "re-prose"))
    # universities / schools (segment containing an edu keyword)
    for s, e, ln, sec in sections:
        for seg in re.finditer(r"[^,;|\n()—–\t]+", ln):
            t = seg.group()
            if is_edu(t):
                a, b = seg.start(), seg.end()
                # drop a "Label:" prefix that is not itself part of the school name
                if ":" in t and not is_edu(t.split(":")[0]):
                    a += t.index(":") + 1
                # drop a leading degree ("MSc Physics, ") when the rest is still a school
                mm = re.match(r"\s*(?:B\.?Sc\.?|M\.?Sc\.?|B\.?A\.?|M\.?A\.?|MBA|Ph\.?D\.?|BEng|MEng|Dipl\.?[-\w]*|DI|LL\.?[BM]|M\.?Eng|B\.?Eng|Master|Bachelor|Licence|Laurea|Grado|Máster|Diplom)\b[^A-Z]*?(?=[A-Z][\w-]*\s*(?:[A-Z]|of|de|di|für|van|i\b))", t)
                if mm and is_edu(t[mm.end():]): a += mm.end()
                # drop a trailing year
                m2 = re.search(r"\s+(?:\d{4}|in \d{4}|\d{2}/\d{4})\s*$", ln[a:b])
                if m2: b = a + m2.start()
                a2, b2 = _trim(ln, a, b)
                if b2 - a2 > 2: out.append(S(s + a2, s + b2, "EDU_ORG", "re-edu"))
                # a bracketed short form right after the school, e.g. "(UCL)"
                pm = re.match(r"\s*\(([A-ZÀ-Ý][A-Za-zÀ-ÿ]{1,9})(?:\s*/[^)]*)?\)", ln[seg.end():])
                if pm: out.append(S(s + seg.end() + pm.start(1), s + seg.end() + pm.end(1), "EDU_ORG", "re-edu"))
    for s_, e_, ln, sec in sections:
        # standalone acronyms in the education section ("MIT,"), minus degrees, exams and tech terms
        if sec == "education" and cfg.get("edu_acronyms", True):
            for m in re.finditer(r"(?:^|(?<=[,(–—|:])|(?<=[,(–—|:] ))([A-Z][A-Za-z]?[A-Z]{1,5}(?:[- ][A-Z][a-z]+)?)(?=\s*(?:[,)–—|]|$))", ln):
                tok = m.group(1)
                if len(tok) < 3 or tok.upper() in ("BS", "MS", "BBA", "MPH", "SELECTED", "IFRS", "GAAP", "HE", "SAS", "STEM", "FIRST", "HONS", "ERASMUS", "EXCHANGE") or tok.upper() in ("MSC", "BSC", "MBA", "PHD", "GPA", "BA", "MA", "BENG", "MENG", "LLB", "LLM", "MD", "CV", "IT", "AI", "ML", "UK", "USA", "EU", "HR", "CS", "EE", "ME", "IB", "GCSE", "ECTS", "CFA", "CPA", "PMP", "ACCA", "HND", "BTEC", "NVQ", "DI", "DEA", "DESS", "BTS", "DUT", "BAC", "ABI", "MATURA", "HBO", "WO", "VWO", "HAVO", "MBO", "SQL", "R", "C", "II", "III", "IV") or is_tech(tok):
                    continue
                out.append(S(s_ + m.start(1), s_ + m.end(1), "EDU_ORG", "re-edu-acr"))
    # self-introduction in prose ("My name is X", multilingual)
    NAMETOK = r"[A-ZÀ-ÝŁŚŻ][\w'’-]+"
    for m in re.finditer(rf"(?:\b[Mm]y name is|\bI am|\bI'm|\bMein Name ist|\bIch bin|\bJe m'appelle|\bJe suis|\bMe llamo|\bMi nombre es|\bSoy|\bMijn naam is|\bIk ben|\bJag heter|\bJag är|\bJeg heter|\bJeg hedder|\bNimeni on|\bOlen|\bMi chiamo|\bSono|\bChamo-me|\bMeu nome é|\bNazywam się)\s+({NAMETOK}(?:\s+(?:de |van |von |der |den |da |di |dos |del |al-|el-|bin )?{NAMETOK}){{1,3}})", text):
        if not TITLE_WORDS.search(m.group(1)):
            out.append(S(m.start(1), m.end(1), "NAME", "re-intro"))
    # signature in brackets at line end: "(Jane Doe)"
    for m in re.finditer(r"(?m)(?:^|\t|\s{3,})\(((?:[A-Z]\.\s?){0,4}[A-ZÀ-Ý][A-Za-zà-ÿ'’-]+(?:\s+[A-ZÀ-Ý][A-Za-zà-ÿ'’-]+){0,3})\)\s*$", text):
        if not TITLE_WORDS.search(m.group(1)) and len(m.group(1).split()) >= 2:
            out.append(S(m.start(1), m.end(1), "NAME", "re-signature"))
    # supervisor / referee in prose
    for m in re.finditer(rf"(?i:supervised by|supervisor|under the supervision of|reporting to|reported to|betreut von|sous la direction de|dirigid[oa] por|tutor|advisor|adviser|promotor|handledare|ohjaaja|relatore)[:\s]+(?:(?:Dr|Prof|Mr|Mrs|Ms|Dr\. ?med)\.?\s+)*({NAMETOK}(?:\s+{NAMETOK}){{1,3}})", text):
        out.append(S(m.start(1), m.end(1), "NAME", "re-supervisor"))
    for s, e, ln, sec in sections:
        # referees: a name at the start of each references line
        if sec == "references":
            m = re.match(rf"[\s•*\-–]*(?:(?:Dr|Prof|Mr|Mrs|Ms|Mx|Herr|Frau|M|Mme|Sr|Sra|Dhr|Mevr|Sig|Sig\.ra)\.?\s+)*({NAMETOK}(?:\s+(?:de |van |von |der |den |da |di )?{NAMETOK}){{1,3}})\s*(?:[,–—|(-]|$)", ln)
            if m and not TITLE_WORDS.search(m.group(1)) and not is_section_title(ln):
                out.append(S(s + m.start(1), s + m.end(1), "NAME", "re-referee"))
        # citation lines: publications section, or any line with >= 2 "Surname, X." patterns
        cite_line = sec == "publications" or len(re.findall(r"[A-ZÀ-Ý][a-zà-ÿ'’]+(?:-[A-ZÀ-Ý][a-zà-ÿ'’]+)?,? (?:[A-Z]\.?){1,3}(?:[,;.]|$)", ln)) >= 2
        if cite_line:
            # "J. Smith"
            for m in re.finditer(r"(?<![\w.])((?:[A-ZÀ-Ý]\.\s?){1,3}\s?[A-ZÀ-Ý][a-zà-ÿ'’\-]+(?:-[A-ZÀ-Ý][a-zà-ÿ'’]+)?)(?=[,;:\s(&]|$)", ln):
                out.append(S(s + m.start(1), s + m.end(1), "NAME", "re-cite"))
            # "Smith, J." / "van Dijk, J.-P."
            for m in re.finditer(r"\b([A-ZÀ-Ý][a-zà-ÿ'’]+(?:-[A-ZÀ-Ý][a-zà-ÿ'’]+)?(?: [a-z]{2,3})?,? (?:[A-Z]\.(?:-[A-Z]\.)?\s?){1,3})(?=[,;\s(&]|$)", ln):
                out.append(S(s + m.start(1), s + m.end(1), "NAME", "re-cite"))
            # "Smith JR" (Vancouver style)
            for m in re.finditer(r"(?:^|(?<=[,;] )|(?<=\. ))([A-ZÀ-Ý][a-zà-ÿ'’]+(?:-[A-ZÀ-Ý][a-zà-ÿ'’]+)? [A-Z]{1,3})(?=[,;.]|\s*\(|$)", ln):
                if not is_tech(m.group(1).split()[0]):
                    out.append(S(s + m.start(1), s + m.end(1), "NAME", "re-cite"))
    return out

# ---------------------------------------------------------------- policy layer

def is_tech(tok):
    """True if the token (ignoring surrounding .,()) is in the TECH allowlist.

    Example:
      tok      = "Python,"
      returns  = True
      tok      = "Leeds"
      returns  = False
    """
    return tok.lower().strip(".,()") in TECH


def span_text(text, sp):
    """Return the substring of text covered by span sp.

    Example:
      text     = "John Smith\\njohn@x.com | Leeds, UK"
      sp       = S(11, 21, "EMAIL", "re")
      returns  = "john@x.com"
    """
    return text[sp["start"]:sp["end"]]


def looks_like_list(line):
    """True if the line looks like a list (>= 3 commas, >= 2 bullets or >= 3 pipes).

    Example:
      line     = "Python, Java, SQL, Docker"
      returns  = True
      line     = "Leeds, UK"
      returns  = False
    """
    return line.count(",") >= 3 or line.count("•") >= 2 or line.count("|") >= 3


def is_location_line(text_seg):
    """True if the segment mentions a country, or starts with a known city and is short (<= 6 words).

    Example:
      text_seg = "Leeds, UK"
      returns  = True                           # "uk" is in COUNTRIES
      text_seg = "Leeds"
      returns  = False                          # city lookup only; cities() is empty without geonamescache
      text_seg = "Acme Ltd"
      returns  = False
    """
    low = text_seg.lower()
    # any country name anywhere in the segment
    if any(re.search(rf"(?<!\w){re.escape(c)}(?!\w)", low) for c in COUNTRIES):
        return True
    toks = re.findall(r"[A-ZÀ-ÝŁŚŻ][\w'’-]+", text_seg)
    # or: a short segment whose first capitalised word is a known city
    return bool(toks) and toks[0] in cities() and len(text_seg.split()) <= 6


EPONYM_RE = re.compile(r"(?i)\b(?:marie sk[łl]odowska-curie|sk[łl]odowska-curie|fulbright|rhodes scholar|chevening|erasmus\+?|nobel|turing award|humboldt|max planck|fraunhofer|carl zeiss|leibniz|marshall scholar|gates cambridge)\b")
DEGREE_RE = re.compile(r"(?i)\b(?:b\.?sc|m\.?sc|b\.?s|m\.?s|mba|b\.?a|m\.?a|ph\.?d|beng|meng|ieng|ceng|hons|degree|diploma|diplom|diplôme|dipl|master|máster|mestrado|laurea|licence|licenciatura|grado|bachelor|bachelier|baccalauréat|abitur|matura|vwo|havo|civilingenjör|ingenjör|engineering|ingeniería|ingegneria|ingénieur|ingénierie|science|sciences|ciência|ciencias|scienze|studies|studien|economics|management|gestão|gestión|computer|informatik|informatique|informática|mathematics|physics|chemistry|biology|law|medicine|energy|research|foundation|level|certificate|certification|conference|summit|course|kurs|programme|program)\b")
COMPANY_RE = re.compile(r"(?i)(?:\b(?:s\.a\.|s\.p\.a\.?|a/s|asa|gmbh|ag|ltd|limited|inc|plc|llc|llp|oy|oyj|ab|as|bv|b\.v\.|nv|sarl|sas|srl|s\.l\.|sp\. z o\.o\.|group|holding|holdings|technologies|solutions|consulting|bank)\b)")
POSTCODE_RE = re.compile(r"\b(?:(?<![\d.,])\d{5}(?=\s+[A-ZÀ-Ý])|\d{4}\s?[A-Z]{2}\b|\d{3}\s\d{2}(?=\s+[A-ZÀ-Ý])|\d{2}-\d{3}(?=\s+[A-ZÀ-ÝŁŚŻ])|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}|[A-Z]\d[A-Z]\s?\d[A-Z]\d|[A-Z]{2}\s\d{5}|(?!19|20)\d{4}(?=\s+[A-ZÀ-Ý][a-zà-ÿ]))\b")
STREET_RE = re.compile(r"(?i)\b(?:street|st\.|road|rd\.|lane|avenue|ave\.|grove|close|drive|way|place|square|crescent|terrace|flat|apt\.?|apartment|suite|"
                       r"rue|boulevard|bd\.?|chemin|allée|impasse|straße|strasse|str\.|weg|platz|gasse|allee|ring|damm|calle|c/|avenida|avda\.?|plaza|paseo|"
                       r"straat|gracht|laan|plein|kade|gatan|vägen|gränd|gade|vej|gate|veien|katu|tie|kuja|polku|via|viale|piazza|corso|largo|ulica|ul\.|"
                       r"aleja|al\.|rua|travessa|heol|lgh|nagar|d\.no\.?|door no\.?|h\.no\.?)\b|\w+(?:straße|strasse|straat|gatan|vägen|gade|vej|veien|katu|tie|gata)\b|\w+str\.(?=\s*\d)")


def policy(text, spans, sections, cfg):
    """Drop spans the policy keeps visible; re-label; expand location lines.

    Example (default cfg):
      text     = "John Smith\\njohn@x.com | Leeds, UK\\n\\nEXPERIENCE\\nEngineer at Acme Ltd, Leeds 2019-2023"
      sections = section_of_lines(text)
               = [(0, 10, "John Smith", "header"), (11, 33, "john@x.com | Leeds, UK", "header"), (34, 34, "", "header"),
                  (35, 45, "EXPERIENCE", "experience"), (46, 83, "Engineer at Acme Ltd, Leeds 2019-2023", "experience")]
      spans    = [S(0, 10, "NAME", "re-headername"),   # "John Smith"  regex -> kept as-is
                  S(11, 21, "EMAIL", "re"),            # "john@x.com"  kept
                  S(24, 29, "LOCATION", "gliner"),     # "Leeds"       model, contact line -> kept
                  S(68, 73, "LOCATION", "gliner"),     # "Leeds"       model, experience line -> dropped (work location)
                  S(58, 62, "NAME", "gliner")]         # "Acme"        model, followed by "Ltd" -> dropped (org suffix)
      returns  = [S(0, 10, "NAME", ...), S(11, 21, "EMAIL", ...), S(24, 29, "LOCATION", "gliner"),
                  S(24, 33, "LOCATION", "re-locline")] # "Leeds, UK" added by location_lines
    """
    keep = []
    for sp in spans:
        if sp["label"] == "NAME":
            # strip leading honorifics from names ("Dr. Amanda Price" -> "Amanda Price"); titles are optional in gold
            mt = re.match(r"(?:(?:Dr|Prof|Mr|Mrs|Ms|Mx|Herr|Frau|Mme|Mlle|Sr|Sra|Dott|Ing|Dipl\.-Ing|Dr\. med|Dr\. rer\. nat)\.?\s+)+", text[sp["start"]:sp["end"]])
            if mt and mt.end() < sp["end"] - sp["start"]:
                sp = dict(sp, start=sp["start"] + mt.end())
        t = span_text(text, sp).strip()
        if not t: continue
        sec, line, ls = section_at(sections, sp["start"])
        # regex spans are trusted; model spans get the checks below
        model = not sp["src"].startswith("re")
        if model and is_section_title(line):
            continue
        lab = sp["label"]
        toks = re.findall(r"[\w'-]+", t)
        # models often tag the Languages section ("Polish", "English") as nationality/location
        if model and lab in ("NAME", "LOCATION", "NATIONALITY", "ORG") and toks and all(x.lower() in LANGUAGE_NAMES for x in toks):
            continue
        if lab == "NAME":
            if not model: keep.append(sp); continue
            # not a name: field labels / section words
            if toks and all(x.lower() in FIELD_LABELS or x.lower() in ALL_SECTION_WORDS for x in toks): continue
            # not a name: tech terms or months ("Django", "May")
            if cfg.get("allowlist", True) and toks and all(is_tech(x) or x.lower() in MONTHS for x in toks):
                if len(toks) == 1 or looks_like_list(line) or sec in ("skills", "experience", "certifications", "other"):
                    continue
            if cfg.get("drop_names_in_skills", True) and sec == "skills" and looks_like_list(line):
                continue
            if cfg.get("org_suffix_rule", True) and re.search(
                    r"(?i)^\s*(?:&|and|und|et|y)\s|^\s*(?:&\s*co\b|group|holdings|partners|consulting|bank|ltd|inc|gmbh|ag\b|ab\b|oy\b|oyj|as\b|a/s|aps|plc|llc|s\.a\.|s\.p\.a|srl|sarl|b\.v\.|nv\b|corp|& sons|associates|law|llp)",
                    text[sp["end"]:sp["end"] + 20]):
                continue
            if cfg.get("org_suffix_rule", True) and re.search(r"(?i)(?:&\s*|\band\s)$", text[max(0, sp["start"] - 4):sp["start"]]):
                continue
            # not a name: single char or a short all-caps token ("AWS", "NHS")
            if len(t) < 2 or (len(toks) == 1 and toks[0].isupper() and len(toks[0]) <= 5):
                continue
            if cfg.get("org_suffix_rule", True) and (COMPANY_RE.search(t) or "&" in t or EPONYM_RE.search(t)):
                continue
            # optional model-confidence floor
            if sp["score"] < cfg.get("name_min_score", 0): continue
            keep.append(sp)
        elif lab in ("LOCATION", "ADDRESS"):
            if not model: keep.append(sp); continue
            # model places only count near contact details or in references; elsewhere they are work locations
            if contact_zone(sec, line) or sec == "references":
                keep.append(sp)
        # employers stay visible; only schools are re-labelled and kept
        elif lab == "ORG":
            if is_edu(t):
                keep.append(dict(sp, label="EDU_ORG"))
            elif sec == "education" and cfg.get("mask_orgs_in_education", True) and not DEGREE_RE.search(t) and not is_tech(t) and not COMPANY_RE.search(t):
                keep.append(dict(sp, label="EDU_ORG"))
        elif lab == "EDU_ORG":
            if not model or is_edu(t) or (sec == "education" and not DEGREE_RE.search(t) and not is_tech(t) and not COMPANY_RE.search(t)
                                          and not re.fullmatch(r"(?i)vwo|havo|mbo|hbo|abitur|matura|gcse|a-levels?|ib|bac", t)):
                keep.append(sp)
        # dates are kept visible unless a DOB key sits just before them
        elif lab == "DATE":
            ctx = text[max(0, sp["start"] - 25):sp["start"]].lower()
            if re.search(rf"(?i)(?:{KV_KEYS['DOB']})\s*[:,]?\s*$", ctx):
                keep.append(dict(sp, label="DOB"))
        elif lab == "NATIONALITY":
            if not model or sec in ("header", "personal") or re.search(rf"(?i)(?:{KV_KEYS['NATIONALITY']})\s*:?\s*$", text[max(0, sp["start"] - 25):sp["start"]]):
                keep.append(sp)
        elif lab == "URL":
            u = t.lower()
            if cfg.get("url_policy", "personal") == "personal":
                label_ctx = line[:sp["start"] - ls].lower()
                # code hosts: profile root is personal; a repo link only if in a contact zone
                gh = re.search(r"(?:github|gitlab|bitbucket)\.(?:com|org)/([^/\s]+)(/[^/\s]+)?", u)
                if gh:
                    if not gh.group(2) or contact_zone(sec, line):
                        keep.append(sp)
                elif any(h in u for h in PERSONAL_HOSTS) or sp["src"] in ("re-bare", "prop") or contact_zone(sec, line) and sec != "other" \
                        or re.search(r"portfolio|website|homepage|personal|web ?site|my blog", label_ctx) and "company" not in label_ctx:
                    keep.append(sp)
            else:
                keep.append(sp)
        elif lab in ("GENDER", "AGE", "RELIGION", "MARITAL", "DOB", "EMAIL", "PHONE", "ID", "HANDLE"):
            if lab == "ID" and model:
                if not re.search(r"\d", t) or sum(c.isdigit() for c in t) < 5: continue
                if YEAR_RANGE_RE.match(t) or re.fullmatch(r"[\d.,%$€£]+[kmb]?", t.lower()): continue
            if lab == "PHONE" and model:
                if sum(c.isdigit() for c in t) < 7: continue
            if lab == "DOB" and model and not re.search(r"\d", t): continue
            if lab == "DOB" and model and re.fullmatch(r"(?:19|20)\d{2}", t) and not re.search(rf"(?i)(?:{KV_KEYS['DOB']})\s*[:,]?\s*$", text[max(0, sp["start"] - 25):sp["start"]]): continue
            if lab == "HANDLE" and model:
                if " " in t or "\n" in t or "/" in t or is_tech(t) or not (t.startswith("@") or contact_zone(sec, line) and sec != "other" or re.search(rf"(?i)(?:^|\W)(?:{HANDLE_KEYS})\s*[:|]", line)):
                    continue
            if lab == "DOB" and model and not (sec in ("header", "personal") or re.search(rf"(?i)(?:{KV_KEYS['DOB']}|{KV_KEYS['AGE']})\s*[:,]?\s*(?:on\s+|le\s+|el\s+|il\s+)?$", text[max(0, sp["start"] - 30):sp["start"]])):
                continue
            if lab == "DOB" and model and YEARISH.search(t):
                continue
            if lab == "MARITAL" and model and not (t.lower() in MARITAL_WORDS or re.search(rf"(?i)(?:{KV_KEYS['MARITAL']})\s*[:|]?\s*$", text[max(0, sp["start"] - 30):sp["start"]])):
                continue
            keep.append(sp)
    # add whole-segment place/address spans, then catch-all segments on contact lines
    if cfg.get("location_lines", True):
        keep += location_lines(text, sections, keep)
    if cfg.get("contact_line_rule", True):
        keep += contact_segments(text, sections, keep)
    return keep


CV_TITLE_RE = re.compile(r"(?i)(?:europass\s+)?(?:curriculum vitae|curriculum|currículum vitae|currículum|currículo|cv|resume|résumé|lebenslauf|ansioluettelo|"
                         r"meritförteckning|levnadsbeskrivning|życiorys|curriculum vitæ|hoja de vida)(?:\s*[-–:]\s*.*)?")


def _name_tok_ok(x, particles):
    """True if a token can be part of a personal name (capitalised, a name particle, or caseless script).

    Example:
      particles = ("de", "van")
      x         = "Smith"  -> returns True
      x         = "van"    -> returns True       # particle
      x         = "smith"  -> returns False
    """
    x = x.strip("(),")
    if not x: return True
    if x.lower() in particles: return True
    c = x[0]
    # caseless scripts (CJK, Arabic, ...) pass; so do capitalised Latin letters
    return c.isupper() or (c.isalpha() and not c.isascii() and c.lower() == c.upper()) or c in "ÀÁÂÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÖØÙÚÛÜÝŁŚŻŽŠČ"


def header_name(text, sections):
    """First lines of the CV that look like a personal name (before content sections).

    Example:
      text     = "John Smith\\njohn@x.com | Leeds, UK\\n\\nEXPERIENCE\\nEngineer at Acme Ltd, Leeds 2019-2023"
      sections = section_of_lines(text)
      returns  = [S(0, 10, "NAME", "re-headername")]   # "John Smith"
      text     = "CURRICULUM VITAE\\nSenior Data Engineer\\njohn@x.com"
      returns  = []                                    # CV title skipped, job title on line 2 -> give up
    """
    seen = 0
    particles = ("de", "van", "der", "den", "da", "di", "von", "bin", "binti", "al", "el", "la", "le", "dos", "das", "del", "du", "ter", "zu", "af", "y", "e")
    for s, e, ln, sec in sections:
        # only look at the first 5 non-empty lines before any content section
        if sec in ("experience", "education", "skills", "publications", "references", "certifications") or seen >= 5: break
        t = ln.strip().strip("#*_ ")
        if not t: continue
        seen += 1
        # skip headings and "Curriculum Vitae"-style titles
        if is_section_title(ln) or CV_TITLE_RE.fullmatch(t):
            continue
        # skip contact / key-value lines
        if re.search(r"\d|@|https?:|\|", t) or ":" in t: continue
        seg = t
        # keep "Smith, John" whole; otherwise take the part before the first separator
        if not re.fullmatch(r"[^,]+,\s*[^,]+", t):
            seg = re.split(r"\s{3,}|\s[–—|·•]\s|,\s", t)[0]
        # drop credentials ("Jane Doe, MSc")
        seg = re.sub(r",?\s*\b(?:MSc|MBA|PhD|Ph\.D\.|BSc|MD|CPA|CFA|PE|PMP|CISSP|ACCA|CIMA|Dr\.?|Prof\.?|M\.Sc\.|Dipl\.-Ing\.|Ing\.|RN|MRICS|FCA|CEng)\b\.?", "", seg).strip(" ,")
        # a job-title line: allowed as line 1 (title above name), otherwise stop
        if TITLE_WORDS.search(seg):
            if seen >= 2: return []
            continue
        toks = seg.split()
        # 2-7 capitalised tokens (or name particles) that are not all labels
        if 2 <= len(toks) <= 7 and all(_name_tok_ok(x, particles) for x in toks) \
                and not all(x.lower().strip(",") in ALL_SECTION_WORDS or x.lower() in FIELD_LABELS for x in toks):
            i = ln.find(seg)
            return [S(s + i, s + i + len(seg), "NAME", "re-headername")]
        if seen >= 2: return []
    return []


def contact_segments(text, sections, spans):
    """In a contact line (has email/phone), any short capitalised segment not already explained is a name or place.

    Example:
      text     = "Jane Doe\\njane@x.com | Leeds | +44 7700 900123"
      sections = section_of_lines(text)
      spans    = [S(9, 19, "EMAIL", "re"), S(30, 45, "PHONE", "re")]   # already explained -> skipped
      returns  = [S(22, 27, "LOCATION", "re-contactseg")]             # "Leeds"
                 # "Jane Doe" line is not a contact line -> ignored
    """
    out = []
    for s, e, ln, sec in sections:
        if not is_contact_line(ln): continue
        for m in re.finditer(r"[^|\n\t]+", ln):
            # split on pipes/tabs, then on spaced dashes, slashes, bullets and wide gaps
            for p in re.split(r"\s+[-–—/•·]\s+|\s{3,}", m.group()):
                txt = p.strip(" ,;")
                if not txt: continue
                i = ln.find(txt, m.start()); a, b = s + i, s + i + len(txt)
                # already covered by another span
                if any(sp["start"] < b and sp["end"] > a for sp in spans): continue
                # skip numbers, emails, key-values and job titles
                if re.search(r"\d|@|:", txt) or TITLE_WORDS.search(txt): continue
                toks = txt.split()
                if all(x.lower().strip(".") in FIELD_LABELS or x.lower() in ALL_SECTION_WORDS for x in toks): continue
                if any(x.lower() in FIELD_LABELS for x in toks): continue
                # 1-4 capitalised words, not all tech terms
                if 1 <= len(toks) <= 4 and all(x[0].isupper() for x in toks) and not all(is_tech(x) for x in toks):
                    out.append(S(a, b, "LOCATION", "re-contactseg"))
    return out


def location_lines(text, sections, spans):
    """Header/contact segments that are a place (gazetteer or postcode or street) are masked whole.

    Example:
      text     = "Jane Doe\\n12 Park Road, Leeds LS1 4AB | jane@x.com\\nLeeds, UK"
      sections = section_of_lines(text)
      spans    = []
      returns  = [S(9, 36, "ADDRESS", "re-street"),     # "12 Park Road, Leeds LS1 4AB"  street + number
                  S(50, 59, "LOCATION", "re-locline")]  # "Leeds, UK"  country match
                  # "Jane Doe" not a place; "jane@x.com" segment skipped (email)
    """
    out = []
    for s, e, ln, sec in sections:
        if not contact_zone(sec, ln):
            continue
        for seg in re.finditer(r"[^|\n\t•·]+", ln):
            parts = re.split(r"\s+[-–—]\s+|\s+/\s+|\s{3,}", seg.group())
            off = seg.start()
            for p in parts:
                i = ln.find(p, off); off = i + len(p)
                txt = p.strip(" ,;")
                if not txt or EMAIL_RE.search(txt): continue
                if ":" in txt:
                    # a key-value segment only counts if the key is an address key; keep just the value
                    key, _, val = txt.partition(":")
                    if not re.search(rf"(?i)^\s*(?:{KV_KEYS['ADDRESS']})\s*$", key): continue
                    txt = val.strip(); i = ln.find(txt, i)
                    if not txt: continue
                if re.search(r"https?://|www\.", txt) or PHONE_RE.fullmatch(txt): continue
                # not a place: company names, or a job-title line
                if COMPANY_RE.search(txt) or (TITLE_WORDS.search(ln) and not is_contact_line(ln)): continue
                a = s + ln.find(txt, i); b = a + len(txt)
                wt = re.findall(r"[^\W\d_][\w'’-]*", txt)
                # street word + a number, or a postcode with a capitalised word -> ADDRESS
                has_street = STREET_RE.search(txt) and re.search(r"\d", txt)
                has_pc = POSTCODE_RE.search(txt) and wt and any(w[0].isupper() for w in wt) and len(txt.split()) <= 12
                if has_street or has_pc and not re.fullmatch(r"[\d\s+().-]+", txt):
                    out.append(S(a, b, "ADDRESS", "re-street")); continue
                # mostly lower-case -> prose, not a place
                if wt and sum(1 for w in wt if w[0].isupper()) / len(wt) < 0.6: continue
                # language lists ("English, Polish")
                if wt and sum(1 for w in wt if w.lower() in LANGUAGE_NAMES) >= 1 and len(wt) > 1: continue
                # country/city segment, or widen an existing place span to its whole segment
                covered_loc = any(sp["label"] in ("LOCATION", "ADDRESS") and sp["start"] < b and sp["end"] > a for sp in spans)
                if (is_location_line(txt) and len(txt.split()) <= 8) or (covered_loc and len(txt.split()) <= 7):
                    out.append(S(a, b, "LOCATION", "re-locline"))
    return out

# ---------------------------------------------------------------- identity propagation

def propagate(text, spans, sections, cfg):
    """Re-find detected names elsewhere: full name (and UPPERCASE), surname tokens, and name-bearing URLs.

    Example (default cfg, propagate="surname"):
      text     = "John Smith\\njohn@x.com\\n\\nEXPERIENCE\\nMr Smith led the team. See johnsmith.dev"
      sections = section_of_lines(text)
      spans    = [S(0, 10, "NAME", "re-headername")]
      returns  = [S(0, 10, "NAME", "prop"),     # "John Smith"   full name
                  S(5, 10, "NAME", "prop"),     # "Smith"        surname
                  S(37, 42, "NAME", "prop"),    # "Smith"        in "Mr Smith" (first name not propagated)
                  S(61, 74, "URL", "prop")]     # "johnsmith.dev" contains the name
      cfg      = {"propagate": "none"} -> returns []
    """
    mode = cfg.get("propagate", "surname")
    if mode == "none": return []
    # seed names: regex names anywhere, model names only from header/footer/personal/references
    names = []
    for sp in spans:
        if sp["label"] == "NAME":
            sec, _, _ = section_at(sections, sp["start"])
            t = span_text(text, sp)
            if sp["src"] == "re-spaced":
                # "R O B E R T" -> "ROBERT"
                t = re.sub(r"(?<=\S) (?=\S)", "", t)
            if sp["src"].startswith("re") or sec in ("header", "footer", "personal", "references"):
                names.append((t, sec, sp["src"]))
    cand_tokens, full = set(), set()
    for t, sec, src in names:
        toks = [x for x in re.findall(r"[\w'’-]+", t) if len(x) > 1]
        # drop short all-caps tokens (credentials like "MSc", "RN") from 3+ token names
        toks = [x for x in toks if not (x.isupper() and len(x) <= 5 and not src == "re-spaced")] if len(toks) > 2 else toks
        if len(toks) >= 2:
            full.add(" ".join(toks))
            last = toks[-1]
            # a token is safe to propagate if it is not a tech term, month, common word or label
            ok = lambda x: not is_tech(x) and x.lower() not in MONTHS | COMMON_WORDS | FIELD_LABELS and x.lower() not in ALL_SECTION_WORDS and len(x) > 2
            # surname (and parts of a double-barrelled surname)
            if mode in ("surname", "all") and ok(last):
                cand_tokens.add(last)
                if "-" in last: cand_tokens.update(p for p in last.split("-") if ok(p) and len(p) > 3)
            # "all" mode also propagates given names, minus particles
            if mode == "all":
                for x in toks[:-1]:
                    if ok(x) and x.lower() not in ("de", "van", "von", "der", "den", "da", "di", "del", "dos", "bin", "al", "el"): cand_tokens.add(x)
    out = []
    for f in full:
        # full name, as written and in UPPERCASE
        for m in re.finditer(re.escape(f), text): out.append(S(m.start(), m.end(), "NAME", "prop"))
        up = f.upper()
        for m in re.finditer(re.escape(up), text): out.append(S(m.start(), m.end(), "NAME", "prop"))
    # single tokens as whole words, not inside emails or domains
    for tok in cand_tokens:
        for m in re.finditer(rf"(?<![\w@.]){re.escape(tok)}(?![\w])", text):
            out.append(S(m.start(), m.end(), "NAME", "prop"))
    # name-bearing URLs/domains (e.g. https://janedoe.dev)
    for m in list(URL_RE.finditer(text)) + list(BARE_DOMAIN_RE.finditer(text)):
        # compare with separators removed: "jane-doe.dev" -> "janedoedev"
        u = m.group().lower().replace("-", "").replace(".", "").replace("_", "")
        for f in full:
            parts = [p.lower() for p in re.findall(r"\w+", f)]
            key = "".join(parts)
            if key and (key in u or (len(parts) >= 2 and len(parts[-1]) > 3 and parts[-1] in u)):
                out.append(S(m.start(), m.end(), "URL", "prop"))
    return out

# ---------------------------------------------------------------- ensemble helpers

def overlap(a, b):
    """True if spans a and b share at least one character (end is exclusive).

    Example:
      a, b     = S(0, 5, "NAME", "a"), S(4, 8, "NAME", "b")
      returns  = True
      a, b     = S(0, 5, "NAME", "a"), S(5, 8, "NAME", "b")
      returns  = False                          # touching, not overlapping
    """
    return a["start"] < b["end"] and b["start"] < a["end"]


def vote(model_spans_by_detector, k):
    """Keep a model span if >= k detectors produced an overlapping span.

    Example:
      model_spans_by_detector = {"gliner": [S(0, 10, "NAME", "gliner"), S(40, 45, "LOCATION", "gliner")],
                                 "spacy":  [S(0, 4, "NAME", "spacy")]}
      k        = 2
      returns  = [S(0, 10, "NAME", "gliner"), S(0, 4, "NAME", "spacy")]   # (40, 45) seen by gliner only -> dropped
      k        = 1 -> returns all 3 spans, no voting
    """
    if k <= 1:
        return [s for spans in model_spans_by_detector.values() for s in spans]
    out = []
    dets = list(model_spans_by_detector.items())
    for name, spans in dets:
        for sp in spans:
            # count this detector plus every other detector with an overlapping span
            n = 1 + sum(1 for other, os in dets if other != name and any(overlap(sp, o) for o in os))
            if n >= k: out.append(sp)
    return out


def run_pipeline(text, detectors, cfg, adjudicator=None):
    """detectors: dict name -> callable(text) -> spans. cfg: parameters. Returns final spans and a trace.

    Example (default cfg):
      text      = "John Smith\\njohn@x.com | Leeds, UK\\n\\nEXPERIENCE\\nEngineer at Acme Ltd, Leeds 2019-2023"
      detectors = {"gliner": lambda t: [S(0, 10, "NAME", "gliner"), S(68, 73, "LOCATION", "gliner")]}
      cfg       = {}
      returns   = ([S(0, 10, "NAME", "gliner"), S(11, 21, "EMAIL", "re"), S(0, 10, "NAME", "re-headername"),
                    S(24, 33, "LOCATION", "re-locline"),               # "Leeds, UK"  (work-location "Leeds" dropped)
                    S(0, 10, "NAME", "prop"), S(5, 10, "NAME", "prop"),   # propagate: full name + "Smith"
                    S(0, 10, "NAME", "net"), S(11, 21, "EMAIL", "net"), S(5, 10, "NAME", "net")],   # safety_net
                   {"gliner": 2})                                       # trace: raw span count per detector
    """
    sections = section_of_lines(text)
    trace = {}
    model_spans = {}
    # 1. model detectors, then voting
    for name, fn in detectors.items():
        model_spans[name] = fn(text)
        trace[name] = len(model_spans[name])
    cand = vote(model_spans, cfg.get("vote_k", 1))
    # 2. rule-based detectors
    if cfg.get("regex", True):
        cand += regex_detect(text, sections, cfg)
        if cfg.get("header_name_rule", True):
            cand += header_name(text, sections)
    # raw mode: candidates only, for measuring detector recall
    if cfg.get("raw", False):
        return cand, trace
    # 3. optional re-judging of candidates (e.g. LLM)
    if adjudicator is not None:
        cand = adjudicator(text, cand, sections, cfg)
    # 4. policy, 5. propagation, 6. safety net
    kept = policy(text, cand, sections, cfg) if cfg.get("policy", True) else cand
    kept += propagate(text, kept, sections, cfg)
    if cfg.get("safety_net", True):
        kept += safety_net(text, kept)
    return kept, trace


def safety_net(text, spans):
    """Deterministic final pass: every high-risk string masked anywhere is masked everywhere (exact match).

    Example:
      text     = "John Smith\\njohn@x.com\\nContact john@x.com or JOHN"
      spans    = [S(11, 21, "EMAIL", "re"), S(0, 4, "NAME", "re")]
      returns  = [S(11, 21, "EMAIL", "net"), S(30, 40, "EMAIL", "net"),   # second "john@x.com" now masked
                  S(0, 4, "NAME", "net")]                                # "John" only; match is case-sensitive, "JOHN" missed
    """
    out, seen = [], set()
    for sp in spans:
        if sp["label"] in ("EMAIL", "PHONE", "ID", "URL", "HANDLE", "NAME", "EDU_ORG"):
            t = span_text(text, sp).strip()
            # skip very short strings (too many accidental matches), except 3-letter school acronyms
            if (len(t) >= 4 or (sp["label"] == "EDU_ORG" and len(t) >= 3 and t.isupper())) and t not in seen:
                seen.add(t)
                for m in re.finditer(re.escape(t), text):
                    out.append(S(m.start(), m.end(), sp["label"], "net"))
    return out
