"""Synthetic multilingual CV generator with exact gold spans (dev/tuning set).
The 20 user-supplied CVs are never used for tuning; this set is."""
import json, random, sys
from pathlib import Path

FIRST = {
 "en": ["James","Olivia","Liam","Emma","Noah","Sophie","Oliver","Grace","Harry","Chloe","Jordan","Madison","Sydney","April","Summer","Julia","Hudson","Chase"],
 "de": ["Lukas","Anna","Maximilian","Lea","Jonas","Hannah","Felix","Katharina","Matthias","Sabine"],
 "fr": ["Louis","Camille","Hugo","Manon","Jules","Chloé","Théo","Élodie","Antoine","Margaux"],
 "es": ["Alejandro","Lucía","Pablo","Carmen","Javier","Sofía","Diego","Marta","Álvaro","Paula"],
 "nl": ["Daan","Sanne","Bram","Lotte","Sem","Femke","Joost","Anouk","Pieter","Eva"],
 "sv": ["Erik","Astrid","Oskar","Ingrid","Lars","Maja","Johan","Linnea","Nils","Freja"],
 "fi": ["Mikko","Aino","Juha","Siiri","Ville","Eveliina","Antti","Kaisa","Tuomas","Johanna"],
 "intl": ["Aarav","Priyanka","Wei","Mei","Thi Lan","Kwame","Chidinma","Mehmet","Ayşe","Yuki","Min-jun","Oluwaseun","Anh","Rania","Omar","Siddharth","Xiaoling","Bartosz","Agnieszka","Giulia","João"],
}
LAST = {
 "en": ["Taylor","Brooks","Jenkins","Hudson","Ruby","Price","Austin","London","Parker","Fletcher","Mason","Hill","Chase","Summers","Webb"],
 "de": ["Müller","Schneider","Fischer","Weber","Becker","Hoffmann","Schäfer","Koch","Richter","Wagner"],
 "fr": ["Martin","Bernard","Dubois","Lefèvre","Moreau","Laurent","Girard","Rousseau","Fontaine","Chevalier"],
 "es": ["García","Fernández","López","Martínez","Sánchez","Romero","Navarro","Torres","Ruiz","Ortega"],
 "nl": ["de Vries","van Dijk","Bakker","Janssen","Visser","Smit","de Jong","van den Berg","Mulder","Bos"],
 "sv": ["Andersson","Johansson","Karlsson","Nilsson","Eriksson","Lindqvist","Berg","Lundgren","Holm","Sjöberg"],
 "fi": ["Virtanen","Korhonen","Mäkinen","Nieminen","Mäkelä","Hämäläinen","Laine","Heikkinen","Koskinen","Järvinen"],
 "intl": ["Patel","Iyer","Zhang","Liu","Nguyen","Mensah","Okafor","Yılmaz","Demir","Tanaka","Kim","Adebayo","Tran","Haddad","Farouk","Raghavan","Chen","Kowalski","Wiśniewska","Romano","Silva"],
}
CITY = {  # (home city, region/postcode style, country) per language
 "en": [("Manchester","M1 4BT","United Kingdom"),("Bristol","BS1 5TR","United Kingdom"),("Cork","T12 X70A","Ireland"),("Leeds","LS1 3AB","United Kingdom")],
 "de": [("München","80331","Deutschland"),("Hamburg","20095","Deutschland"),("Köln","50667","Deutschland")],
 "fr": [("Lyon","69002","France"),("Bordeaux","33000","France"),("Nantes","44000","France")],
 "es": [("Valencia","46001","España"),("Sevilla","41001","España"),("Bilbao","48001","España")],
 "nl": [("Utrecht","3511 AB","Nederland"),("Rotterdam","3011 AD","Nederland"),("Eindhoven","5611 AZ","Nederland")],
 "sv": [("Göteborg","411 03","Sverige"),("Malmö","211 20","Sverige"),("Uppsala","753 10","Sverige")],
 "fi": [("Tampere","33100","Suomi"),("Espoo","02150","Suomi"),("Oulu","90100","Suomi")],
}
STREET = {"en": ["{n} Oak Lane","{n} Station Road","Flat {n}, {m} Victoria Street"], "de": ["Lindenstraße {n}","Hauptstraße {n}a"],
          "fr": ["{n} rue de la République","{n} avenue Victor Hugo"], "es": ["Calle de Alcalá {n}","Avenida del Puerto {n}, {m}º"],
          "nl": ["Kerkstraat {n}","Prinsengracht {n}"], "sv": ["Storgatan {n}","Kungsgatan {n}"], "fi": ["Hämeenkatu {n} B {m}","Mannerheimintie {n} A {m}"]}
PHONE = {"en": ["+44 7{a} {b}", "07{a} {b}"], "de": ["+49 151 {c}", "0151 {c}"], "fr": ["+33 6 {d}", "06 {d}"],
         "es": ["+34 6{e}", "6{e}"], "nl": ["+31 6 {f}", "06-{f}"], "sv": ["+46 70 {g}", "070-{g}"], "fi": ["+358 40 {h}", "040 {h}"]}
NAT = {"en": ["British","Irish","Indian","Nigerian"], "de": ["deutsch","österreichisch"], "fr": ["française","belge"],
       "es": ["española","mexicana"], "nl": ["Nederlandse","Belgische"], "sv": ["svensk","finländsk"], "fi": ["suomalainen","ruotsalainen"]}
MARITAL = {"en": ["Married","Single"], "de": ["verheiratet","ledig"], "fr": ["marié","célibataire"], "es": ["casado","soltera"],
           "nl": ["gehuwd","ongehuwd"], "sv": ["gift","ogift"], "fi": ["naimisissa","naimaton"]}
UNI = {"en": ["University of Leeds","Imperial College London","University of Edinburgh","King's College London","University College Cork"],
       "de": ["Technische Universität München","Universität Hamburg","RWTH Aachen"], "fr": ["Université Claude Bernard Lyon 1","Sorbonne Université","École Polytechnique"],
       "es": ["Universidad Politécnica de Valencia","Universidad de Sevilla","Universidad Complutense de Madrid"],
       "nl": ["Universiteit Utrecht","TU Delft","Erasmus Universiteit Rotterdam"], "sv": ["Chalmers tekniska högskola","KTH Royal Institute of Technology","Lunds universitet"],
       "fi": ["Tampereen yliopisto","Aalto-yliopisto","Oulun yliopisto"]}
EMPLOYERS = ["Morgan Stanley","Goldman Sachs","Johnson & Johnson","Procter & Gamble","Ernst & Young","Philips","Siemens","Boston Scientific",
             "Santander","Bank of Ireland","Nokia","Ericsson","Kone","Wärtsilä","Supercell","Zalando","Adyen","Spotify","Klarna","Deutsche Bank",
             "BNP Paribas","Capgemini","Jan de Nul","Wells Fargo","Charles Schwab","Marks & Spencer","Lloyds Banking Group","Phoenix Group",
             "Denver Analytics","Paris Saint-Germain","Aston Martin","Ben & Jerry's","Hugo Boss","Louis Vuitton","Elisa","Fiskars"]
WORK_CITIES = ["Berlin","Paris","Madrid","Amsterdam","Stockholm","Helsinki","Dublin","Lisbon","Vienna","Prague","Warsaw","Milan","Zurich","Copenhagen"]
TOOLS = ["Jenkins","Hudson","Julia","Ruby","Crystal","Elixir","Rails","Phoenix","Django","Flask","Jasmine","Mocha","Karma","Selenium","Kafka",
         "Cassandra","Hadoop","Hive","Presto","Airflow","Luigi","Dagster","Prefect","Ansible","Chef","Puppet","Terraform","Grafana","Kibana",
         "Prometheus","Nagios","Zabbix","Jaeger","Sentry","Heroku","Vercel","Alexa","Watson","Oracle","SAP","Tableau","Looker","Pandas","Keras",
         "Maven","Gradle","Jira","Confluence","Bamboo","Figma","Sketch","Python","Java","Go","Rust","Swift","Kotlin","React","Angular","Vue","Svelte"]
TITLES = ["Software Engineer","Data Scientist","Product Manager","DevOps Engineer","Financial Analyst","Nurse","Quality Manager","UX Designer","Sales Director"]
HDR = {
 "en": dict(exp="EXPERIENCE", edu="EDUCATION", sk="SKILLS", ref="REFERENCES", pers="PERSONAL DETAILS", pub="PUBLICATIONS",
            dob="Date of birth", nat="Nationality", mar="Marital status", addr="Address", tel="Phone", mail="Email",
            job="{t} at {emp} ({y1}-{y2}). Delivered {n} projects with {a} and {b}; also worked on-site in {wc}.",
            deg="MSc, {uni}, {y}.", prose="My name is {name} and I have {n} years of experience.", born="born"),
 "de": dict(exp="BERUFSERFAHRUNG", edu="AUSBILDUNG", sk="KENNTNISSE", ref="REFERENZEN", pers="PERSÖNLICHE DATEN", pub="PUBLIKATIONEN",
            dob="Geburtsdatum", nat="Staatsangehörigkeit", mar="Familienstand", addr="Anschrift", tel="Telefon", mail="E-Mail",
            job="{t} bei {emp} ({y1}-{y2}). Verantwortlich für {n} Projekte mit {a} und {b}; Einsätze in {wc}.",
            deg="M.Sc., {uni}, {y}.", prose="Mein Name ist {name}, ich habe {n} Jahre Berufserfahrung.", born="geb."),
 "fr": dict(exp="EXPÉRIENCE PROFESSIONNELLE", edu="FORMATION", sk="COMPÉTENCES", ref="RÉFÉRENCES", pers="INFORMATIONS PERSONNELLES", pub="PUBLICATIONS",
            dob="Date de naissance", nat="Nationalité", mar="Situation familiale", addr="Adresse", tel="Téléphone", mail="Courriel",
            job="{t} chez {emp} ({y1}-{y2}). Pilotage de {n} projets avec {a} et {b} ; missions à {wc}.",
            deg="Master, {uni}, {y}.", prose="Je m'appelle {name} et j'ai {n} ans d'expérience.", born="née le"),
 "es": dict(exp="EXPERIENCIA PROFESIONAL", edu="FORMACIÓN", sk="HABILIDADES", ref="REFERENCIAS", pers="DATOS PERSONALES", pub="PUBLICACIONES",
            dob="Fecha de nacimiento", nat="Nacionalidad", mar="Estado civil", addr="Dirección", tel="Teléfono", mail="Correo",
            job="{t} en {emp} ({y1}-{y2}). Lideré {n} proyectos con {a} y {b}; trabajo presencial en {wc}.",
            deg="Máster, {uni}, {y}.", prose="Me llamo {name} y tengo {n} años de experiencia.", born="nacida el"),
 "nl": dict(exp="WERKERVARING", edu="OPLEIDING", sk="VAARDIGHEDEN", ref="REFERENTIES", pers="PERSOONLIJKE GEGEVENS", pub="PUBLICATIES",
            dob="Geboortedatum", nat="Nationaliteit", mar="Burgerlijke staat", addr="Adres", tel="Telefoon", mail="E-mail",
            job="{t} bij {emp} ({y1}-{y2}). Verantwoordelijk voor {n} projecten met {a} en {b}; gedetacheerd in {wc}.",
            deg="MSc, {uni}, {y}.", prose="Mijn naam is {name} en ik heb {n} jaar ervaring.", born="geboren"),
 "sv": dict(exp="ARBETSLIVSERFARENHET", edu="UTBILDNING", sk="KOMPETENSER", ref="REFERENSER", pers="PERSONUPPGIFTER", pub="PUBLIKATIONER",
            dob="Födelsedatum", nat="Medborgarskap", mar="Civilstånd", addr="Adress", tel="Telefon", mail="E-post",
            job="{t} på {emp} ({y1}-{y2}). Ansvarade för {n} projekt med {a} och {b}; uppdrag i {wc}.",
            deg="Civilingenjör, {uni}, {y}.", prose="Jag heter {name} och har {n} års erfarenhet.", born="född"),
 "fi": dict(exp="TYÖKOKEMUS", edu="KOULUTUS", sk="OSAAMINEN", ref="SUOSITTELIJAT", pers="HENKILÖTIEDOT", pub="JULKAISUT",
            dob="Syntymäaika", nat="Kansalaisuus", mar="Siviilisääty", addr="Osoite", tel="Puhelin", mail="Sähköposti",
            job="{t}, {emp} ({y1}-{y2}). Vastasin {n} projektista, teknologiat {a} ja {b}; työskentely myös {wc}ssa.",
            deg="DI, {uni}, {y}.", prose="Nimeni on {name}, ja minulla on {n} vuoden kokemus.", born="s."),
}
ID_FMT = {"en": ("NI Number", lambda r: f"{r.choice('ABCEGHJ')}{r.choice('ABCEGHJ')} {r.randint(10,99)} {r.randint(10,99)} {r.randint(10,99)} {r.choice('ABCD')}"),
          "de": ("Steuer-ID", lambda r: f"{r.randint(10,99)} {r.randint(100,999)} {r.randint(100,999)} {r.randint(100,999)}"),
          "fr": ("N° de sécurité sociale", lambda r: f"{r.choice('12')} {r.randint(60,99)} {r.randint(10,12):02d} {r.randint(10,95)} {r.randint(100,999)} {r.randint(100,999)} {r.randint(10,97)}"),
          "es": ("DNI", lambda r: f"{r.randint(10000000,99999999)}{r.choice('TRWAGMYFPDXBNJZSQVHLCKE')}"),
          "nl": ("BSN", lambda r: f"{r.randint(100000000,999999999)}"),
          "sv": ("Personnummer", lambda r: f"{r.randint(60,99)}{r.randint(1,12):02d}{r.randint(1,28):02d}-{r.randint(1000,9999)}"),
          "fi": ("Henkilötunnus", lambda r: f"{r.randint(1,28):02d}{r.randint(1,12):02d}{r.randint(60,99)}-{r.randint(100,999)}{r.choice('ABCDEFHJKLMNPRSTUVWXY')}")}
IBAN_CC = {"en": "GB", "de": "DE", "fr": "FR", "es": "ES", "nl": "NL", "sv": "SE", "fi": "FI"}
MONTHS = {"en": ["March","July","October"], "de": ["März","Juli","Oktober"], "fr": ["mars","juillet","octobre"], "es": ["marzo","julio","octubre"],
          "nl": ["maart","juli","oktober"], "sv": ["mars","juli","oktober"], "fi": ["maaliskuuta","heinäkuuta","lokakuuta"]}


class B:
    def __init__(self):
        """Start an empty text buffer with no gold spans.

        Example:
          returns  = B() with t = "", g = []
        """
        self.t = ""; self.g = []
    def add(self, s, label=None, req=True):
        """Append s to the text; if labelled, record its gold span. Returns self for chaining.

        Example:
          b        = B().add("Name: ")                # no label -> text only
          s        = "Ana Ruiz"
          label    = "NAME"
          returns  = b with t = "Name: Ana Ruiz",
                     g = [{"start": 6, "end": 14, "label": "NAME", "required": True, "text": "Ana Ruiz"}]
        """
        if label: self.g.append({"start": len(self.t), "end": len(self.t) + len(s), "label": label, "required": req, "text": s})
        self.t += s
        return self


def ascii_fold(s):
    """Strip diacritics, lowercase and remove spaces (for email/URL slugs).

    Example:
      s        = "Ayşe Yılmaz"
      returns  = "ayseyılmaz"                        # dotless "ı" has no NFKD decomposition -> kept
      s        = "van den Berg"
      returns  = "vandenberg"
    """
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower().replace(" ", "")


def phone(r, lang):
    """Random phone number in one of the language's national/international formats.

    Example (seed fixed):
      r        = random.Random(0)
      lang     = "de"
      returns  = "0151 1679215"
      (next call, lang = "fi") -> "+358 40 462 8113"
    """
    f = r.choice(PHONE[lang])
    return f.format(a=f"{r.randint(100,999)}", b=f"{r.randint(100000,999999)}", c=f"{r.randint(1000000,9999999)}",
                    d=f"{r.randint(10,99)} {r.randint(10,99)} {r.randint(10,99)} {r.randint(10,99)}",
                    e=f"{r.randint(10,99)} {r.randint(100,999)} {r.randint(100,999)}", f=f"{r.randint(10000000,99999999)}",
                    g=f"{r.randint(100,999)} {r.randint(10,99)} {r.randint(10,99)}", h=f"{r.randint(100,999)} {r.randint(1000,9999)}")


def make_cv(r, idx):
    """Generate one synthetic CV with exact gold spans (language, layout and fields drawn from r).

    Example (seed fixed):
      r        = random.Random(7)
      idx      = 1
      returns  = {"id": "DEV-001", "lang": "en", "layout": "prose",
                  "text": "DevOps Engineer\\nEmail: summers81(at)gmail(dot)com | Phone: 07171 352353\\nAddress: Flat 29, ...",
                  "gold": [{"start": 23, "end": 49, "label": "EMAIL", "required": True, "text": "summers81(at)gmail(dot)com"},
                           {"start": 59, "end": 71, "label": "PHONE", ...}, ...]}   # 10 gold spans
    """
    lang = r.choices(["en","de","fr","es","nl","sv","fi"], weights=[4,1,1,1,1,1,1])[0]
    H = HDR[lang]
    pool = lang if r.random() < 0.6 else "intl"
    if lang == "en" and r.random() < 0.4: pool = "en"
    fn, ln = r.choice(FIRST[pool]), r.choice(LAST[pool])
    name = f"{fn} {ln}"
    city, pc, country = r.choice(CITY[lang])
    email_user = r.choice([f"{ascii_fold(fn)}.{ascii_fold(ln)}", f"{ascii_fold(fn)[0]}{ascii_fold(ln)}", f"{ascii_fold(ln)}{r.randint(70,99)}"])
    dom = r.choice(["gmail.com","outlook.com","proton.me","mail.fi","web.de","orange.fr","hotmail.es","ziggo.nl","telia.se","icloud.com"])
    b = B()
    layout = r.choices(["header", "spaced", "footer", "prose"], weights=[12, 2, 3, 3])[0]
    title = r.choice(TITLES)
    if layout == "spaced":
        b.add(" ".join(name.upper()).replace("   ", "   "), "NAME").add("\n")
    elif layout in ("header",):
        b.add(name, "NAME").add("\n")
    b.add(f"{title}\n")
    if layout != "footer":
        style = r.random()
        if style < 0.15:
            ob = r.choice(["{u} [at] {d0} [dot] {d1}", "{u}(at){d0}(dot){d1}", "{u} at {d0} dot {d1}"])
            d0, d1 = dom.split(".", 1)
            b.add(f"{H['mail']}: ").add(ob.format(u=email_user.replace('.', ' [dot] ') if '[' in ob else email_user, d0=d0, d1=d1), "EMAIL")
        else:
            b.add(f"{H['mail']}: ").add(f"{email_user}@{dom}", "EMAIL")
        b.add(" | ").add(f"{H['tel']}: ").add(phone(r, lang), "PHONE").add("\n")
        if r.random() < 0.5:
            n, m = r.randint(1, 120), r.randint(1, 30)
            street = r.choice(STREET[lang]).format(n=n, m=m)
            b.add(f"{H['addr']}: ").add(f"{street}, {pc} {city}, {country}", "ADDRESS").add("\n")
        else:
            b.add(f"{city}, {country}", "LOCATION").add("\n")
        if r.random() < 0.5:
            slug = f"{ascii_fold(fn)}-{ascii_fold(ln)}-{r.randint(10,999)}"
            b.add("LinkedIn: ").add(f"https://www.linkedin.com/in/{slug}", "URL").add("\n")
        if r.random() < 0.3:
            b.add("GitHub: ").add(f"github.com/{ascii_fold(fn)[0]}{ascii_fold(ln)}", "URL").add("\n")
    # personal details block (common in DE/NL/FR/FI CVs)
    if r.random() < (0.7 if lang != "en" else 0.3):
        b.add(f"\n{H['pers']}\n")
        mon = r.choice(MONTHS[lang]); d = r.randint(1, 28); y = r.randint(1965, 2001)
        dob = r.choice([f"{d:02d}.{r.randint(1,12):02d}.{y}", f"{d} {mon} {y}"])
        b.add(f"{H['dob']}: ").add(dob, "DOB").add("\n")
        b.add(f"{H['nat']}: ").add(r.choice(NAT[lang]), "NATIONALITY").add("\n")
        if r.random() < 0.6:
            b.add(f"{H['mar']}: ").add(r.choice(MARITAL[lang]), "MARITAL").add("\n")
        if r.random() < 0.4:
            lab, fmt = ID_FMT[lang]; b.add(f"{lab}: ").add(fmt(r), "ID").add("\n")
        if r.random() < 0.2:
            cc = IBAN_CC[lang]; iban = f"{cc}{r.randint(10,99)} " + " ".join(str(r.randint(1000, 9999)) for _ in range(4)) + f" {r.randint(10,99)}"
            b.add("IBAN: ").add(iban, "ID").add("\n")
    if layout == "prose":
        b.add("\n").add(H["prose"].split("{name}")[0]).add(name, "NAME").add(H["prose"].split("{name}")[1].format(n=r.randint(3, 20))).add("\n")
    # experience
    b.add(f"\n{H['exp']}\n")
    for _ in range(r.randint(1, 3)):
        y1 = r.randint(2005, 2020); y2 = y1 + r.randint(1, 5)
        a, c = r.sample(TOOLS, 2)
        wc = r.choice([w for w in WORK_CITIES if w != city])
        b.add(H["job"].format(t=r.choice(TITLES), emp=r.choice(EMPLOYERS), y1=y1, y2=y2, n=r.randint(2, 40), a=a, b=c, wc=wc) + "\n")
    b.add(f"\n{H['edu']}\n").add(H["deg"].split("{uni}")[0]).add(r.choice(UNI[lang]), "EDU_ORG").add(H["deg"].split("{uni}")[1].format(y=r.randint(1995, 2020))).add("\n")
    b.add(f"\n{H['sk']}\n" + ", ".join(r.sample(TOOLS, r.randint(6, 14))) + ".\n")
    if r.random() < 0.25:
        b.add(f"\n{H['pub']}\n")
        co = f"{r.choice(LAST['intl'])} {r.choice('ABCDEFGHJKLMNPRST')}."
        b.add(f"{ln} {fn[0]}.", "NAME").add(", ").add(co, "NAME").add(f" ({r.randint(2010,2024)}) \"Scalable {r.choice(TOOLS)} pipelines\", Journal of Systems {r.randint(10,60)}({r.randint(1,12)}).\n")
    if r.random() < 0.3:
        b.add(f"\n{H['ref']}\n")
        for _ in range(r.randint(1, 2)):
            rl = r.choice(list(LAST.values())); rf = r.choice(list(FIRST.values()))
            rfn, rln = r.choice(rf), r.choice(rl)
            if r.random() < 0.5: b.add(r.choice(["Dr.", "Prof.", "Mr.", "Ms."]), "TITLE", False).add(" ")
            b.add(f"{rfn} {rln}", "NAME").add(f", {r.choice(TITLES)}, ").add(r.choice(EMPLOYERS), "ORG", False).add(". ")
            b.add(f"{ascii_fold(rfn)}.{ascii_fold(rln)}@{r.choice(['corp.com','firma.de','societe.fr','empresa.es'])}", "EMAIL").add(", ")
            b.add(phone(r, lang), "PHONE").add("\n")
    if layout == "footer":
        b.add("\n---\n").add(name, "NAME").add(" - ").add(f"{email_user}@{dom}", "EMAIL").add(" - ").add(phone(r, lang), "PHONE").add(" - ").add(city, "LOCATION").add("\n")
    return {"id": f"DEV-{idx:03d}", "lang": lang, "layout": layout, "text": b.t, "gold": b.g}


def main(n=80, seed=7, out="data/dev_cvs.jsonl"):
    """Generate n CVs with a fixed seed, check every gold span, and write them as JSONL.

    Example (seed fixed):
      n        = 2
      seed     = 7
      out      = "/tmp/dev2.jsonl"
      returns  = None                                # prints "wrote 2 /tmp/dev2.jsonl"; line 1 is {"id": "DEV-001", "lang": "en", ...}
    """
    r = random.Random(seed)
    rows = [make_cv(r, i + 1) for i in range(n)]
    for row in rows:
        for g in row["gold"]:
            assert row["text"][g["start"]:g["end"]] == g["text"]
    Path(out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
    print("wrote", len(rows), out)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3])) if len(sys.argv) > 2 else main()
