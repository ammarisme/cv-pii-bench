"""Generic lexicons used by the CV policy layer. Deliberately NOT derived from the 20 test CVs
(no test-set employer, city or person names appear here)."""

TECH = set("""
python java javascript typescript c c++ c# go golang rust ruby swift kotlin scala elixir erlang haskell julia crystal perl php r matlab dart lua clojure groovy fortran cobol
django flask fastapi rails laravel symfony spring spring-boot boot express nestjs next next.js nuxt react angular vue svelte ember backbone jquery phoenix
node node.js deno bun redux mobx tailwind bootstrap storybook vite webpack babel jest mocha jasmine karma chai cypress playwright selenium puppeteer
postgres postgresql mysql mariadb sqlite oracle mongodb cassandra redis couchdb dynamodb elasticsearch opensearch neo4j snowflake bigquery redshift
kafka rabbitmq pulsar spark hadoop hive pig presto trino flink airflow luigi dagster prefect dbt beam storm
aws azure gcp heroku vercel netlify firebase supabase cloudflare digitalocean openstack
docker kubernetes k8s helm istio linkerd envoy terraform pulumi ansible chef puppet salt vagrant packer consul vault nomad
jenkins hudson bamboo teamcity circleci travis gitlab github bitbucket argo argocd spinnaker
prometheus grafana kibana logstash jaeger zipkin datadog splunk sentry nagios zabbix newrelic opentelemetry dynatrace
pandas numpy scipy keras tensorflow pytorch torch sklearn scikit-learn xgboost lightgbm spacy huggingface
maven gradle ant sbt npm yarn pnpm pip poetry conda
jira confluence trello asana notion slack teams figma sketch zeplin invision miro
tableau looker powerbi power bi qlik excel visio sap salesforce hubspot marketo zendesk servicenow workday
alexa siri cortana watson stripe twilio sendgrid mailchimp shopify magento wordpress drupal realm room
linux unix windows macos ios android bash zsh powershell git svn mercurial
graphql rest grpc soap json xml yaml html css sass less
tdd ddd bdd ci cd ci/cd devops mlops agile scrum kanban safe itil prince2 pmp iso six sigma lean
bpmn uml sql nosql etl elt api sdk crm erp seo sem ppc
wireshark nessus burp suite crowdstrike snort suricata metasploit nmap
epic cerner
""".split())

MONTHS = set("""january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec
januar februar märz mai juni juli oktober dezember janvier février mars avril juin juillet août septembre octobre novembre décembre
enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre
januari februari maart mei augustus oktober
tammikuu helmikuu maaliskuu huhtikuu toukokuu kesäkuu heinäkuu elokuu syyskuu lokakuu marraskuu joulukuu""".split())

COMMON_WORDS = set("""summer winter spring autumn fall rose may june grace hope faith joy chase hunter mason parker price hill
brook brooks lane wood stone young king rich page bell baker cook miller turner ward""".split())

SECTION_WORDS = {
 "experience": ["doświadczenie zawodowe", "doświadczenie", "esperienza professionale", "esperienze professionali", "esperienza lavorativa", "esperienza",
                "experiência profissional", "experiência", "erhvervserfaring", "arbeidserfaring", "yrkeserfaring", "erfaring", "work experience",
                "professional history", "employment history", "berufliche erfahrung", "beruflicher werdegang", "werdegang", "parcours professionnel",
                "trayectoria profesional", "experience", "employment", "career", "work history", "professional experience", "berufserfahrung", "expérience",
                "experiencia", "werkervaring", "arbetslivserfarenhet", "erfarenhet", "työkokemus", "achievements"],
 "education": ["wykształcenie", "istruzione", "istruzione e formazione", "formazione", "formação", "formação académica", "uddannelse", "utdanning",
               "utbildning", "ausbildung und studium", "bildungsweg", "études", "estudios", "opleidingen", "education and training", "education", "ausbildung", "studium", "formation", "formación", "opleiding", "utbildning", "koulutus", "academic"],
 "skills": ["umiejętności", "competenze", "competências", "kompetencer", "kompetanse", "färdigheter", "fähigkeiten", "languages spoken", "language skills",
            "języki", "lingue", "idiomas", "línguas", "sprog", "språk", "kielet", "kielitaito", "technical skills", "core skills", "skills", "tooling", "tools", "tech stack", "stack", "kenntnisse", "compétences", "habilidades", "vaardigheden",
            "kompetenser", "osaaminen", "technologies", "skills summary", "languages", "sprachen", "langues", "idiomas", "teaching"],
 "references": ["references", "referees", "emergency contact", "notfallkontakt", "contacto de emergencia", "contact d'urgence", "referenzen", "références", "referencias", "referenties", "referenser", "suosittelijat"],
 "personal": ["dane osobowe", "informazioni personali", "dati personali", "dados pessoais", "personlige oplysninger", "personalia", "personlig informasjon",
              "personal information", "personal data", "contact", "kontakt", "contacto", "contatti", "yhteystiedot", "about me", "om mig", "über mich", "à propos",
              "personal details", "personal information", "persönliche daten", "informations personnelles", "datos personales",
              "persoonlijke gegevens", "personuppgifter", "henkilötiedot", "curriculum vitae", "profile", "profil", "perfil", "profiel", "summary"],
 "publications": ["publications", "publikationen", "publicaciones", "publicaties", "publikationer", "julkaisut", "grants"],
 "certifications": ["certifications", "certificates", "zertifikate", "certificats", "certificaciones", "certificaten", "certifikat", "sertifikaatit"],
 "links": ["links", "online", "profiles"],
 "other": ["key skills", "core competencies", "core competences", "hobbies", "interests", "hobbies and interests", "freizeit", "interessen", "loisirs",
           "centres d'intérêt", "aficiones", "intereses", "hobby", "interessi", "fritid", "harrastukset", "zainteresowania", "achievements", "awards",
           "projects", "projekte", "projets", "proyectos", "volunteering", "additional information", "sonstiges", "divers", "otros", "muuta", "övrigt", "diverse"],
}

EDU_KEYWORDS = ["university", "universität", "universite", "université", "universidad", "universiteit", "universitet", "università",
                "uniwersytet", "yliopisto", "hochschule", "högskola", "college", "institute of technology", "polytechnic", "politecnico",
                "école", "ecole", "business school", "school of", "academy", "akademie", "technische universität", "sorbonne", "sciences po",
                "gymnasium", "gymnasiet", "gimnazjum", "lycée", "lycee", "liceo", "liceu", "iut ", "institut ", "instituto", "istituto",
                "szkoła", "szkola", "hogeschool", "hochschule", "berufsschule", "kantonsschule", "fachhochschule", "akademia", "politechnika",
                "polytechnique", "universidade", "universitat", "høgskole", "høgskule", "handelshøyskole", "handelshøjskole", "handelshögskolan",
                "gymnasie", "ammattikorkeakoulu", "lukio", "conservatoire", "conservatorio", "kolleg", "high school", "secondary school",
                "grammar school", "sixth form", "escuela", "escola", "colegio", "college"]
EDU_EXCLUDE = ["hospital", "hospitals", "nhs", "trust", "klinikum", "chu ", "hôpital", "ospedale", "sjukhus", "fc ", "f.c.", "club",
               "gmbh", "ltd", "inc", "plc", "department of", "school of informatics", "faculty of"]
EDU_ACRONYMS = ["INSEAD", "RWTH", "KTH", "ETH", "EPFL", "TU", "HEC", "ESADE", "MIT", "LSE", "UCL", "NTNU", "DTU", "IE Business", "LUT", "UCLA",
                "KIT", "HSG", "TUM", "LMU", "UvA", "KU Leuven", "ESSEC", "ESCP", "IESE", "SGH", "UPC", "UPM", "UAM", "UCM", "NHH", "CBS", "SSE", "IIT", "NUS"]

COUNTRIES = set(x.lower() for x in """Afghanistan Albania Algeria Argentina Armenia Australia Austria Azerbaijan Bahrain Bangladesh Belarus Belgium Bolivia Bosnia
Brazil Bulgaria Cambodia Cameroon Canada Chile China Colombia Croatia Cyprus Czechia Denmark Ecuador Egypt Estonia Ethiopia Finland France Georgia Germany
Ghana Greece Hungary Iceland India Indonesia Iran Iraq Ireland Israel Italy Japan Jordan Kazakhstan Kenya Korea Kuwait Latvia Lebanon Lithuania Luxembourg
Malaysia Malta Mexico Moldova Morocco Nepal Netherlands Nigeria Norway Oman Pakistan Peru Philippines Poland Portugal Qatar Romania Russia Rwanda Serbia
Singapore Slovakia Slovenia Spain Sweden Switzerland Syria Taiwan Tanzania Thailand Tunisia Turkey Uganda Ukraine Uruguay USA US UK Venezuela Vietnam Yemen
Zambia Zimbabwe Deutschland Suomi Sverige Nederland España Italia Polska Österreich Schweiz Danmark Norge Belgique België Éire""".split())
COUNTRIES |= {"united kingdom", "united states", "united arab emirates", "sri lanka", "new zealand", "south africa", "saudi arabia",
              "czech republic", "hong kong", "south korea", "the netherlands"}

KV_KEYS = {
 "NAME": r"surname\(s\)(?:\s*/\s*first name\(s\))?|first name\(s\)|nombre y apellidos|apellidos|nom et prénom|prénom|nome e cognome|cognome|nachname|vorname|imię|nazwisko|efternamn|förnamn|etunimi|sukunimi|fornavn|etternavn|efternavn|achternaam|voornaam|full name|name|nimi|nom|nom complet|prénom et nom|nombre|nombre completo|naam|namn|navn|nome|imię i nazwisko|vor- und nachname|surname|first name|last name|given name",
 "DOB": r"year of birth|jahrgang|geburtsjahr|année de naissance|año de nacimiento|anno di nascita|ano de nascimento|rok urodzenia|fødselsår|födelseår|syntymävuosi|date of birth|d\.o\.b\.?|dob|birth ?date|born|geburtsdatum|geb\.|date de naissance|née? le|fecha de nacimiento|fecha nac\.?|nacid[oa] el|geboortedatum|geboren|födelsedatum|född|syntymäaika|syntynyt|data di nascita|nato il|nata il|data de nascimento|data urodzenia|fødselsdato|født|luogo e data di nascita",
 "NATIONALITY": r"nationality|citizenship|staatsangehörigkeit|nationalität|nationalité|nacionalidad|nationaliteit|medborgarskap|nationalitet|statsborgerskap|statsborgerskab|kansalaisuus|nazionalità|cittadinanza|nacionalidade|obywatelstwo|narodowość",
 "MARITAL": r"zivilstand|perhe|family|familie|famille|familia|famiglia|família|rodzina|familj|marital status|civil status|family status|familienstand|situation familiale|état civil|estado civil|burgerlijke staat|civilstånd|civilstand|sivilstatus|sivilstand|siviilisääty|stato civile|stan cywilny|children|kinder|enfants|hijos|kinderen|barn|børn|lapset|figli|filhos|dzieci",
 "GENDER": r"gender|sex|pronouns|geschlecht|sexe|genre|sexo|género|geslacht|kön|kjønn|køn|sukupuoli|sesso|płeć",
 "ADDRESS": r"address|home address|home|residence|location|based in|anschrift|adresse|wohnort|wohnhaft|dirección|domicilio|residencia|adres|woonplaats|adress|bostadsort|hemort|bopæl|bosted|osoite|kotiosoite|asuinpaikka|indirizzo|residenza|morada|endereço|adres zamieszkania|miejsce zamieszkania",
 "AGE": r"age|alter|âge|edad|idade|età|leeftijd|ålder|alder|ikä|wiek",
 "RELIGION": r"religion|religious|konfession|religión",
}

FIELD_LABELS = set("""mobile mobil mob cell cellulare cellular handy phone tel telephone telefon telefono teléfono téléphone tél puhelin puh tlf gsm
email e-mail mail courriel correo sähköposti epost e-post web website homepage site linkedin github gitlab xing twitter skype telegram whatsapp
address adresse contact kontakt contacto contatti profile profil perfil portfolio cv resume""".split())
PHONE_KEYS = r"tel|tél|tlf|phone|mobile|mob|cell|cellulare|cellular|handy|telefon|telefono|teléfono|téléphone|portable|móvil|celular|puhelin|puh|gsm|mobil|telefoon|whatsapp|☎|📞|ph"
HANDLE_KEYS = r"skype|telegram|whatsapp|signal|discord|twitter|x|instagram|mastodon|threads|bluesky|wechat|line|kakao|tiktok|slack|github|gitlab"

MARITAL_WORDS = set("""married single divorced widowed separated engaged partnered cohabiting verheiratet ledig geschieden verwitwet
marié mariée célibataire divorcé divorcée pacsé pacsée veuf veuve casado casada soltero soltera divorciado divorciada viudo viuda
gehuwd ongehuwd gescheiden samenwonend gift ogift skild sambo särbo naimisissa naimaton eronnut avoliitossa leski
sposato sposata celibe nubile divorziato divorziata vedovo vedova solteiro solteira divorciado żonaty zamężna kawaler panna rozwiedziony
rozwiedziona gift ugift fraskilt samboer samboende""".split())
LOC_KEYS = r"place|ort|lieu|lugar|luogo|local|paikka|sted|place of birth|birthplace|born in|geburtsort|heimatort|lieu de naissance|lugar de nacimiento|luogo di nascita|local de nascimento|miejsce urodzenia|födelseort|fødested|syntymäpaikka"

LANGUAGE_NAMES = set("""english englisch anglais inglés ingles inglese engels engelska engelsk englanti angielski inglês
german deutsch allemand alemán aleman tedesco duits tyska tysk saksa niemiecki alemão
french französisch français francais francés frances francese frans franska fransk ranska francuski francês
spanish spanisch espagnol español espanol spagnolo spaans spanska spansk espanja hiszpański espanhol
italian italienisch italien italiano italiaans italienska italiensk italia włoski
dutch niederländisch néerlandais neerlandés olandese nederlands nederländska nederlandsk hollanti niderlandzki holandês
swedish schwedisch suédois sueco svedese zweeds svenska svensk ruotsi szwedzki
finnish finnisch finnois finlandés finlandese fins finska finsk suomi fiński finlandês
polish polnisch polonais polaco polacco pools polska polsk puola polski
portuguese portugiesisch portugais portugués portoghese portugees portugisiska portugisisk portugali portugalski português
danish dänisch danois danés danese deens danska dansk tanska duński dinamarquês norwegian norwegisch norvégien noruego norvegese noors norska norsk norja norweski norueguês
russian russisch russe ruso russo russisch ryska russisk venäjä rosyjski arabic arabisch arabe árabe arabo arabisch arabiska arabisk arabia arabski
chinese mandarin chinesisch chinois chino cinese chinees kinesiska kinesisk kiina chiński chinês japanese japanisch japonais japonés giapponese japans japanska japansk japani japoński japonês
korean hindi urdu turkish türkisch turc turco turks turkiska tyrkisk turkki turecki greek griechisch grec griego greco grieks grekiska gresk kreikka grecki grego
muttersprache native nativo madrelingua moedertaal modersmål morsmål modersmål äidinkieli ojczysty""".split())
