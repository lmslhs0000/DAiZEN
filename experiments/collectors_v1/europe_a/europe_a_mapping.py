"""Name normalisation: country make/model strings -> OUR Europe series (raw/data.csv Country=Europe).

Used by build_mapping.py (mapping_europe_a.csv) and build_validation.py.

Matching order for one source string (make, model):
  1. make -> our Brand via BRAND_ALIAS (normalised); cross-brand fixes (e.g. SEAT 'CUPRA FORMENTOR').
  2. brand-specific PRE rewrites (e.g. Audi 'RS 3' -> 'A 3', Peugeot 'E 208' -> '208').
  3. brand-specific REGEX rules (ordered, first hit wins; target None = known 'no Europe series').
  4. automatic token-prefix match against aliases of our model names (+ ALIAS extras);
     the longest alias wins (so 'Q4 SPORTBACK' beats 'Q4', 'YARIS CROSS' beats 'YARIS').
Normalisation: upper case, '+' -> ' PLUS', '#' dropped, every non-alphanumeric -> space,
space inserted at every letter<->digit boundary ('ID3' == 'ID.3' == 'ID 3', 'CX-30' == 'CX30').
"""
import re
import unicodedata


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    s = s.upper().replace('+', ' PLUS ').replace('#', ' ')
    s = re.sub(r'[^A-Z0-9]+', ' ', s)
    s = re.sub(r'(?<=[A-Z])(?=[0-9])|(?<=[0-9])(?=[A-Z])', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


# normalised make (any country, incl. KBA) -> our Europe Brand
BRAND_ALIAS = {
    'VW': 'Volkswagen', 'VOLKSWAGEN': 'Volkswagen', 'VOLKSWAGEN VW': 'Volkswagen',
    'OPEL': 'Opel/Vauxhall', 'VAUXHALL': 'Opel/Vauxhall',
    'MERCEDES': 'Mercedes-Benz', 'MERCEDES BENZ': 'Mercedes-Benz', 'MERCEDES AMG': 'Mercedes-Benz',
    'LYNK CO': 'Lynk & Co', 'LYNK': 'Lynk & Co', 'LYNK AND CO': 'Lynk & Co',
    'KG MOBILITY': 'KG Mobility', 'KGM': 'KG Mobility', 'SSANGYONG': 'KG Mobility',
    'DS': 'DS', 'DS AUTOMOBILES': 'DS', 'DR': 'DR Motor', 'DR MOTOR': 'DR Motor', 'DR AUTOMOBILES': 'DR Motor',
    'GWM': 'GWM', 'GREAT WALL': 'GWM', 'GREAT WALL MOTOR COMPANY LIMIT': 'GWM', 'GREAT WALL MOTOR': 'GWM',
    'ORA': 'GWM', 'WEY': 'GWM',
    'BAIC': 'Beijing', 'BEIJING': 'Beijing',
    'MG': 'MG', 'MG ROEWE': 'MG', 'MG MOTOR UK': 'MG', 'SAIC MG': 'MG',
    'ROLLS ROYCE': 'Rolls-Royce', 'LAND ROVER': 'Land Rover', 'RANGE ROVER': 'Land Rover',
    'ALFA ROMEO': 'Alfa Romeo', 'ASTON MARTIN': 'Aston Martin', 'MINI': 'MINI', 'BMW MINI': 'MINI',
    'NIO': 'Nio', 'XPENG': 'Xpeng', 'XPENG MOTORS': 'Xpeng', 'ICH X': 'ICH-X', 'ICHX': 'ICH-X',
    'SPORTEQUIPE': 'Sportequipe', 'EVO': 'EVO', 'EBRO': 'Ebro', 'TOGG': 'Togg',
    'LEAPMOTOR': 'Leapmotor', 'LEAP MOTOR': 'Leapmotor', 'CITROEN': 'Citroen', 'SKODA': 'Skoda',
    'GENESIS': 'Genesis', 'DONGFENG': 'Dongfeng', 'DFSK': 'DFSK', 'FIREFLY': 'Firefly',
    'JAECOO': 'Jaecoo', 'OMODA': 'Omoda', 'CHERY': 'Chery', 'BYD': 'BYD', 'ZEEKR': 'Zeekr',
    'POLESTAR': 'Polestar', 'TESLA': 'Tesla', 'TESLA MOTORS': 'Tesla', 'SMART': 'Smart',
    'HONGQI': 'Hongqi', 'VOYAH': 'Voyah', 'JAC': 'JAC', 'JETOUR': 'Jetour', 'MAXUS': 'Maxus',
    'TIGER': 'Tiger', 'LADA': 'Lada', 'INEOS': 'Ineos', 'INEOS AUTOMOTIVE': 'Ineos',
    'DODGE': 'Dodge', 'RAM': 'RAM', 'CADILLAC': 'Cadillac', 'CHEVROLET': 'Chevrolet', 'CHRYSLER': 'Chrysler',
    'MCLAREN': 'McLaren', 'BENTLEY': 'Bentley', 'BUGATTI': 'Bugatti', 'FERRARI': 'Ferrari',
    'LAMBORGHINI': 'Lamborghini', 'MASERATI': 'Maserati', 'LOTUS': 'Lotus', 'PORSCHE': 'Porsche',
    'ABARTH': 'Abarth', 'ALPINE': 'Alpine', 'CUPRA': 'Cupra', 'SEAT': 'Seat', 'DENZA': 'Denza',
    'SHINERAY': 'Geely',  # UK DfT files Geely Starray under make SHINERAY (checked: SHINERAY STARRAY)
}

# brand -> list of (regex on normalised model string, our Model or None). First hit wins.
REGEX = {
    'Mercedes-Benz': [
        (r'^(MERCEDES )?AMG GT (4|63 S E|63 E|43|53|63) .*(4 ?D|DOOR|COUPE 4)', 'AMG GT 4 Door'),
        (r'^(AMG )?GT 4 ?(DOOR|PUERTAS)', 'AMG GT 4 Door'),
        (r'^(MERCEDES )?AMG GT', 'AMG GT'), (r'^EQS SUV|^EQS \d+ SUV|^EQS \d+ 4 ?MATIC SUV', 'EQS SUV'),
        (r'^EQE SUV|^EQE \d+ SUV|^EQE \d+ 4 ?MATIC SUV', 'EQE SUV'),
        (r'^(MERCEDES )?(AMG )?A (\d|CLASS|KLASSE|AMG|LIMOUSINE)', 'A-Class'), (r'^A$', 'A-Class'),
        (r'^(AMG )?B (\d|CLASS|KLASSE)', 'B-Class'), (r'^B$', 'B-Class'),
        (r'^(AMG )?C (\d|CLASS|KLASSE|ESTATE)', 'C-Class'), (r'^C$', 'C-Class'),
        (r'^(AMG )?E (\d|CLASS|KLASSE|ESTATE)', 'E-Class'), (r'^E$', 'E-Class'),
        (r'^(AMG )?S (\d|CLASS|KLASSE)|^MAYBACH S', 'S-Class'), (r'^S$', 'S-Class'),
        (r'^(AMG )?G (\d|CLASS|KLASSE)', 'G-Class'), (r'^G$', 'G-Class'),
        (r'^V (\d|CLASS|KLASSE)', 'V-Class'), (r'^V$', 'V-Class'), (r'^VLE', 'VLE'),
        (r'^(AMG )?SL (\d|CLASS|$)', 'SL'), (r'^(AMG )?CLE', 'CLE'), (r'^(AMG )?CLA', 'CLA'),
        (r'^(AMG )?GLA', 'GLA'), (r'^(AMG )?GLB', 'GLB'), (r'^(AMG )?GLC', 'GLC'),
        (r'^(AMG )?GLE', 'GLE'), (r'^(AMG )?GLS|^MAYBACH GLS', 'GLS'),
        (r'^E ?VITO', 'Vito'), (r'^E ?SPRINTER', 'Sprinter'), (r'^E ?CITAN', 'Citan'), (r'^T (\d|CLASS)', None),
    ],
    'BMW': [
        (r'ACTIVE TOURER|^2 \d\d [A-Z]* ?(A ?T)\b|^2 ER ACTIVE', '2 Series Active Tourer'),
        (r'GRAN TOURER', None),
        (r'^I 4\b', 'i4'), (r'^I 5\b', 'i5'), (r'^I 7\b', 'i7'), (r'^I X 1\b|^IX 1\b', 'iX1'),
        (r'^I X 2\b|^IX 2\b', 'iX2'), (r'^I X 3\b|^IX 3\b', 'iX3'), (r'^IX 5\b', 'iX5'), (r'^I X\b|^IX\b', 'iX'),
        (r'^(M )?1\d\d\b|^1 SERIES|^1 ER', '1 Series'),
        (r'^M 2\b|^(M )?2\d\d\b|^2 SERIES|^2 ER', '2 Series'),
        (r'^M 3\b|^(M )?3\d\d\b|^3 SERIES|^3 ER', '3 Series'),
        (r'^M 4\b|^(M )?4\d\d\b|^4 SERIES|^4 ER', '4 Series'),
        (r'^M 5\b|^(M )?5\d\d\b|^5 SERIES|^5 ER', '5 Series'),
        (r'^(M )?7\d\d\b|^7 SERIES|^7 ER', '7 Series'),
        (r'^(M )?8\d\d\b|^8 SERIES|^M 8\b', None),
        (r'^X 1\b', 'X1'), (r'^X 2\b', 'X2'), (r'^X 3\b', 'X3'), (r'^X 4\b', None), (r'^X 5\b', 'X5'),
        (r'^X 6\b', 'X6'), (r'^X 7\b', 'X7'), (r'^XM\b', 'XM'), (r'^Z 4\b', None),
    ],
    'Audi': [
        (r'^Q 4 .*SPORTBACK|^Q 4 SB', 'Q4 Sportback'), (r'^Q 3 .*SPORTBACK', 'Q3 Sportback'),
        (r'^Q 5 .*SPORTBACK', 'Q5 Sportback'), (r'^Q 6 .*SPORTBACK', 'Q6 Sportback e-tron'),
        (r'^Q 6\b', 'Q6 e-tron'), (r'^A 6 .*E TRON|^A 6 E\b', 'A6 e-tron'), (r'^E TRON GT|^GT E TRON', 'e-tron GT'),
        (r'^Q 8 .*E TRON|^E TRON( |$)', 'Q8'), (r'^A 4\b', None), (r'^A 1\b', None), (r'^Q 2\b', None),
        (r'^A 7\b', None), (r'^TT\b|^R 8\b', None),
    ],
    'Volkswagen': [
        (r'^ID 7', 'ID.7'), (r'^ID 4', 'ID.4'), (r'^ID 5', 'ID.5'), (r'^ID 3', 'ID.3'), (r'^ID BUZZ', 'ID. Buzz'),
        (r'^ID POLO', 'ID.Polo'), (r'^ID CROSS', 'ID.Cross'), (r'^T ?ROC', 'T-Roc'), (r'^T ?CROSS', 'T-Cross'),
        (r'^E GOLF|^GOLF', 'Golf'), (r'^MULTIVAN|^CARAVELLE|^TRANSPORTER|^CALIFORNIA|^ARTEON|^UP\b|^E UP', None),
    ],
    'Volvo': [
        (r'^XC 40 .*(RECHARGE|P 6|P 8|SINGLE|TWIN|EXTENDED|BEV|ELECTRIC)', 'EX40'),
        (r'^C 40|^EC 40', 'C40'), (r'^[SV] 60', 'S60/V60'), (r'^[SV] 90', 'S90/V90'),
    ],
    'Toyota': [
        (r'^AYGO', 'Aygo X'), (r'^PROACE CITY', 'Proace City'), (r'^PROACE MAX', 'Proace Max'),
        (r'^PROACE', 'Proace'), (r'^LAND ?CRUISER', 'Land Cruiser'), (r'^BZ 4', 'bZ4X'),
        (r'^GR YARIS', 'Yaris'), (r'^GR 86|^GR SUPRA|^SUPRA|^MIRAI', None),
    ],
    'Ford': [
        (r'^KUGA', 'Escape'), (r'^(TRANSIT|TOURNEO) CUSTOM', 'Transit Custom'),
        (r'^(TRANSIT|TOURNEO) COURIER', 'Transit Courier'), (r'^(TRANSIT|TOURNEO) CONNECT', None),
        (r'^TRANSIT|^TOURNEO', 'Transit'), (r'^MUSTANG MACH|^MACH E', 'Mustang Mach-E'),
        (r'^FOCUS|^FIESTA|^S MAX|^GALAXY|^RANGER RAPTOR', None),
    ],
    'Honda': [(r'^JAZZ', 'Fit'), (r'^E NY 1|^ENY 1', 'e:Ny1')],
    'Nissan': [(r'^X TRAIL|^XTRAIL', 'Rogue'), (r'^E NV 200|^NV 200', None)],
    'Suzuki': [(r'^E VITARA|^EVITARA', 'e Vitara'), (r'^GRAND VITARA|^VITARA', 'Vitara/Grand Vitara'),
               (r'^S CROSS|^SX 4', 'S-Cross')],
    'Subaru': [(r'^OUTBACK|^LEGACY', 'Legacy/Outback'), (r'^XV\b', 'Crosstrek')],
    'Kia': [(r'^X ?CEED', 'XCeed'), (r'^E NIRO|^NIRO', 'Niro'), (r'^E SOUL', None), (r'^PRO ?CEED|^CEED', None)],
    'Hyundai': [(r'^IONIQ 5', 'Ioniq 5'), (r'^IONIQ 6', 'Ioniq 6'), (r'^IONIQ 9', 'Ioniq 9'),
                (r'^IONIQ\b', None), (r'^SANTA ?FE', 'Santa Fe')],
    'Renault': [(r'^MEGANE.*E ?TECH.*ELEC|^MEGANE E TECH', 'Megane E-Tech'),
                (r'^SCENIC', 'Scenic E-Tech'), (r'^TWINGO', 'Twingo E-Tech'), (r'^R ?5\b', '5'), (r'^R ?4\b', '4'),
                (r'^5\b', '5'), (r'^4\b', '4'), (r'^ZOE|^MEGANE|^TALISMAN|^KADJAR', None)],
    'Citroen': [(r'^E? ?C 3 AIRC', 'C3 Aircross'), (r'^E? ?C 4 X', 'C4 X'), (r'^E? ?C 5 X', 'C5 X'),
                (r'^E? ?C 5 AIRC', 'C5 Aircross'), (r'^E? ?C 3\b', 'C3'), (r'^E? ?C 4\b', 'C4'),
                (r'^E? ?BERLINGO', 'Berlingo'), (r'^E? ?(JUMPY|SPACE ?TOURER)', 'Jumpy'),
                (r'^E? ?(JUMPER|RELAY)', 'Jumper'), (r'^AMI', 'Ami')],
    'Opel/Vauxhall': [(r'^GRANDLAND', 'Grandland'), (r'^COMBO', 'Combo'), (r'^ZAFIRA|^VIVARO|^MOVANO|^INSIGNIA', None)],
    'Fiat': [(r'^500 ?[XL]\b', None), (r'^DUCATO', 'Ducato'), (r'^E? ?DOBLO', 'Doblo'), (r'^E? ?SCUDO', 'Scudo'),
             (r'^E? ?ULYSSE|^TIPO|^QUBO', None)],
    'Mazda': [(r'^MX 5', 'MX-5 Miata'), (r'^MX 30', None), (r'^6 E\b|^MAZDA 6 E', 'Mazda6e'),
              (r'^CX 6 E', 'CX-6e'), (r'^MAZDA 6\b|^6\b', None)],
    'MG': [(r'^MGS ?5|^S 5\b', 'S5'), (r'^MGS ?6|^S 6\b', 'S6'), (r'^MGS ?9|^S 9\b', 'S9'),
           (r'^(MG ?)?4 URBAN', 'MG4 Urban'), (r'^(E )?HS\b|^EHS', None), (r'^MG ?5\b|^5\b', None)],
    'Skoda': [(r'^ENYAQ .*COUPE', 'Enyaq Coupe')],
    'Dacia': [(r'^SANDERO', 'Sandero')],
    'Seat': [(r'^ATECA', None), (r'^TARRACO|^ALHAMBRA|^MII', None)],
    'Lexus': [(r'^GS\b', 'ES')],
    'Porsche': [(r'^718|^BOXSTER|^CAYMAN', '718')],
    'Land Rover': [(r'^(RANGE|R) ?ROVER (SPORT|SPRT|SPT)\b', 'Range Rover Sport'), (r'^RANGE ROVER EVOQUE|^EVOQUE', 'Range Rover Evoque'),
                   (r'^RANGE ROVER VELAR|^VELAR', 'Range Rover Velar'), (r'^RANGE ROVER', 'Range Rover'),
                   (r'^DISCOVERY SPORT', 'Discovery Sport'), (r'^DISCOVERY', 'Discovery'), (r'^DEFENDER', 'Defender')],
    'Omoda': [(r'^E 5\b|^5\b', '5')],
    'Jaecoo': [(r'^J 7\b|^7\b', '7'), (r'^J 5\b|^E J 5|^5\b', '5'), (r'^J 8\b|^8\b', '8')],
    'Chery': [(r'^TIGGO 4|^TIGGO 2', None)],
    'Geely': [(r'^STARRAY', 'Starray'), (r'^E 5\b|^EX 5', 'EX5')],
    'Peugeot': [(r'^TRAVELLER', 'Expert'), (r'^RIFTER|^PARTNER|^508\b', None)],
    'MINI': [(r'^JCW|^JOHN COOPER WORKS', 'Cooper'),
             (r'^COOPER|^ONE\b|^HATCH|^CONVERTIBLE|^CABRIO|^3 DOOR|^5 DOOR|^ELECTRIC', 'Cooper')],
    'DS': [(r'^N 4\b', '4')],
    'DFSK': [(r'^E 5\b', 'Fengon E5')],
    'Zeekr': [(r'^7 GT', None)],
    'Smart': [(r'^FOR ?TWO|^FOR ?FOUR', None)],
    'Mitsubishi': [(r'^SPACE STAR|^MIRAGE', None)],
}

# extra aliases (our Model -> extra source names), brand-scoped
ALIAS = {
    ('Mazda', 'Mazda2'): ['2', 'MAZDA 2'], ('Mazda', 'Mazda3'): ['3', 'MAZDA 3'],
    ('MG', 'MG3'): ['3', 'MG 3'], ('MG', 'MG4'): ['4', 'MG 4'],
    ('Hyundai', 'i10'): ['I 10'], ('Hyundai', 'i20'): ['I 20'], ('Hyundai', 'i30'): ['I 30'],
    ('Kia', 'XCeed'): ['X CEED'], ('Volvo', 'S60/V60'): ['S 60', 'V 60'], ('Volvo', 'S90/V90'): ['S 90', 'V 90'],
    ('Subaru', 'Legacy/Outback'): ['OUTBACK'], ('Suzuki', 'Vitara/Grand Vitara'): ['VITARA'],
    ('DR Motor', 'DR 4.0/EVO 4'): ['DR 4', '4 0', 'EVO 4'], ('DR Motor', 'DR 5.0'): ['5 0', 'DR 5'],
    ('DR Motor', 'DR 1.0'): ['1 0', 'DR 1'],
    ('Toyota', 'C-HR+'): ['C HR PLUS', 'CHR PLUS'], ('Toyota', 'C-HR'): ['CHR'],
    ('Xpeng', 'P7+'): ['P 7 PLUS'], ('Tesla', 'Model 3'): ['MODEL 3'], ('Tesla', 'Model Y'): ['MODEL Y'],
    ('Volkswagen', 'T-Roc'): ['TROC'], ('Volkswagen', 'T-Cross'): ['TCROSS'],
    ('Renault', 'Megane E-Tech'): ['MEGANE E TECH'], ('Renault', 'Scenic E-Tech'): ['SCENIC E TECH'],
    ('Leapmotor', 'T03'): ['T 03'], ('Leapmotor', 'C10'): ['C 10'], ('Leapmotor', 'B10'): ['B 10'],
    ('Mercedes-Benz', 'EQA'): ['EQA'], ('BYD', 'Atto 2'): ['ATTO 2'],
    ('Fiat', 'Grande Panda'): ['GRANDE PANDA'],
    ('Ebro', 'S700'): ['S 700'], ('Ebro', 'S400'): ['S 400'], ('Ebro', 'S800'): ['S 800'], ('Ebro', 'S900'): ['S 900'],
    ('Nissan', 'Rogue'): ['X TRAIL'], ('Honda', 'Fit'): ['JAZZ'], ('Ford', 'Escape'): ['KUGA'],
    ('Cupra', 'Leon'): ['LEON'], ('Seat', 'Leon'): ['LEON'],
    ('BYD', 'Seal 6 DM-i'): ['SEAL 6'], ('BYD', 'Seal 5 DM-i'): ['SEAL 5'], ('BYD', 'Sealion 5 DM-i'): ['SEALION 5'],
    ('BYD', 'Dolphin G DM-i'): ['DOLPHIN G'],
}


NO_STRIP = {'RANGE ROVER'}  # brand alias that is also the start of model names


def our_series(raw_csv, country='Europe'):
    import pandas as pd
    d = pd.read_csv(raw_csv)
    e = d[d['Country'] == country]
    return sorted(set(map(tuple, e[['Brand', 'Model']].drop_duplicates().values)))


class Matcher:
    def __init__(self, series):
        self.series = series
        self.by_brand = {}
        for b, m in series:
            self.by_brand.setdefault(b, []).append(m)
        self.brand_norm = {norm(b): b for b in self.by_brand}
        self.aliases = {}  # brand -> list of (alias_tokens, model)
        for b, models in self.by_brand.items():
            bt = norm(b).split()
            lst = []
            for m in models:
                al = {norm(m)}
                t = norm(m).split()
                if len(t) > len(bt) and t[:len(bt)] == bt:  # 'Mazda2' -> '2'
                    al.add(' '.join(t[len(bt):]))
                for x in ALIAS.get((b, m), []):
                    al.add(norm(x))
                for a in al:
                    if a:
                        lst.append((a.split(), m))
            self.aliases[b] = lst

    def brand(self, make):
        n = norm(make)
        if n in BRAND_ALIAS:
            return BRAND_ALIAS[n]
        if n in self.brand_norm:
            return self.brand_norm[n]
        return None

    def match(self, make, model):
        """Return (Brand, Model or None, rule, confidence). Brand None = brand not in our Europe list."""
        b = self.brand(make)
        s = norm(model)
        s = re.sub(r'^(NUEVO|NUEVA|NEW|ALL NEW|THE NEW) ', '', s)
        # cross-brand: SEAT/CUPRA naming, MINI under BMW, Abarth under Fiat, Genesis under Hyundai
        if s.startswith('CUPRA ') and b in ('Seat', 'Cupra'):
            b, s = 'Cupra', s[6:]
        if b == 'Fiat' and s.startswith('ABARTH '):
            b, s = 'Abarth', s[7:]
        if s.startswith('ICH X ') and b == 'Sportequipe':
            b, s = 'ICH-X', s[6:]
        if b is None:
            return None, None, 'brand_not_in_europe_list', ''
        # strip a leading brand word repeated inside the model text ('TOYOTA YARIS CROSS'); longest first
        prefs = ({norm(make), norm(b)} | {k for k, v in BRAND_ALIAS.items() if v == b}) - NO_STRIP
        for pref in sorted(prefs, key=lambda x: (-len(x), x)):
            if pref and (s == pref or s.startswith(pref + ' ')) and not (b, s) in self.series:
                rest = s[len(pref):].strip()
                if rest:
                    s = rest
                break
        s = PRE.get(b, lambda x: x)(s)
        for pat, target in REGEX.get(b, []):
            if re.search(pat, s):
                if target is None:
                    return b, None, f'regex_none:{pat}', 'high'
                if (b, target) in self.series:
                    return b, target, f'regex:{pat}', 'medium'
        toks = s.split()
        best = None
        for at, m in self.aliases.get(b, []):
            if toks[:len(at)] == at and (best is None or len(at) > len(best[0])):
                best = (at, m)
        if best:
            exact = len(best[0]) == len(toks)
            return b, best[1], 'prefix' if not exact else 'exact', 'high' if exact else 'medium'
        return b, None, 'no_match', ''


def _pre_audi(s):
    s = re.sub(r'^R?S ?Q (\d)', r'Q \1', s)
    s = re.sub(r'^RS (\d)\b', r'A \1', s)
    s = re.sub(r'^S (\d)\b', r'A \1', s)
    return s


PRE = {
    'Audi': _pre_audi,
    'Peugeot': lambda s: re.sub(r'^E ', '', s),
    'Opel/Vauxhall': lambda s: re.sub(r'^E ', '', s),
    'Mercedes-Benz': lambda s: re.sub(r'^(MERCEDES )?(BENZ )?', '', s),
    'Kia': lambda s: s,
}
