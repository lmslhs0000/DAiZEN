"""여러 출처의 제조사/차종 문자열 → 우리 Europe 시계열(Brand, Model) 분류기.

출처: EEA CO2(Mk, Cn, Ft), 영국 DfT VEH0160(Make, GenModel, Fuel), 스위스 auto-schweiz(Marke, Modell),
폴란드 CEPiK(marka, model), 프랑스 PFA 상위 100, 이탈리아 UNRAE 상위 50.

규칙 순서
 1) 제조사 문자열 → 표준 브랜드(BRAND_MAKE 정규식). 다른 브랜드가 모델 이름 앞에 붙으면(SEAT 아래 'CUPRA ...',
    FIAT 아래 'ABARTH ...', CITROEN 아래 'DS 7' 등) 그 브랜드로 바꾼다(EXTRA).
 2) 모델 문자열 정규화: 대문자, 악센트 제거, 영숫자 외 → 공백, '+' → PLUS, 앞의 브랜드 이름·NEW/NUEVO 등 제거.
 3) 브랜드별 특수 정규식(SPECIAL: BMW·벤츠 숫자/문자 계열, 제외 규칙) → 연료 규칙(FUEL) → 공백 뺀 문자열의 접두어 일치
    (우리 모델 이름 + ALIASES, 긴 것부터 비교: 'YARIS CROSS' 가 'YARIS' 보다 먼저).
 4) 아무 것도 안 맞으면 미대응(None). 추정으로 억지 대응하지 않는다.

신뢰도: special/alias = 규칙 기반(high 또는 지정값), prefix = 이름 접두어(high), 'low' 는 정의가 다를 수 있는 대응
(예: 포드 Tourneo Custom → 우리 'Transit Custom', 벤츠 T-Class → 'Citan').
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

from _common import RAW_DATA

# ---------------------------------------------------------------- 정규화


def norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).upper().replace("+", " PLUS ").replace("°", " ").replace("º", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("Ł", "L").replace("Ø", "O").replace("ß", "SS")
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def squash(s: str) -> str:
    return s.replace(" ", "")


# ---------------------------------------------------------------- 브랜드
# (표준 브랜드, 정규화된 제조사 문자열에 대한 정규식). 위에서부터 첫 일치.
BRAND_MAKE = [
    ("Mercedes-Benz", r"^(MERCEDES|DAIMLER|MERCEDES AMG)\b"),
    ("Volkswagen", r"^(VOLKSWAGEN|VW|VOLKSWAGENWERK)\b"),
    ("Opel/Vauxhall", r"^(OPEL|VAUXHALL)\b"),
    ("Land Rover", r"^(LAND ROVER|LANDROVER|RANGE ROVER|JAGUAR LAND ROVER)\b"),
    ("Alfa Romeo", r"^ALFA\b"),
    ("Aston Martin", r"^ASTON\b"),
    ("Rolls-Royce", r"^ROLLS\b"),
    ("Lynk & Co", r"^LYNK\b"),
    ("KG Mobility", r"^(KG MOBILITY|KGM|SSANGYONG|SSANG YONG)\b"),
    ("GWM", r"^(GWM|GREAT WALL|WEY|ORA)\b"),
    ("Beijing", r"^(BAIC|BEIJING)\b"),
    ("DR Motor", r"^DR\b"),
    ("DS", r"^DS\b"),
    ("MG", r"^(MG|SAIC MG|SAIC MOTOR|MG ROEWE|MORRIS GARAGES)\b"),
    ("Lamborghini", r"^(AUTOMOBILI )?LAMBORGHINI\b"),
    ("Ferrari", r"^FERRARI\b"),
    ("DFSK", r"^(DFSK|FORTHING|DONGFENG FORTHING|SERES)\b"),
    ("Dongfeng", r"^(DONGFENG|DFM)\b"),
    ("Omoda", r"^OMODA\b"),
    ("Jaecoo", r"^JAECOO\b"),
    ("Chery", r"^CHERY\b"),
    ("Cupra", r"^CUPRA\b"),
    ("Seat", r"^SEAT\b"),
    ("Skoda", r"^SKODA\b"),
    ("Smart", r"^SMART\b"),
    ("Tesla", r"^TESLA\b"),
    ("MINI", r"^MINI\b"),
    ("BMW", r"^(BMW|B M W|BMW I)\b"),
    ("Rolls-Royce", r"^ROLLS ROYCE\b"),
    ("ICH-X", r"^ICH\b"),
    ("Sportequipe", r"^SPORTEQUIPE\b"),
    ("Ineos", r"^INEOS\b"),
    ("Firefly", r"^FIREFLY\b"),
    ("Xpeng", r"^(XPENG|XIAOPENG)\b"),
    ("Leapmotor", r"^LEAPMOTOR\b"),
    ("Polestar", r"^POLESTAR\b"),
    ("Lexus", r"^LEXUS\b"),
    ("Toyota", r"^TOYOTA\b"),
]
# 단순 이름(정규화한 우리 브랜드 이름으로 시작) 은 아래에서 자동 추가
SIMPLE_BRANDS = [
    "Abarth", "Alpine", "Audi", "BYD", "Bentley", "Bugatti", "Cadillac", "Chevrolet", "Chrysler", "Citroen",
    "Dacia", "Denza", "Dodge", "EVO", "Ebro", "Fiat", "Ford", "Geely", "Genesis", "Honda", "Hongqi", "Hyundai",
    "JAC", "Jeep", "Jetour", "Kia", "Lada", "Lancia", "Lotus", "Maserati", "Maxus", "Mazda", "McLaren",
    "Mitsubishi", "Nio", "Nissan", "Peugeot", "Porsche", "RAM", "Renault", "Subaru", "Suzuki", "Tiger",
    "Togg", "Volvo", "Voyah", "Zeekr",
]
for _b in SIMPLE_BRANDS:
    BRAND_MAKE.append((_b, r"^" + re.escape(norm(_b)) + r"\b"))
BRAND_MAKE.append(("Lada", r"^(VAZ|AVTOVAZ)\b"))

# 모델 이름 앞에 붙어 있으면 그 브랜드로 바꾸는 하위 브랜드
EXTRA = {
    "Seat": ["Cupra"], "Fiat": ["Abarth"], "Citroen": ["DS"], "Renault": ["Alpine"], "Chery": ["Omoda", "Jaecoo"],
    "Toyota": ["Lexus"], "BMW": ["MINI"], "Volvo": ["Polestar"], "Mercedes-Benz": ["Smart"], "Dongfeng": ["Voyah"],
    "Nio": ["Firefly"], "Geely": ["Lynk & Co", "Zeekr"], "DFSK": ["Dongfeng"], "Dongfeng": ["Voyah", "DFSK"],
    "Hyundai": ["Genesis"],
}

# EEA 형식 승인 유형(T)으로만 갈리는 차종(같은 Cn 아래 섞여 있음; 2025 잠정판에서 확인)
#  (브랜드, 1차 분류 모델, T 정규식, 바꿀 모델)
TYPE_RULES = [
    ("Fiat", "Panda", r"^C$", "Grande Panda"),            # T=312 = 기존 Panda, T=C = Grande Panda
    ("Toyota", "Corolla", r"^XG1", "Corolla Cross"),       # XG1TJ(JP) = Corolla Cross, ZE1.. = Corolla
    ("Mercedes-Benz", "EQE", r"EQEX", "EQE SUV"),          # E2EQEX = X294 SUV, E2EQEW = V295 세단
    ("Mercedes-Benz", "EQS", r"EQS ?X", "EQS SUV"),
]
TYPE_SPLIT_MAKES = ["FIAT", "TOYOTA", "MERCEDES-BENZ", "MERCEDES BENZ", "MERCEDES", "MERCEDES-AMG"]


def refine_by_type(df: pd.DataFrame, type_col: str = "T") -> pd.DataFrame:
    """classify_frame 결과에 형식 유형 규칙을 덮어쓴다(type_col 이 있는 행만)."""
    df = df.copy()
    t = df[type_col].fillna("").astype(str)
    for b, m, rx, new in TYPE_RULES:
        hit = (df["Brand"] == b) & (df["Model"] == m) & t.str.contains(rx, regex=True)
        df.loc[hit, "Model"] = new
        df.loc[hit, "rule"] = df.loc[hit, "rule"].astype(str) + f"|type:{rx}"
    return df

# 모델 문자열 앞에서 지울 브랜드 별칭(정규화형)
BRAND_ALIASES = {
    "Mercedes-Benz": ["MERCEDES BENZ", "MERCEDES AMG", "MERCEDES", "MB"],
    "Volkswagen": ["VOLKSWAGEN", "VW"],
    "Opel/Vauxhall": ["OPEL", "VAUXHALL"],
    "Land Rover": ["LAND ROVER"],
    "KG Mobility": ["KG MOBILITY", "KGM", "SSANGYONG"],
    "GWM": ["GWM", "GREAT WALL"],
    "Beijing": ["BAIC", "BEIJING"],
    "DR Motor": ["DR AUTOMOBILES", "DR MOTOR"],
    "Lynk & Co": ["LYNK CO", "LYNK AND CO", "LYNK"],
    "MG": ["MG"],
    "BMW": ["BMW", "B M W"],
    "Rolls-Royce": ["ROLLS ROYCE"],
    "Alfa Romeo": ["ALFA ROMEO", "ALFA"],
    "Aston Martin": ["ASTON MARTIN"],
    "DFSK": ["DFSK"],
    "ICH-X": ["ICH X", "ICHX"],
    "Lamborghini": ["AUTOMOBILI LAMBORGHINI", "LAMBORGHINI"],
}
NEW_WORDS = r"^(NEW|NUEVO|NUEVA|NOUVEAU|NOUVELLE|NOUVEL|NEUER|NEUE|NUOVA|NUOVO|NOWY|NOWA|ALL NEW)\s+"
E_PREFIX_BRANDS = {"Peugeot", "Citroen", "Opel/Vauxhall", "Fiat", "DS", "Jeep", "Lancia", "Alfa Romeo", "Abarth"}

# ---------------------------------------------------------------- 모델 규칙
# 별칭(공백 뺀 접두어). 우리 모델 이름 자체는 자동 포함.
ALIASES = {
    ("Ford", "Escape"): ["KUGA"],
    ("Ford", "Transit Custom"): [("TOURNEOCUSTOM", "low"), ("CUSTOM", "low"), ("TCUSTOM", "low")],
    ("Ford", "Transit Courier"): [("TOURNEOCOURIER", "low"), ("COURIER", "low")],
    ("Ford", "Explorer"): ["EXPLORER"],
    ("Nissan", "Rogue"): ["XTRAIL"],
    ("Honda", "Fit"): ["JAZZ"],
    ("Honda", "e:Ny1"): ["ENY1"],
    ("Suzuki", "Vitara/Grand Vitara"): ["VITARA", "GRANDVITARA"],
    ("Subaru", "Legacy/Outback"): ["OUTBACK", "LEGACY"],
    ("Subaru", "Crosstrek"): ["CROSSTREK", "XV"],
    ("Volvo", "S60/V60"): ["S60", "V60"],
    ("Volvo", "S90/V90"): ["S90", "V90"],
    ("Volvo", "C40"): ["EC40", "C40"],
    ("Volvo", "EX40"): ["EX40"],
    ("Mazda", "MX-5 Miata"): ["MX5"],
    ("Toyota", "Yaris"): ["GRYARIS"],
    ("Toyota", "RAV4"): ["RAV4"],
    ("Toyota", "Proace"): ["PROACEVERSO", "PROACE"],
    ("Toyota", "Proace City"): ["PROACECITY"],
    ("Toyota", "Aygo X"): ["AYGOX", ("AYGO", "low")],
    ("Porsche", "718"): ["718", "BOXSTER", "CAYMAN"],
    ("Alfa Romeo", "Junior"): ["JUNIOR", "MILANO"],
    ("Abarth", "500"): ["500", "595", "695", "F595"],
    ("Renault", "Megane E-Tech"): ["MEGANEETECH", "MEGANEE"],
    ("Renault", "Scenic E-Tech"): ["SCENIC"],
    ("Renault", "Twingo E-Tech"): ["TWINGO"],
    ("Renault", "Trafic"): ["TRAFIC", "TRAFFIC"],
    ("Renault", "4"): ["4ETECH", "4", "R4"],
    ("Renault", "5"): ["5ETECH", "5", "R5"],
    ("Land Rover", "Range Rover Evoque"): ["RANGEROVEREVOQUE", "EVOQUE"],
    ("Land Rover", "Range Rover Velar"): ["RANGEROVERVELAR", "VELAR"],
    ("Land Rover", "Range Rover Sport"): ["RANGEROVERSPORT", "RRSPORT", "RANGEROVERSPRT", "RANGEROVERSPT",
                                          "RROVERSPORT", "RROVERSPRT", "RROVERSPT"],
    ("Audi", "Q3 Sportback"): ["Q3SPORTBACK", "RSQ3SPORTBACK", "Q3SB", "RSQ3SB"],
    ("Omoda", "5"): ["5", ("E5", "medium")],
    ("Citroen", "Jumpy"): ["JUMPY", ("SPACETOURER", "low")],
    ("Citroen", "Berlingo"): ["BERLINGO"],
    ("Peugeot", "Expert"): ["EXPERT", ("TRAVELLER", "low")],
    ("Fiat", "Doblo"): ["DOBLO"],
    ("Fiat", "Scudo"): ["SCUDO", ("ULYSSE", "low")],
    ("Fiat", "Panda"): ["PANDA", "PANDINA"],
    ("Opel/Vauxhall", "Combo"): ["COMBO"],
    ("Opel/Vauxhall", "Grandland"): ["GRANDLAND"],
    ("Opel/Vauxhall", "Mokka"): ["MOKKA"],
    ("Volkswagen", "ID. Buzz"): ["IDBUZZ"],
    ("Volkswagen", "Caddy"): ["CADDY"],
    ("Volkswagen", "Golf"): ["GOLF", "EGOLF"],
    ("Volkswagen", "Tiguan"): ["TIGUAN"],
    ("MINI", "Countryman"): ["COUNTRYMAN", "COUNTRY"],
    ("MINI", "Aceman"): ["ACEMAN"],
    ("MINI", "Cooper"): ["COOPER", "ONE", "JOHNCOOPERWORKS", "JCW", "HATCH", "MINI", "CABRIO", "3DOOR", "5DOOR",
                         "3PORTES", "5PORTES", "ELECTRIC"],
    ("Audi", "e-tron GT"): ["ETRONGT", "RSETRONGT", "SETRONGT"],
    ("Audi", "A3"): ["A3", "S3", "RS3"],
    ("Audi", "A5"): ["A5", "S5", "RS5"],
    ("Audi", "A6"): ["A6", "S6", "RS6"],
    ("Audi", "A8"): ["A8", "S8"],
    ("Audi", "Q3"): ["Q3", "RSQ3"],
    ("Audi", "Q5"): ["Q5", "SQ5"],
    ("Audi", "Q5 Sportback"): ["Q5SPORTBACK", "SQ5SPORTBACK", "SQ5SB", "Q5SB"],
    ("Audi", "Q7"): ["Q7", "SQ7"],
    ("Audi", "Q4 Sportback"): ["Q4SPORTBACK", "Q4SB"],
    ("Audi", "Q8"): ["Q8", "SQ8", "RSQ8", ("Q8ETRON", "low"), ("SQ8ETRON", "low")],
    ("Audi", "Q6 e-tron"): ["Q6", "SQ6"],
    ("Audi", "Q6 Sportback e-tron"): ["Q6SPORTBACK", "SQ6SPORTBACK"],
    ("Audi", "A6 e-tron"): ["A6ETRON", "S6ETRON", "A6AVANTETRON", "A6SPORTBACKETRON", "S6AVANTETRON",
                            "S6SPORTBACKETRON"],
    ("Hyundai", "Ioniq 5"): ["IONIQ5"],
    ("Hyundai", "Ioniq 6"): ["IONIQ6"],
    ("Kia", "XCeed"): ["XCEED"],
    ("Kia", "EV3"): ["EV3"],
    ("Lada", "Niva"): ["NIVA", "4X4"],
    ("Lynk & Co", "01"): ["01"],
    ("Smart", "#1"): ["1", "HASHTAG1", "HASHTAGONE"],
    ("Smart", "#3"): ["3", "HASHTAG3"],
    ("Smart", "#5"): ["5", "HASHTAG5"],
    ("GWM", "Ora 03"): ["ORA03", "FUNKYCAT", "ORAFUNKYCAT", "03"],
    ("GWM", "Ora 07"): ["ORA07", "07"],
    ("GWM", "Wey 03"): ["WEY03", "COFFEE01", "WEYCOFFEE01"],
    ("GWM", "Wey 05"): ["WEY05"],
    ("BYD", "Seal 6 DM-i"): ["SEAL6DMI", ("SEAL6", "low")],
    ("BYD", "Seal U"): ["SEALU"],
    ("DS", "N°8"): ["N8", "N 8"],
    ("DS", "N°7"): ["N7"],
    ("DS", "3"): ["3", "DS3"],
    ("DS", "4"): ["4", "DS4", "N4"],
    ("DS", "7"): ["7", "DS7"],
    ("Bentley", "Continental"): ["CONTINENTAL"],
    ("Ferrari", "Roma"): ["ROMA"],
    ("Lexus", "LBX"): ["LBX"],
    ("Mitsubishi", "Colt"): ["COLT"],
    ("Mercedes-Benz", "Citan"): ["CITAN", ("TCLASS", "low"), ("TKLASSE", "low"), ("CLASSET", "low")],
    ("Land Rover", "Range Rover"): ["RANGEROVER"],
    ("Jeep", "Grand Cherokee"): ["GRANDCHEROKEE"],
    ("Hongqi", "EHS7"): ["EHS7"],
    ("Hyundai", "Kona"): ["KONA", "KAUAI"],
    ("Jetour", "Shanhai T2"): ["SHANHAIT2", "T2"],
    ("Jetour", "Dasheng"): ["DASHENG", "DASHING"],
    ("DR Motor", "DR 5.0"): ["DR50", "DR5"],
    ("DR Motor", "DR 1.0"): ["DR10", "DR1"],
    ("DR Motor", "DR 4.0/EVO 4"): ["DR40", "DR4", "EVO4"],
    # 우리 'Kia K4' 시계열은 2016년부터 있다(누적 63만 대) → 출처가 Ceed 계열을 K4 로 부른 것. Ceed/ProCeed → K4.
    ("Kia", "K4"): ["K4", ("CEED", "medium"), ("PROCEED", "medium"), ("CEEDSW", "medium")],
    ("Audi", "Q6 Sportback e-tron"): ["Q6SPORTBACK", "SQ6SPORTBACK", "Q6SB", "SQ6SB"],
    ("Land Rover", "Discovery Sport"): ["DISCOVERYSPORT", "DISCOVERYSPT", "DISCOVRYSPORT", "DISCOVRYSPT"],
    ("Land Rover", "Discovery"): ["DISCOVERY", "DISCOVRY"],
    ("Dongfeng", "Nammi 01"): ["NAMMI01", "NAMMI", ("BOX", "medium")],
    ("Geely", "Binyue"): ["BINYUE", ("COOLRAY", "medium")],
    ("Geely", "EX5"): ["EX5", ("GALAXYE5", "medium")],
    ("Suzuki", "S-Cross"): ["SCROSS", "SX4SCROSS", "SX4SUZUKISX4SCROSS", ("SX4", "low")],
    ("GWM", "Haval H6"): ["HAVALH6", "H6"],
    ("Sportequipe", "6 GT"): ["6GT", "SPORTEQUIPE6", "6"],
    ("Sportequipe", "7 GTW"): ["7GTW", "SPORTEQUIPE7", "7"],
    ("Sportequipe", "8"): ["8", "SPORTEQUIPE8"],
    ("DFSK", "Fengon E5"): ["FENGONE5", "E5", "NAVORE5", ("GLORYE5", "low")],
    ("Leapmotor", "C10"): ["C10"],
    ("Xpeng", "P7+"): ["P7PLUS"],
    ("Toyota", "C-HR+"): ["CHRPLUS"],
}

# 브랜드별 특수 정규식(공백 있는 정규화 문자열에 적용). (모델 또는 None=제외, 정규식, 신뢰도)
_SER = lambda n: (rf"^(M\s?)?{n}\d\d[A-Z]*\b|^M{n}\b|^{n} ?(SERIES|SERIE|ER|REEKS)\b|^(SERIE|SERIES|SERIA) ?{n}\b"
                  rf"|^{n}ER\b|^{n}(ER)? (GRAN COUPE|GRAN TURISMO|COUPE|CABRIO|TOURING|LIMOUSINE)\b")
SPECIAL = {
    "BMW": [
        ("iX1", r"^IX1\b", "high"), ("iX2", r"^IX2\b", "high"), ("iX3", r"^IX3\b", "high"),
        ("iX5", r"^IX5\b", "high"),
        ("iX", r"^IX\b", "high"),
        ("i4", r"^I4\b|^SERIE ?4 I4\b", "high"), ("i5", r"^I5\b|^SERIE ?5 I5\b", "high"),
        ("i7", r"^I7\b|^SERIE ?7 I7\b", "high"),
        ("X1", r"^X1\b", "high"), ("X2", r"^X2\b", "high"), ("X3", r"^X3\b", "high"), ("X5", r"^X5\b", "high"),
        ("X6", r"^X6\b", "high"), ("X7", r"^X7\b", "high"), ("XM", r"^XM\b", "high"),
        ("2 Series Active Tourer", r"ACTIVE TOURER|^2\d\d[A-Z]* A\b|^(SERIE|SERIES) 2 ACTIVE", "high"),
        (None, r"GRAN TOURER", "high"),
        ("1 Series", _SER(1), "high"), ("2 Series", _SER(2), "high"), ("3 Series", _SER(3), "high"),
        ("4 Series", _SER(4), "high"), ("5 Series", _SER(5), "high"), ("7 Series", _SER(7), "high"),
    ],
    "Mercedes-Benz": [
        ("AMG GT 4 Door", r"AMG GT.*(4 DOOR|4 TURER|4 TUERER|4 PORTES|COUPE 4)|^AMG GT 63 S E PERFORMANCE|^AMG GT 4", "low"),
        ("AMG GT", r"^(MERCEDES )?AMG GT\b|^GT \d\d", "medium"),
        ("AMG PureSpeed", r"PURESPEED", "high"),
        ("EQE SUV", r"^(AMG )?EQE\b.*\bSUV\b|^EQE SUV", "high"),
        ("EQS SUV", r"^(AMG |MAYBACH )?EQS\b.*\bSUV\b|^EQS SUV", "high"),
        ("EQA", r"^(AMG )?EQA\b", "high"), ("EQB", r"^(AMG )?EQB\b", "high"),
        ("EQE", r"^(AMG )?EQE\b", "high"), ("EQS", r"^(AMG )?EQS\b", "high"),
        ("EQT", r"^EQT\b", "high"), ("EQV", r"^EQV\b", "high"),
        ("GLA", r"^(AMG )?GLA\b", "high"), ("GLB", r"^(AMG )?GLB\b", "high"),
        ("GLC", r"^(AMG )?GLC\b", "high"), ("GLE", r"^(AMG )?GLE\b", "high"),
        ("GLS", r"^(AMG |MAYBACH )?GLS\b", "high"),
        ("CLA", r"^(AMG )?CLA\b", "high"), ("CLE", r"^(AMG )?CLE\b", "high"),
        ("SL", r"^(AMG )?SL\b", "high"),
        ("VLE", r"^VLE\b", "high"),
        ("V-Class", r"^V \d{3}|^V ?(KLASSE|CLASS)\b|^CLASSE V\b|MARCO POLO|^V\b", "high"),
        ("Vito", r"VITO", "high"),
        ("Sprinter", r"SPRINTER", "high"),
        ("Citan", r"CITAN|^T \d{3}|^T ?(KLASSE|CLASS)\b|^CLASSE T\b", "low"),
        ("G-Class", r"^(AMG )?G \d{2,3}|^G ?(KLASSE|CLASS)\b|^CLASSE G\b|^G 580|^AMG G\b", "high"),
        ("S-Class", r"^(AMG |MAYBACH |MERCEDES MAYBACH )?S \d{3}|^S ?(KLASSE|CLASS)\b|^CLASSE S\b|^MAYBACH\b", "high"),
        ("E-Class", r"^(AMG )?E \d{2,3}|^E ?(KLASSE|CLASS)\b|^CLASSE E\b|^E\d{3}", "high"),
        ("C-Class", r"^(AMG )?C \d{2,3}|^C ?(KLASSE|CLASS)\b|^CLASSE C\b|^C\d{3}", "high"),
        ("B-Class", r"^B \d{3}|^B ?(KLASSE|CLASS)\b|^CLASSE B\b|^B\d{3}", "high"),
        ("A-Class", r"^(AMG )?A \d{2,3}|^A ?(KLASSE|CLASS)\b|^CLASSE A\b|^A\d{3}", "high"),
    ],
    "Ford": [(None, r"^((TRANSIT|TOURNEO) )?CONNECT", "high"), (None, r"^(E )?TOURNEO$|^TOURNEO \d", "high")],
    "Seat": [(None, r"^ATECA|^TARRACO|^MII", "high")],
    "Audi": [(None, r"^(S|RS ?)?A4\b", "high"), (None, r"^(S|RS ?)?A7\b|^A1\b|^Q2\b|^TT\b|^R8\b|^A1 ", "high")],
    "Toyota": [(None, r"^GR ?86|^SUPRA|^MIRAI|^BZ\b", "high")],
}

# 연료 규칙: (브랜드, 정규식, 연료 집합, 결과 모델)
ELEC = {"ELECTRIC", "ELECTRICITY", "BATTERY ELECTRIC", "E", "ELEKTRO", "ELECTRIQUE", "ENERGIA ELEKTRYCZNA",
        "ELETTRICA", "ELETTRICO", "BEV"}
FUEL = [
    ("Volvo", r"^XC40\b", ELEC, "EX40"),
    ("Volvo", r"^C40\b", ELEC, "C40"),
    ("Audi", r"^(A6|S6|RS ?6)\b", ELEC, "A6 e-tron"),
    ("Renault", r"^MEGANE\b", ELEC, "Megane E-Tech"),
    ("Toyota", r"^C HR\b", ELEC, "C-HR+"),
    ("Toyota", r"^URBAN CRUISER\b", ELEC, "Urban Cruiser"),
    ("Suzuki", r"^VITARA\b", ELEC, "e Vitara"),
    ("Opel/Vauxhall", r"^GRANDLAND\b", ELEC, "Grandland"),
]


@lru_cache(maxsize=1)
def our_models() -> dict[str, list[str]]:
    d = pd.read_csv(RAW_DATA, usecols=["Country", "Brand", "Model"])
    d = d[d["Country"] == "Europe"].drop_duplicates(["Brand", "Model"])
    out: dict[str, list[str]] = {}
    for b, m in zip(d["Brand"], d["Model"]):
        out.setdefault(b, []).append(m)
    return out


@lru_cache(maxsize=None)
def _alias_table(brand: str) -> list[tuple[str, str, str]]:
    """(접두어, 모델, 신뢰도) 목록, 긴 접두어부터."""
    rows = []
    bsq = squash(norm(brand))
    for m in our_models().get(brand, []):
        parts = [m] + ([p for p in m.split("/")] if "/" in m else [])
        for p in parts:
            sq = squash(norm(p))
            if sq:
                rows.append((sq, m, "high"))
                if sq.startswith(bsq) and len(sq) > len(bsq):  # Mazda2 → '2', MG3 → '3', DR 5.0 → '50'
                    rows.append((sq[len(bsq):], m, "high"))
        for a in ALIASES.get((brand, m), []):
            if isinstance(a, tuple):
                rows.append((squash(norm(a[0])), m, a[1]))
            else:
                rows.append((squash(norm(a)), m, "high"))
    rows = list(dict.fromkeys(rows))
    rows.sort(key=lambda r: -len(r[0]))
    return rows


def brand_of(make) -> str | None:
    nm = norm(make)
    if not nm:
        return None
    for b, rx in BRAND_MAKE:
        if re.search(rx, nm):
            return b
    return None


def _strip_brand(brand: str, s: str) -> str:
    al = BRAND_ALIASES.get(brand, []) + [norm(brand)]
    changed = True
    while changed:
        changed = False
        s2 = re.sub(NEW_WORDS, "", s)
        if s2 != s:
            s, changed = s2, True
        for a in sorted(set(al), key=len, reverse=True):
            if a and (s == a or s.startswith(a + " ")) and len(s) > len(a):
                s, changed = s[len(a):].strip(), True
                break
    if brand in E_PREFIX_BRANDS and s.startswith("E ") and len(s) > 2:
        s = s[2:]
    return s


def _unique_model_guess(nmod: str, fuel) -> tuple[str | None, str | None, str, str]:
    """제조사 빈칸(CEPiK 의 CUPRA·OMODA·BAIC·LEAPMOTOR 등): 모델 이름이 우리 목록에서 한 차종에만 맞을 때만 대응
    (4자 이상은 접두어, 3자는 완전 일치: X55·B10·C10·T03). 2자 이하(G9 등)는 하지 않는다."""
    sq = squash(nmod)
    hits = set()
    for b in our_models():
        for pre, m, conf in _alias_table(b):
            if (len(pre) >= 4 and sq.startswith(pre)) or (len(pre) == 3 and sq == pre):
                hits.add((b, m))
                break
    if len(hits) == 1:
        b, m = hits.pop()
        return b, m, "no_make_unique_model", "medium"
    return None, None, "unknown_make" + (f":ambiguous{len(hits)}" if hits else ""), ""


def classify(make, model, fuel=None) -> tuple[str | None, str | None, str, str]:
    """→ (Brand, Model, rule, confidence). 대응 없으면 Brand/Model 이 None."""
    brand = brand_of(make)
    nmod = re.sub(r"^(OMODA|JAECOO)(\d)", r"\1 \2", norm(model))
    if brand is None:  # 제조사 빈칸(CEPiK 일부) → 모델 문자열 첫 단어로 추정
        brand = brand_of(nmod)
        if brand is None:
            m = re.match(r"^(OMODA|JAECOO) \d", nmod)
            brand = {"OMODA": "Omoda", "JAECOO": "Jaecoo"}[m.group(1)] if m else None
        if brand is None:
            return _unique_model_guess(nmod, fuel)
    s = _strip_brand(brand, nmod)
    for ex in EXTRA.get(brand, []):
        ne = norm(ex)
        if s == ne or s.startswith(ne + " ") or (ne in ("OMODA", "JAECOO") and re.match(ne + r"\d", s)):
            brand = ex
            s = _strip_brand(ex, re.sub("^" + ne, "", s).strip()) or s
            break
    if brand not in our_models():
        return None, None, f"brand_not_in_our_list:{brand}", ""
    if brand == "Mercedes-Benz":
        s = re.sub(r"^(MERCEDES )?AMG (?!GT\b)", "", s)
        s = re.sub(r"^(MERCEDES )?MAYBACH (?=S\b|S \d)", "", s)
    fu = norm(fuel) if fuel is not None else ""
    for b, rx, fuels, m in FUEL:
        if b == brand and fu in fuels and re.search(rx, s):
            return brand, m, f"fuel:{rx}", "high"
    for m, rx, conf in SPECIAL.get(brand, []):
        if re.search(rx, s):
            if m is None:
                return None, None, f"excluded:{rx}", ""
            return brand, m, f"special:{rx}", conf
    sq = squash(s)
    if not sq:
        return None, None, "empty_model", ""
    for pre, m, conf in _alias_table(brand):
        if sq.startswith(pre):
            if norm(make) == "SEAT CUPRA" and m in ("Leon",):  # 스위스 자료는 Seat·Cupra Leon 합산
                return brand, m, f"prefix:{pre}|CH_seat_cupra_combined", "low"
            return brand, m, f"prefix:{pre}", conf
    for ex in EXTRA.get(brand, []):  # 하위 브랜드 전용 모델(예: 'Seat / Cupra' 아래 Formentor)
        if ex in our_models():
            for pre, m, conf in _alias_table(ex):
                if len(pre) >= 3 and sq.startswith(pre):
                    return ex, m, f"extra_prefix:{pre}", "medium"
    return None, None, f"no_rule:{brand}", ""


def classify_frame(df: pd.DataFrame, make_col: str, model_col: str, fuel_col: str | None = None) -> pd.DataFrame:
    """고유 (make, model[, fuel]) 조합마다 분류 결과를 붙인다."""
    keys = [make_col, model_col] + ([fuel_col] if fuel_col else [])
    u = df[keys].drop_duplicates().copy()
    res = [classify(r[0], r[1], r[2] if fuel_col else None) for r in u.itertuples(index=False)]
    u["Brand"] = [r[0] for r in res]
    u["Model"] = [r[1] for r in res]
    u["rule"] = [r[2] for r in res]
    u["confidence"] = [r[3] for r in res]
    return df.merge(u, on=keys, how="left")


if __name__ == "__main__":
    tests = [
        ("TOYOTA", "TOYOTA YARIS CROSS", None), ("TOYOTA", "YARIS", None), ("DACIA", "SANDERO   1.0", None),
        ("BMW", "120", None), ("BMW", "IX1 EDRIVE20", None), ("BMW", "220I ACTIVE TOURER", None),
        ("MERCEDES-BENZ", "GLC 220 D 4MATIC", None), ("MERCEDES-BENZ", "A 180", None),
        ("MERCEDES-BENZ", "EQE 350 4MATIC SUV", None), ("VOLKSWAGEN, VW", "ID.4 PRO 210KW", None),
        ("PEUGEOT", "E-208", None), ("PEUGEOT", "2008 ALLURE HYBRID 145 EDCS6", None), ("SEAT", "CUPRA FORMENTOR", None),
        ("FORD", "KUGA", None), ("VOLVO", "XC40", "electric"), ("VOLVO", "XC40", "petrol"),
        ("AUDI", "A6 AVANT E-TRON", "electric"), ("AUDI", "Q6 SUV E-TRON", "electric"), ("MAZDA", "MAZDA 2", None),
        ("MG", "MG 3", None), ("VAUXHALL", "VAUXHALL CORSA", None), ("LAND ROVER", "LAND ROVER RANGE ROVER EVOQUE", None),
        (None, "OMODA5", None), ("SKODA", "ENYAQ COUPE 85", None), ("RENAULT", "RENAULT 5 E-TECH ELECTRIC", None),
        ("FORD", "TRANSIT CONNECT", None), ("B.M.W.", "SERIE 1", None), ("MERCEDES", "CLASSE A", None),
        ("MINI", "COUNTRY3", None), ("Škoda", "Kodiaq", None), ("BMW", "1er", None), ("Mercedes-Benz", "GLC-Klasse", None),
    ]
    for t in tests:
        print(t, "→", classify(*t))
