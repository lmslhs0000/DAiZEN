# Hyundai IR "Unit Sales by Model" (Domestic) component definitions per our series.
# Best-fit definitions determined empirically (see explore_hmc.txt).
HMC_DEF = {
    ('Hyundai', 'Grandeur'): ['Grandeur (HG)', 'Grandeur (IG)', 'Grandeur (HEV)', 'Grandeur (IG HEV)', 'Grandeur (GN7)', 'Grandeur (GN7 HEV)'],
    ('Hyundai', 'Elantra'): ['Avante (AD)', 'Avante (CN7)', 'Avante (CN7 HEV)', 'Avante (CN8)', 'Avante (CN8 HEV)'],
    ('Hyundai', 'Elantra N'): ['Avante (CN7 N)'],
    ('Hyundai', 'Sonata'): ['Sonata (YF)', 'Sonata (LF)', 'Sonata (HEV)', 'Sonata (PHEV)', 'Sonata (DN8)', 'Sonata (DN8 HEV)'],
    ('Hyundai', 'Santa Fe'): ['Santa-Fe (DM)', 'Santa-Fe (TM)', 'Santa-Fe (TM HEV)', 'Santa-Fe (MX5)', 'Santa-Fe (MX5 HEV)', 'SantaFe PHEV (MX5 PHEV)'],
    ('Hyundai', 'Tucson'): ['Tucson (TL)', 'Tucson (NX4)', 'Tucson (NX4 HEV)'],
    ('Hyundai', 'Palisade'): ['Palisade (LX2)', 'Palisade (LX3)', 'Palisade (LX3 HEV)'],
    ('Hyundai', 'Kona'): ['Kona', 'Kona (HEV)', 'Kona (EV)', 'Kona (OS)', 'Kona (OS HEV)', 'Kona (OS EV)', 'Kona (SX2)', 'Kona (SX2 HEV)', 'Kona (SX2 EV)'],
    ('Hyundai', 'Casper'): ['Casper (AX)'],
    ('Hyundai', 'Inster'): ['Casper (AX EV)'],
    ('Hyundai', 'Ioniq 5'): ['IONIQ 5 (NE)'],
    ('Hyundai', 'Ioniq 5 N'): ['IONIQ 5 N (NE N)'],
    ('Hyundai', 'Ioniq 6'): ['IONIQ 6 (CE)', 'IONIQ 6 N (CE N)'],
    ('Hyundai', 'Ioniq 9'): ['IONIQ 9 (ME EV)'],
    ('Hyundai', 'Venue'): ['Venue', 'Venue (QX)'],
    ('Hyundai', 'Nexo'): ['NEXO(FE)', 'NEXO (FE)', 'NEXO  (NH2)'],
    ('Genesis', 'G80'): ['G80', 'G80 (RG3)', 'G80 (RG3 EV)'],
    ('Genesis', 'G90'): ['Equus (VI)', 'EQ900', 'G90', 'G90 (HI)', 'G90 (RS4)'],
    ('Genesis', 'G70'): ['G70', 'G70 (IK)', 'G70 S/B (IK S/B)'],
    ('Genesis', 'GV80'): ['GV80 (JX)', 'GV80 Coupe (JX Coupe)'],
    ('Genesis', 'GV70'): ['GV70 (JK)', 'GV70 (JK EV)'],
    ('Genesis', 'GV60'): ['GV60 (JW)', 'GV60 (JW EV)'],
}
# Components that exist in the IR file but are intentionally excluded from our series
HMC_EXCLUDED_NOTE = {
    ('Hyundai', 'Santa Fe'): 'Santa-Fe (Long)/(NC) = Maxcruz (separate nameplate) excluded',
    ('Hyundai', 'Tucson'): 'Tucson (LMFC) = Tucson fuel-cell excluded',
    ('Hyundai', 'Kona'): 'Kona (OS N) excluded from 2021-07 (no separate Kona N series in our data)',
    ('Hyundai', 'Elantra'): 'Avante (CN7 N) excluded (separate series Elantra N)',
    ('Hyundai', 'Casper'): 'Casper (AX EV) excluded (our series Inster)',
    ('Hyundai', 'Ioniq 5'): 'IONIQ 5 N excluded (separate series Ioniq 5 N)',
    ('Hyundai', 'Sonata'): 'From 2024-01 the IR file no longer includes Sonata taxi (DN8 LPG taxi) in the Sonata rows; ours includes taxi (= manufacturer press-release/Danawa definition)',
}
HMC_URL = {
    2016: '2016/2016-sales-by-model.xlsx', 2017: '2017/2017-sales-by-model.xlsx', 2018: '2018/2018-sales-by-model.xlsx',
    2019: '2019/2019-sales-by-model.xlsx', 2020: '2020/2020-sales-by-model.xlsx', 2021: '2021/2021-sales-by-model.xlsx',
    2022: '2022/2022-sales-by-model.xlsx', 2023: '2023/hmc-sales-by-model-december-y2023.xlsx',
    2024: '2024/hmc-sales-by-model-december-y2024.xlsx', 2025: '2025/hmc-sales-by-model-dec-y2025.xlsx',
    2026: '2026/hmc-sales-by-model-august-y2026.xlsx (Jan-Jul columns only)',
}
HMC_BASE = 'https://www.hyundai.com/content/dam/hyundai/ww/en/images/company/investor-relations/ir-resources/sales-results-report/'
