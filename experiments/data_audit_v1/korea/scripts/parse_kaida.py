import re,html,json,pandas as pd
base='C:/Users/이원빈/AppData/Local/Temp/claude/D--daizen/d9c7c60a-e77b-4a0d-a0fb-015e0e604f1b/scratchpad/kr/'
idx=json.load(open(base+'kaida_monthly.json',encoding='utf-8'))
rows=[];best=[]
for per,(tot,aid,dt) in sorted(idx.items()):
    if per>'2026-07': continue
    t=open(base+f'kaida_art/{aid}.html',encoding='utf-8',errors='ignore').read()
    b=re.sub(r'<script.*?</script>|<style.*?</style>','',t,flags=re.S)
    x=html.unescape(re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',b)))
    i=x.find('브랜드별 등록대수'); j=x.find('배기량별',i)
    seg=x[i:j]
    url=f'https://www.kaida.co.kr/ko/news/newsView.do?seq={aid}'
    rows.append(dict(period=per,source='KAIDA monthly press release',brand='ALL_IMPORT',model_name_source='TOTAL (passenger car registrations)',units=tot,url_or_file=f'kr/kaida_art/{aid}.html'))
    annual = '연간 브랜드별' in x[i-10:i+20] or x[i-3:i].strip().endswith('연간')
    bper = per[:4]+'-ANNUAL' if annual else per
    for m in re.finditer(r'([가-힣·\s]+)\(([^)]+)\)\s*([\d,]+)\s*대?',seg):
        rows.append(dict(period=bper,source='KAIDA monthly press release',brand=m.group(2).strip(),model_name_source='BRAND_TOTAL',units=float(m.group(3).replace(',','')),url_or_file=f'kr/kaida_art/{aid}.html'))
    k=x.find('베스트셀링 모델은'); 
    if k>0:
        s=x[k:k+400]; s=s[:s.find('순')]
        ann = x[k-8:k].strip().endswith('년')
        for m in re.finditer(r'([^,()]+?)\(([\d,]+)대\)',s.replace('베스트셀링 모델은','')):
            best.append(dict(period=(per[:4]+'-ANNUAL') if ann else per,source='KAIDA monthly press release (best-selling trims)',brand='',model_name_source=m.group(1).strip(),units=float(m.group(2).replace(',','')),url_or_file=f'kr/kaida_art/{aid}.html'))
df=pd.DataFrame(rows); bd=pd.DataFrame(best)
df.to_csv('kraudit/kaida_brand_long.csv',index=False,encoding='utf-8-sig'); bd.to_csv('kraudit/kaida_best_long.csv',index=False,encoding='utf-8-sig')
print(len(df),len(bd)); print(df.brand.value_counts().to_string()); print(bd.tail(10).to_string())
# check sum of brands vs total
g=df[df.model_name_source=='BRAND_TOTAL'].groupby('period').units.sum(); tt=df[df.brand=='ALL_IMPORT'].set_index('period').units
print((g-tt).describe()); print((g-tt)[(g-tt)!=0])
