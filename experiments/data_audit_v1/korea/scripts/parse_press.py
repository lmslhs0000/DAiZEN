import re,html,glob,pandas as pd
rows=[]
for f in sorted(glob.glob('press/*.html')):
    t=open(f,encoding='utf-8',errors='ignore').read()
    ti=re.search('<title>(.*?)</title>',t,re.S).group(1).strip()
    m=re.search(r'(현대차|기아), (\d{4})년 (\d{1,2})월 .*판매',ti)
    if not m: continue
    per=f'{m.group(2)}-{int(m.group(3)):02d}'
    if per>'2026-07': continue
    url=re.search(r'<meta property="og:url" content="([^"]*)"',t)
    url=url.group(1) if url else f
    br='Hyundai/Genesis' if m.group(1)=='현대차' else 'Kia'
    b=re.sub(r'<script.*?</script>|<style.*?</style>','',t,flags=re.S)
    x=html.unescape(re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',b)))
    i=x.find('■ 국내'); j=x.find('■ 해외',i); seg=x[i:j]
    for mm in re.finditer(r'([가-힣A-Za-z0-9Ⅲ]+(?: [0-9A-Za-z]+)?)(?:가|로|는)? (\d{1,3}(?:,\d{3})*)대',seg):
        nm=mm.group(1)
        nm=re.sub(r'(로|가|는)$','',nm) if nm not in ('레이','카니발','니로') else nm
        if re.match(r'^\d+만$|^총$',nm): continue
        rows.append(dict(period=per,source=('Hyundai Motor' if br.startswith('H') else 'Kia')+' monthly sales press release (primary, text)',brand=br,model_name_source=nm,units=float(mm.group(2).replace(',','')),url_or_file=url))
p=pd.DataFrame(rows).drop_duplicates(['period','brand','model_name_source'])
p.to_csv('press_long.csv',index=False,encoding='utf-8-sig'); print(p.groupby('period').size()); print(sorted(p.model_name_source.unique()))
