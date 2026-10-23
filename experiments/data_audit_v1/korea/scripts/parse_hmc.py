import openpyxl, pandas as pd, re
rows=[]
for y in range(2016,2027):
    wb=openpyxl.load_workbook(f'hmc_{y}.xlsx',data_only=True)
    ws=wb.worksheets[0]
    sect=None; seg=None
    for r in ws.iter_rows(values_only=True):
        r=list(r)
        lab=[x for x in r[:3]]
        s1=r[1]; s2=r[2]
        if isinstance(s1,str) and s1.strip() in ('Domestic','Export'): sect=s1.strip(); continue
        if isinstance(s1,str) and s1.strip() in ('Total','Grand Total') or (isinstance(s1,str) and s1.startswith('Total')):
            if sect=='Domestic' and s1.strip()=='Total':
                name='TOTAL'; seg='TOTAL'
            else:
                if sect=='Domestic': sect='after'
                continue
        elif isinstance(s1,str) and s1.strip() in ('PC','RV','CV'):
            seg=s1.strip(); name=s2
        else:
            name=s2
        if sect!='Domestic' or not isinstance(name,str): continue
        name=name.strip()
        vals=r[3:15]
        for m,v in enumerate(vals,1):
            per=f'{y}-{m:02d}'
            if per>'2026-07': continue
            if isinstance(v,str):
                v2=v.strip().replace(',','')
                v = 0.0 if v2 in ('-','') else float(v2)
            rows.append(dict(period=per,source='Hyundai Motor IR - Unit Sales by Model (Domestic)',brand='Hyundai/Genesis',segment=seg,model_name_source=name,units=v,
                url_or_file=f'hmc_{y}.xlsx'))
df=pd.DataFrame(rows)
df.to_csv('hmc_long.csv',index=False,encoding='utf-8-sig')
print(df.groupby(['segment','model_name_source']).period.agg(['min','max','count']).to_string())
