import re,glob,os,pandas as pd,html
def num(s): 
    s=re.sub(r'[^0-9\-]','',s); return float(s) if s not in ('','-') else None
def parse(f):
    t=open(f,encoding='utf-8',errors='ignore').read()
    nat,per=os.path.basename(f)[:-5].split('_')
    out=[]; trims=[]
    for m in re.finditer(r"<tr>\s*<td><input type='checkbox' name='compItemCk' value='record_(\d+)' title='([^']*)' brand='(\d+)'>.*?<td class='num'>([\d,]+)",t,re.S):
        out.append(dict(nat=nat,period=per,model_id=m.group(1),name=html.unescape(m.group(2)).strip(),brand_id=m.group(3),sales=num(m.group(4))))
    for m in re.finditer(r"<tr class='sub model_(\d+)' name='sub_model'>.*?<td class='title'><span>(.*?)</span></td>\s*<td class='num'>([\d,]+)</td>",t,re.S):
        trims.append(dict(nat=nat,period=per,model_id=m.group(1),trim=html.unescape(m.group(2)).strip(),sales=num(m.group(3))))
    brands=dict(re.findall(r"brand(?:=|_)?['\"]?(\d+)['\"]?[^>]*>\s*(?:<[^>]+>\s*)*([가-힣A-Za-z]+)",'')) 
    return out,trims
if __name__=='__main__':
    import sys
    rows=[];tr=[]
    for f in sorted(glob.glob('dn/*.html'))+sys.argv[1:]:
        a,b=parse(f); rows+=a; tr+=b
    pd.DataFrame(rows).to_csv('dn_models.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(tr).to_csv('dn_trims.csv',index=False,encoding='utf-8-sig')
    print(len(rows),len(tr))
