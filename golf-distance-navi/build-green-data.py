"""Build small same-origin OSM feature tiles for Japan; no browser cross-origin dependency."""
import json, math, pathlib, urllib.parse, urllib.request, datetime

def point(e):
    p=e.get('center') or e
    if p.get('lat') is not None and p.get('lon') is not None:
        return float(p['lat']),float(p['lon'])
    paths=[e.get('geometry') or []]+[m.get('geometry') or [] for m in e.get('members',[]) if m.get('role')!='inner']
    pts=[p for path in paths for p in path if p.get('lat') is not None and p.get('lon') is not None]
    if pts:return sum(p['lat'] for p in pts)/len(pts),sum(p['lon'] for p in pts)/len(pts)

def build(raw,out):
    if not isinstance(raw.get('elements'),list) or raw.get('remark'):raise ValueError('Incomplete Overpass response')
    tiles={}
    for e in raw['elements']:
        if (e.get('tags') or {}).get('golf') not in ('green','hole'):continue
        p=point(e)
        if not p or not all(math.isfinite(n) for n in p):continue
        # Only keep tags required by the app; geometry remains unmodified.
        e=dict(e);e['tags']={k:v for k,v in e.get('tags',{}).items() if k in ('golf','ref','name')}
        key=f'{math.floor(p[0]*4)}_{math.floor(p[1]*4)}'
        tiles.setdefault(key,[]).append(e)
    out.mkdir(parents=True,exist_ok=True)
    for key,features in tiles.items():
        (out/(key+'.json')).write_text(json.dumps({'elements':features},ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    manifest={'version':1,'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'available':True,'count':sum(map(len,tiles.values())),'tiles':{k:len(v) for k,v in tiles.items()}}
    (out/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')),encoding='utf-8')
    return manifest

def main():
    out=pathlib.Path('golf-distance-navi/green-data')
    query='[out:json][timeout:90];nwr["golf"~"^(green|hole)$"](20,122,46,154);out geom;'
    for endpoint in ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter']:
        try:
            req=urllib.request.Request(endpoint+'?'+urllib.parse.urlencode({'data':query}),headers={'User-Agent':'GolfDistanceNavi/2.7.1 (GitHub Pages build)','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as response:raw=json.load(response)
            manifest=build(raw,out)
            print('Green data build success:',manifest['count'],'features in',len(manifest['tiles']),'tiles',flush=True)
            return
        except Exception as error:print('Green data fetch failed:',endpoint,str(error),flush=True)
    # Never present missing upstream data as an empty successful search.
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps({'version':1,'available':False,'reason':'upstream_unavailable','tiles':{}}),encoding='utf-8')
    print('WARNING: green source unavailable; browser fallback remains enabled',flush=True)

if __name__=='__main__':main()
