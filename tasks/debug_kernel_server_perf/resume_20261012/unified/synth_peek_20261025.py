import json,glob,collections
for p in sorted(glob.glob('/opt/tiger/w/synth_meld_results_20261025/*/kernel_results.json')):
  r=json.load(open(p)); t=collections.defaultdict(lambda: collections.defaultdict(list))
  for x in r['records']: t[x['capture']][x['arm']].append(x['milliseconds'])
  print(p.split('/')[-2])
  for c in sorted(t):
    m={a:sum(v)/len(v) for a,v in t[c].items()}; b=m.get('u_np_u')
    s=' '.join(f"{a.replace('native_pdl_geometry_','').replace('u_','')}={m[a]:.3f}" + (f"({b/m[a]:.2f})" if b and a!='u_np_u' else '') for a in ['u_np_u','native_pdl_geometry_v3','fa3_full_default','u_np_nosync','u_npp','u_nppj','u_pers','u_dsmem_c8','u_dsmem_c16','u_np_lpt2','u_np_kvpos','u_np16_1','u_np16_0'] if a in m)
    print('  ',c[:6],s)
