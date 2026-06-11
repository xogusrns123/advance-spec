import os, json, time
os.environ['MPLBACKEND']='Agg'
import matplotlib; matplotlib.use('Agg')
nb=json.load(open('/home/muchwater/advance-spec/simulation/notebooks/anchor_depth_analysis.ipynb'))
ns={}
t0=time.time()
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    src=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
    print(f'>>> CELL {i} ({len(src)} chars)', flush=True)
    exec(src, ns)
    print(f'    cell {i} done @ {time.time()-t0:.0f}s', flush=True)
print('>>> NOTEBOOK RENDER COMPLETE @ %.0fs'%(time.time()-t0), flush=True)
