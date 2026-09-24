import json, time, urllib.request, os, subprocess, sys
paths=json.load(open('/tmp/us8k_paths.json'))
base='https://huggingface.co/datasets/danavery/urbansound8K/resolve/main/'
outdir='data/uscraping/parquet'; os.makedirs(outdir, exist_ok=True)
for i,p in enumerate(paths):
    fn=os.path.join(outdir, os.path.basename(p))
    if os.path.exists(fn) and os.path.getsize(fn) > 1e8: print('have',i,fn,flush=True); continue
    for attempt in range(8):
        try:
            t0=time.time()
            subprocess.run(['curl','-sL','--retry','5','--retry-all-errors','--retry-delay','3',
                            '-C','-','-o',fn, base+p], check=True, timeout=900)
            print(f'done {i} {os.path.getsize(fn)/1e6:.0f}MB in {time.time()-t0:.0f}s', flush=True)
            break
        except Exception as e:
            print(f'retry {attempt} for {i}: {e}', flush=True); time.sleep(5)
    else:
        print('GAVEUP', i, flush=True)
print('ALL DONE', flush=True)
