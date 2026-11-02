"""Persist an unfiltered full regression and independent browser evidence QA."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def regression(out):
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();cmd=[sys.executable,'-m','pytest','-o','addopts=','-ra']
    paths=list(Path('src').rglob('*.py'))+list(Path('tests').rglob('*.py'))+list(Path('demo').glob('*.py'))+list(Path('scripts').glob('*.py'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    with (out/'pytest.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    record=dict(command=cmd,exit_code=result.returncode,seconds=time.perf_counter()-started,unfiltered=True,
        log_sha256=hashlib.sha256((out/'pytest.log').read_bytes()).hexdigest(),
        source_hashes_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items()))
    (out/'regression.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    if result.returncode:raise SystemExit(result.returncode)


def browser(out):
    from playwright.sync_api import sync_playwright
    import threading
    from http.server import ThreadingHTTPServer
    from demo.server import Handler
    out.mkdir(parents=True,exist_ok=False);errors=[]
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    try:
        with sync_playwright() as p:
            b=p.chromium.launch(headless=True);page=b.new_page(viewport={'width':1280,'height':900})
            page.on('pageerror',lambda error:errors.append(str(error)));page.goto(url);page.click('#nav-progress')
            card=page.locator('#current-evidence');card.scroll_into_view_if_needed();assert card.is_visible()
            assert card.locator('tbody tr').count()==2
            links=card.locator('a').all()
            for link in links:assert page.request.get(url+link.get_attribute('href')).ok
            card.screenshot(path=str(out/'desktop.png'))
            page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            card.screenshot(path=str(out/'mobile.png'));assert not errors
            b.close()
        (out/'result.json').write_text(json.dumps(dict(passed=True,links=len(links),rows=2,page_errors=errors,mobile_overflow=False),indent=2))
        print('Browser checks passed',flush=True)
    finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['regression','browser']);p.add_argument('--output',required=True)
    a=p.parse_args();(regression if a.mode=='regression' else browser)(Path(a.output))
