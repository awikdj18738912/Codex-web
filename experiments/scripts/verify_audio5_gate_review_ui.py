"""Check the packaged UI in a disposable browser without creating labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(out):
    from playwright.sync_api import sync_playwright
    checks, errors = {}, []
    rows = [json.loads(line) for line in (out/'review_queue.jsonl').open()]
    partial_count = sum(row['assistant_recommendation'] == 'partial' for row in rows)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True,args=['--no-sandbox'])
        page = browser.new_context().new_page()
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:8084/review.html')
        page.wait_for_function("document.querySelector('#loadStatus').textContent.includes('159')")
        checks['loaded_all159'] = '159' in page.locator('#loadStatus').inner_text()
        checks['no_initial_labels'] = page.evaluate('localStorage.length===0')
        checks['decision_blank'] = page.locator('select').nth(1).input_value() == ''
        checks['target_blank'] = page.locator('textarea').nth(0).input_value() == ''
        checks['heard_unchecked'] = not page.locator('input[type=checkbox]').is_checked()
        page.select_option('#filter','supplement_missed')
        checks['miss_filter27'] = '27' in page.locator('#position').inner_text()
        page.select_option('#filter','recommend_partial')
        checks['partial_filter_matches_queue'] = page.locator('#position').inner_text().strip().endswith('/ '+str(partial_count))
        page.fill('#jump','120'); page.click('#go')
        checks['jump120'] = '120' in page.locator('h2').inner_text()
        page.locator('audio').evaluate('(a)=>{a.muted=true;a.load()}')
        page.wait_for_function('document.querySelector("audio").readyState>=2')
        page.locator('audio').evaluate('(a)=>a.play()')
        page.wait_for_function('!document.querySelector("audio").paused')
        checks['audio_loaded_and_played'] = page.locator('audio').evaluate('(a)=>a.duration>840&&!a.error')
        page.locator('audio').evaluate('(a)=>a.pause()')
        checks['terminal_info'] = page.get_by_text('整条转录终态结果：包含后级局部补救',exact=True).count() == 1
        page.fill('#jump','118'); page.click('#go')
        checks['unsafe_flag'] = page.locator('.unsafe').count() == 1
        checks['unsafe_target_not_prefilled'] = page.locator('textarea').nth(0).input_value() == ''
        page.get_by_role('button',name='保存本条核验',exact=True).click()
        checks['blank_save_rejected'] = page.evaluate('localStorage.length===0')
        checks['no_JS_errors'] = not errors
        browser.close()
    (out/'browser_verification.json').write_text(json.dumps(dict(checks=checks,errors=errors,
        no_quality_labels_created=True),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(checks,ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit('Browser verification failed')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
