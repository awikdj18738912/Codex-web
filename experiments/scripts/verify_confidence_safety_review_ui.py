"""Exercise review UI in an isolated browser; never create audio gold labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile


def verify(args):
    from playwright.sync_api import sync_playwright

    checks = {}; errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        context = browser.new_context(accept_downloads=True, viewport={'width': 1100, 'height': 1100})
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        checks['initial_unknown'] = (page.locator('#decision').input_value() == 'uncertain'
            and page.locator('#target').input_value() == '' and not page.locator('#heard').is_checked())
        checks['opinions_initially_collapsed'] = page.locator('#opinions').evaluate('(el)=>!el.parentElement.open')
        page.locator('#audio').evaluate('(el)=>{el.muted=true;el.load()}')
        page.wait_for_function('document.querySelector("#audio").readyState>=2')
        page.locator('#audio').evaluate('(el)=>el.play()')
        page.wait_for_function('document.querySelector("#audio").currentTime>0.05')
        checks['audio_playback'] = page.locator('#audio').evaluate('(el)=>!el.paused&&!el.error')
        page.locator('#audio').evaluate('(el)=>el.pause()')
        page.click('#save')
        checks['missing_reviewer_rejected'] = '审核者' in page.locator('#message').inner_text()
        page.fill('#reviewer', 'ui_automation_fixture_not_audio_reviewer')
        page.click('#save')
        checks['unheard_rejected'] = '勾选' in page.locator('#message').inner_text()
        # The next checkbox and saved values are UI simulation only. They are
        # destroyed with this browser and never imported into diagnostic data.
        page.check('#heard'); page.click('#save')
        checks['uncertain_requires_note'] = '备注' in page.locator('#message').inner_text()
        page.fill('#note', 'UI automation fixture; no quality assessment')
        page.click('#save')
        checks['pending_filter_advances'] = '1 / 61' == page.locator('#position').inner_text()
        page.select_option('#filter', 'all')
        checks['unknown_roundtrip'] = page.locator('#decision').input_value() == 'uncertain'
        page.select_option('#decision', 'keep'); page.select_option('#effect', 'neutral'); page.click('#save')
        page.reload()
        page.select_option('#filter', 'all')
        checks['keep_saved_after_reload'] = page.locator('#decision').input_value() == 'keep'
        with tempfile.TemporaryDirectory() as temporary:
            with page.expect_download() as downloaded:
                page.click('#download')
            path = Path(temporary) / 'synthetic_ui_export.jsonl'
            downloaded.value.save_as(path)
            exported = [json.loads(x) for x in path.read_text().splitlines()]
            checks['export_correct_fields'] = (len(exported) == 1 and exported[0]['decision'] == 'keep'
                and exported[0]['desired_text'] == page.locator('#baseline').inner_text()
                and exported[0]['edit_tasks'] == [] and bool(exported[0]['packet_basis_sha256']))
        page.select_option('#decision', 'edit')
        page.fill('#target', page.locator('#baseline').inner_text() + '测试。')
        page.check('input[name="task"][value="local_typo"]')
        page.check('input[name="task"][value="boundary_punctuation"]')
        page.click('#save')
        checks['mixed_task_order_required'] = '顺序' in page.locator('#message').inner_text()
        page.fill('#taskOrder', 'boundary_punctuation, local_typo, boundary_punctuation')
        page.click('#save')
        with page.expect_download() as downloaded:
            page.click('#download')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'synthetic_ui_export.jsonl'; downloaded.value.save_as(path)
            exported = json.loads(path.read_text().splitlines()[0])
            checks['mixed_task_order_preserved'] = exported['edit_tasks'] == ['boundary_punctuation', 'local_typo', 'boundary_punctuation']
        page.click('#undo')
        checks['undo_restores_unknown'] = page.locator('#decision').input_value() == 'uncertain'
        page.locator('#rawAsr').evaluate('(el)=>el.closest("details").open=true')
        checks['full_clip_ASR_displayed'] = page.locator('#rawAsr').inner_text() == page.evaluate('current().raw_recording_text')
        if args.screenshot:
            page.screenshot(path=str(args.screenshot), full_page=True)
        context.close(); browser.close()
    checks['no_javascript_errors'] = not errors
    result = dict(checks=checks, javascript_errors=errors,
        isolated_browser=True, synthetic_reviews_not_imported=True, audio_quality_labels_created=0)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8083/review.html')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--screenshot', type=Path)
    verify(parser.parse_args())
