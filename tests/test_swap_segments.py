"""Source-window and multi-person prompt contract; no network or model inference."""
import copy
import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
fake_server = types.ModuleType('server')
fake_server.PromptServer = None
sys.modules.setdefault('server', fake_server)

from h3_prompt_enhancer import nodes_swap_segments as swap


def fixture_plan():
    windows = swap.source_windows(481, 141, 22, 481)
    observed = swap.observation_indices(481, windows, 2)
    raw = dict(unresolved=[], edit_instructions=['Remove the source watermark.'], cast=[
        dict(person_id='P1', source_description='Short-haired man first seen at opening.',
             reference_slot='ref_image_0', replacement_description='Red-haired reference man.', evidence_frames=[0]),
        dict(person_id='P2', source_description='Woman entering later, same identity through crossing.',
             reference_slot='ref_image_1', replacement_description='Dark-haired reference woman.', evidence_frames=[240]),
    ], segments=[])
    for i, window in enumerate(windows):
        raw['segments'].append(dict(**window, present_people=['P1'] if i == 0 else ['P1', 'P2'],
                                   description='P1 turns in the source lighting.', start_state='P1 faces left.',
                                   end_state='P1 turns toward the door.', events=[]))
    raw['segments'][-1]['events'] = [dict(kind='fade_out', start_frame=456, end_frame=481,
                                         evidence_frame=480, description='The source image fades out here.')]
    return windows, observed, raw


def validated():
    windows, observed, raw = fixture_plan()
    return swap.validate_plan(raw, windows, ['ref_image_0', 'ref_image_1'], observed, 481)


def test_windows_match_actual_four_segment_chain():
    windows, observed, _ = fixture_plan()
    assert [(w['start_frame'], w['head_trim'], w['deliver']) for w in windows] == [
        (0, 0, 141), (119, 22, 119), (238, 22, 119), (357, 22, 102)]
    assert len(observed) == 20
    assert {0, 140, 119, 141, 259, 238, 260, 378, 357, 379, 480} <= set(observed)
    assert sum(w['deliver'] for w in windows) == 481
    assert swap.source_windows(22, 22, 22, 22)[0]['deliver'] == 22


def test_absent_person_does_not_leak_and_reference_ids_stay_stable():
    plan = validated()
    first = swap.render_segment(plan, plan['segments'][0])
    assert '<Picture 1>' in first and '<Picture 2>' not in first
    assert '<Subject 3>' not in first and 'Dark-haired' not in first
    assert 'Remove the source watermark.' in first
    second = plan['segments'][1]
    second.update(present_people=['P2'], description='P2 enters from the right.',
                  start_state='P2 is offscreen until the source entrance.', end_state='P2 stands in cold light.')
    text = swap.render_segment(plan, second)
    assert '<Picture 2>' in text and '<Picture 1>' not in text
    assert '<Subject 3>' in text and '<Subject 4>' in text
    assert 'cold light' in text


def test_final_event_keeps_source_position_and_is_not_copied():
    plan = validated()
    assert 'source image fades out' not in swap.render_segment(plan, plan['segments'][0])
    last = swap.render_segment(plan, plan['segments'][-1])
    assert 'At local 4.125-5.167s' in last
    assert 'source image fades out' in last
    assert 'repeated-source-frame padding' in last


@pytest.mark.parametrize('problem', ['outside_event', 'absent_person', 'unknown_reference', 'wrong_window', 'unresolved'])
def test_invalid_plan_stops_before_sampling(problem):
    windows, observed, raw = fixture_plan()
    if problem == 'outside_event':
        raw['segments'][0]['events'] = copy.deepcopy(raw['segments'][-1]['events'])
    elif problem == 'absent_person':
        raw['segments'][0]['description'] = 'P2 is dancing here.'
    elif problem == 'unknown_reference':
        raw['cast'][1]['reference_slot'] = 'ref_image_8'
    elif problem == 'wrong_window':
        raw['segments'][1]['start_frame'] += 1
    else:
        raw['unresolved'] = ['P2 cannot be distinguished from P1; specify clothing.']
    with pytest.raises(ValueError):
        swap.validate_plan(raw, windows, ['ref_image_0', 'ref_image_1'], observed, 481)


def test_evidence_at_action_endpoint_is_valid_but_other_windows_are_not():
    windows, observed, raw = fixture_plan()
    raw['segments'][0]['events'] = [dict(kind='action', start_frame=0, end_frame=48,
                                       evidence_frame=48, description='P1 lifts both hands.')]
    swap.validate_plan(raw, windows, ['ref_image_0', 'ref_image_1'], observed, 481)
    raw['segments'][0]['events'][0].update(end_frame=141, evidence_frame=141)
    with pytest.raises(ValueError, match='real window'):
        swap.validate_plan(raw, windows, ['ref_image_0', 'ref_image_1'], observed, 481)


def test_empty_window_does_not_spawn_any_reference_person():
    plan = validated()
    segment = plan['segments'][0]
    segment.update(present_people=[], description='The source shows an empty corridor.',
                   start_state='Empty corridor.', end_state='Light gradually changes in the empty corridor.')
    text = swap.render_segment(plan, segment)
    assert '<Picture' not in text and '<Subject' not in text
    assert 'no replacement target is present' in text
    assert 'Light gradually changes' in text


def test_picker_rejects_changed_segment_plan():
    node = swap.H3SwapSegmentPrompt.PREPARE_CLASS_CLONE(None)
    encoded = json.dumps(validated())
    out = node.execute(encoded, 2, 119, 141, 22, 119)
    assert 'Motion Context' in out.result[0]
    with pytest.raises(ValueError, match='no longer matches'):
        node.execute(encoded, 2, 120, 141, 22, 119)


def test_planner_one_cloud_call_labels_images_and_history(monkeypatch, tmp_path):
    from h3_prompt_enhancer import common
    _, _, raw = fixture_plan()
    calls = []
    monkeypatch.setattr(swap, 'chat', lambda *a, **kw: calls.append((a, kw)) or json.dumps(raw))
    monkeypatch.setattr(swap, 'make_progress_cb', lambda _: None)
    monkeypatch.setattr(swap, 'image_part', lambda image: {'type':'image_url', 'image_url':{'url':str(float(image.mean()))}})
    monkeypatch.setattr(common, 'HISTORY_PATH', str(tmp_path/'history.jsonl'))
    node = swap.H3SwapPromptPlan.PREPARE_CLASS_CLONE(None)
    kwargs = dict(instruction='ref_image_0 replaces P1; ref_image_1 replaces P2; remove watermark',
                  model='workbuddy/deepseek-v4.1-flash', thinking='enabled', temperature=0.7, seed=0,
                  sample_length=141, context_length=22, delivered_frames=481,
                  source_frames=torch.zeros(481, 8, 8, 3),
                  ref_images={'ref_image_0':[torch.zeros(1,8,8,3)], 'ref_image_1':[torch.ones(1,8,8,3)]})
    result = node.execute(**kwargs)
    assert len(calls) == 1
    content = calls[0][0][3]
    assert len([p for p in content if p['type']=='image_url']) == 22
    assert any('SOURCE FRAME 119' in p.get('text','') for p in content)
    assert 'Segment 1' in result.result[1] and 'ref_image_1' in result.result[1]
    entry = common.read_history()[0]
    assert entry['kind'] == 'swap_plan'
    monkeypatch.setattr(swap, 'find_history_entry', lambda _: entry)
    replay = node.execute(**kwargs, history_entry='saved')
    assert replay.result[0] == result.result[0] and len(calls) == 1
    reversed_raw = copy.deepcopy(raw)
    reversed_raw['cast'][0]['reference_slot'] = 'ref_image_1'
    reversed_raw['cast'][1]['reference_slot'] = 'ref_image_0'
    monkeypatch.setattr(swap, 'chat', lambda *a, **kw: json.dumps(reversed_raw))
    with pytest.raises(ValueError, match='mapping was not respected'):
        node.execute(**kwargs)
    kwargs['source_frames'] = torch.ones(481,8,8,3)
    with pytest.raises(ValueError, match='differs'):
        node.execute(**kwargs, history_entry='saved')


def test_schema_registration():
    schema = swap.H3SwapPromptPlan.define_schema()
    assert schema.node_id == 'H3SwapPromptPlan'
    assert len(schema.outputs) == 2
