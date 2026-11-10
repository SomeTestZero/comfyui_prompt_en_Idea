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
                                   end_state='P1 turns toward the door.',
                                   target_description='P1 turns in the source lighting.',
                                   target_start_state='P1 faces left.', target_end_state='P1 turns toward the door.', events=[]))
    raw['segments'][-1]['events'] = [dict(kind='fade_out', start_frame=456, end_frame=481,
                                         evidence_frame=480, description='The source image fades out here.',
                                         target_description='The source image fades out here.')]
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
                  start_state='P2 is offscreen until the source entrance.', end_state='P2 stands in cold light.',
                  target_description='P2 enters from the right.', target_start_state='P2 is offscreen until the source entrance.',
                  target_end_state='P2 stands in cold light.')
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
                                       evidence_frame=48, description='P1 lifts both hands.', target_description='P1 lifts both hands.')]
    swap.validate_plan(raw, windows, ['ref_image_0', 'ref_image_1'], observed, 481)
    raw['segments'][0]['events'][0].update(end_frame=141, evidence_frame=141)
    with pytest.raises(ValueError, match='real window'):
        swap.validate_plan(raw, windows, ['ref_image_0', 'ref_image_1'], observed, 481)


def test_empty_window_does_not_spawn_any_reference_person():
    plan = validated()
    segment = plan['segments'][0]
    segment.update(present_people=[], description='The source shows an empty corridor.',
                   start_state='Empty corridor.', end_state='Light gradually changes in the empty corridor.',
                   target_description='The source shows an empty corridor.', target_start_state='Empty corridor.',
                   target_end_state='Light gradually changes in the empty corridor.')
    text = swap.render_segment(plan, segment)
    assert '<Picture' not in text and '<Subject' not in text
    assert 'no replacement target is present' in text
    assert 'Light gradually changes' in text


@pytest.mark.parametrize('field', ['target_description', 'target_start_state', 'target_end_state'])
def test_plan_requires_edited_target_separate_from_source_evidence(field):
    windows, observed, raw = fixture_plan()
    del raw['segments'][0][field]
    with pytest.raises(ValueError, match=field):
        swap.validate_plan(raw, windows, ['ref_image_0', 'ref_image_1'], observed, 481)


def test_picker_rejects_legacy_plan_before_sampling():
    node = swap.H3SwapSegmentPrompt.PREPARE_CLASS_CLONE(None)
    plan = validated()
    plan['version'] = 1
    with pytest.raises(ValueError, match='unsupported plan version'):
        node.execute(json.dumps(plan), 1, 0, 141, 0, 141)


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
    assert 'Anti-copy clause' in calls[0][0][2]
    assert swap.APPEARANCE_AUTHORITY in calls[0][0][2]
    content = calls[0][0][3]
    assert len([p for p in content if p['type']=='image_url']) == 22
    assert any('SOURCE FRAME 119' in p.get('text','') for p in content)
    assert 'Segment 1' in result.result[1] and 'ref_image_1' in result.result[1]
    entry = common.read_history()[0]
    assert entry['kind'] == 'swap_plan'
    # Exercise the real history lookup: its default kind is 'optimize'.
    replay = node.execute(**kwargs, history_entry=common.history_label(entry))
    assert replay.result[0] == result.result[0] and len(calls) == 1
    reversed_raw = copy.deepcopy(raw)
    reversed_raw['cast'][0]['reference_slot'] = 'ref_image_1'
    reversed_raw['cast'][1]['reference_slot'] = 'ref_image_0'
    monkeypatch.setattr(swap, 'chat', lambda *a, **kw: json.dumps(reversed_raw))
    with pytest.raises(ValueError, match='mapping was not respected'):
        node.execute(**kwargs)
    kwargs['source_frames'] = torch.ones(481,8,8,3)
    with pytest.raises(ValueError, match='differs'):
        node.execute(**kwargs, history_entry=common.history_label(entry))


def test_source_clothing_stays_source_evidence_not_replacement_appearance():
    plan = validated()
    segment = plan['segments'][0]
    segment.update(description='P1 turns with her gold skirt billowing.',
                   start_state='P1 has pale-blonde hair and a silver halter top.',
                   end_state='P1 raises both arms.',
                   target_description='P1 turns in the white reference robe.',
                   target_start_state='P1 has the red hair of the reference.', target_end_state='P1 raises both arms.')
    text = swap.render_segment(plan, segment)
    assert '[Shot 1] <Subject 1> turns in the white reference robe.' in text
    assert 'gold skirt' not in text and 'silver halter' not in text and 'pale-blonde' not in text
    assert 'From the very first frame in which <Subject 2> is visible, <Subject 1> seamlessly replaces' in text
    assert 'The face, hair, ears, outfit, proportions and distinguishing features of <Subject 1> come from <Picture 1>' in text
    assert 'reverting to <Subject 2>' in text
    assert 'color, costume state' not in text
    # Rendering must not rewrite the evidence used for tracking or replay.
    assert segment['description'] == 'P1 turns with her gold skirt billowing.'


def test_explicit_appearance_exception_is_scoped_and_does_not_spawn_absent_person():
    windows, observed, raw = fixture_plan()
    raw['edit_instructions'] += ["Keep P2's source outfit; replace only the face and hair."]
    plan = swap.validate_plan(raw, windows, ['ref_image_0', 'ref_image_1'], observed, 481)
    first = swap.render_segment(plan, plan['segments'][0])
    second = swap.render_segment(plan, plan['segments'][1])
    assert "Keep <Subject 4>'s source outfit" not in first
    assert "Keep <Subject 4>'s source outfit" in second
    assert 'explicit user appearance exceptions' in second
    assert '<Subject 3> is the replacement identity from <Picture 2>' in second
    raw['edit_instructions'] += ['Keep P9 unchanged.']
    with pytest.raises(ValueError, match='unknown source person'):
        swap.validate_plan(raw, windows, ['ref_image_0', 'ref_image_1'], observed, 481)


def test_preserved_person_keeps_source_appearance_and_performance():
    plan = validated()
    plan['cast'][1].update(reference_slot=None, replacement_description='')
    segment = plan['segments'][1]
    segment.update(description='P1 crosses behind P2 in a green coat.',
                   start_state='P2 holds a red umbrella.', end_state='P1 exits left while P2 stays.',
                   target_description='P1 crosses behind P2 in a green coat.',
                   target_start_state='P2 holds a red umbrella.', target_end_state='P1 exits left while P2 stays.')
    text = swap.render_segment(plan, segment)
    assert '<Subject 4>: fully_preserved, including source appearance and performance.' in text
    assert '<Subject 1> crosses behind <Subject 4> in a green coat.' in text
    assert '<Subject 4> holds a red umbrella.' in text
    assert '<Picture 2>' not in text
    assert 'reverting to <Subject 4>' not in text


def test_cloud_plan_corrects_invalid_timeline_once_before_returning(monkeypatch, tmp_path):
    from h3_prompt_enhancer import common
    windows, _, raw = fixture_plan()
    invalid = copy.deepcopy(raw)
    invalid['segments'][-1]['events'][0]['end_frame'] = windows[-1]['end_frame']
    responses = iter([json.dumps(invalid), json.dumps(raw)])
    calls = []
    monkeypatch.setattr(swap, 'chat', lambda *a, **kw: calls.append(a) or next(responses))
    monkeypatch.setattr(swap, 'make_progress_cb', lambda _: None)
    monkeypatch.setattr(swap, 'image_part', lambda _: {'type':'image_url', 'image_url':{'url':'test'}})
    monkeypatch.setattr(common, 'HISTORY_PATH', str(tmp_path/'history.jsonl'))
    node = swap.H3SwapPromptPlan.PREPARE_CLASS_CLONE(None)
    kwargs = dict(instruction='replace P1 and P2', model='workbuddy/deepseek-v4.1-flash',
                  thinking='enabled', temperature=0.7, seed=0, sample_length=141, context_length=22,
                  delivered_frames=481, source_frames=torch.zeros(481,8,8,3),
                  ref_images={'ref_image_0':[torch.zeros(1,8,8,3)], 'ref_image_1':[torch.zeros(1,8,8,3)]})
    result = node.execute(**kwargs)
    assert len(calls) == 2 and 'Validation error' in calls[1][3][-1]['text']
    assert json.loads(result.result[0])['segments'][-1]['events'][0]['end_frame'] == 481
    assert len(common.read_history()) == 1
    monkeypatch.setattr(swap, 'chat', lambda *a, **kw: json.dumps(invalid))
    with pytest.raises(ValueError, match='real source window'):
        node.execute(**kwargs)
    assert len(common.read_history()) == 1  # failed correction never becomes reusable history


def test_schema_registration_and_plan_history_kind(monkeypatch):
    from h3_prompt_enhancer import common
    monkeypatch.setattr(common, 'read_history', lambda: [
        dict(ts='1', kind='swap_plan', task_type='segments', input='swap'),
        dict(ts='2', kind='optimize', task_type='other', input='story'),
    ])
    schema = swap.H3SwapPromptPlan.define_schema()
    assert schema.node_id == 'H3SwapPromptPlan'
    assert len(schema.outputs) == 2
    history = next(field for field in schema.inputs if field.id == 'history_entry')
    assert any('segments' in option for option in history.options)
    assert not any('story' in option for option in history.options)
