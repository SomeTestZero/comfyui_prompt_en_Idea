"""Plan a source-driven swap once; select a validated local prompt per MC window."""
import hashlib
import json
import math
import re
from datetime import datetime

from comfy_api.latest import io

from .categories import PROMPT_ENHANCE
from .common import append_history, find_history_entry, log, make_progress_cb
from .nodes_cloud import ENHANCE_MAX_TOKENS, chat, image_part, parse_model_option, require_vision
from .nodes_swap import H3SwapPromptCloud, build_swap_user_text

FPS = 24
PLAN_VERSION = 1
PERSON = re.compile(r"\bP([1-9]\d*)\b")
KINDS = {"action", "entrance", "exit", "lighting", "camera", "cut", "fade_in", "fade_out"}

PLAN_SYSTEM = """Analyze the complete source video for a segmented MiniMax H3 character replacement edit.
Return ONLY one JSON object, no markdown. The supplied frames have absolute SOURCE FRAME indices at 24fps.
This is a source-driven edit, not a new story. Image references supply identity; source frames supply performance,
expression, cast presence, scene, lighting and changes. Never invent motion, audio, entrances or endings.

Global analysis is identity/context, NOT a fixed scene/lighting/outfit contract. Preserve observed changes.
Identify ALL source people across ALL windows first. Assign P1, P2, ... by first observed appearance; simultaneous
arrivals are ordered left-to-right at first appearance. Preserve these IDs through crossings, occlusion, exits
and re-entry; never restart numbering per segment. A montage reference sheet represents ONE identity.
Resolve the user's mappings (ref_image_0 -> P1 etc). The explicit attached socket-to-Picture table is authoritative.
An unassigned source person is preserved. A reference alone is never evidence that a person is in the source.
If a requested person/reference mapping cannot be resolved, or an identity across views is ambiguous, return
nonempty unresolved with actionable reasons; do not silently pick a target. Sparse images cannot prove unseen events.

Inspect each supplied sampling window, including its MC head overlap. present_people contains only the source
people observed in that window, including someone present only in the overlap or someone entering mid-window.
An absent person gets no local description and must not be spawned. Events for entrances/exits preserve source
population, timing, occlusion and interaction. Source garments may change; follow user intent/reference without
inventing wardrobe transitions. Neighboring windows inform identification and continuity only: do NOT move their
events into this window. No segment is a standalone story with its own opening/ending.

Write compact English descriptions with P1/P2 labels, never <Subject N> or <Picture N> tags; software binds labels.
Keep user names/trigger words verbatim. Define each source identity and replacement identity once in cast. These descriptions contain identification
cues only, with no timeline, action, lighting, scene or presence claims. Put source appearance changes in the
local segments. A replacement description must not impose a fixed pose, expression, lighting or environment.
Also return edit_instructions: a compact English list of the user's non-identity edits (e.g. remove the watermark).
These are task instructions, not scene/lighting locks. Empty list when no additional edit was requested.
Do not include P IDs or cinematic narrative in edit_instructions.
For EVERY exact supplied window, return one segment with the SAME integer index/start_frame/end_frame/head_trim/
deliver values. start_frame inclusive, end_frame exclusive; these include repeated-frame tail padding.
description covers this window only, with no timestamps or whole-film duration. start_state/end_state describe
observed boundary states, not commands to freeze them. events use integer ABSOLUTE source frames, confined to
the actual nonpadded portion of that window. Each event needs evidence_frame from the supplied images in that
window. Evidence may be an observed endpoint at start_frame or end_frame (provided it still belongs to this
source window); it is an observation of that action's boundary, not a license to describe another window.
Do not invent exact entrance timing between observations: keep descriptions source-relative when unsure.
A black sample at the full video's end is NOT a fade at each segment end. Describe a fade/cut only in a window
with visible local evidence of it; preserve real source changes, including changes in the middle of the film.
Never repeat a final fade in earlier windows. No sound was provided: do not infer dialogue/music from stills.
All free-text segment fields may mention ONLY that segment's present_people.

JSON schema (no additional keys needed):
{"unresolved": [], "edit_instructions": [], "cast": [
 {"person_id":"P1", "source_description":"visible identity and tracking cues",
  "reference_slot":"ref_image_0", "replacement_description":"identity from that reference, once",
  "evidence_frames":[0,24]},
 {"person_id":"P2", "source_description":"...", "reference_slot":null,
  "replacement_description":"", "evidence_frames":[240]}],
 "segments":[{"index":1,"start_frame":0,"end_frame":141,"head_trim":0,"deliver":141,
 "present_people":["P1"], "description":"P1 performs the observed action; exact performance follows the source.",
 "start_state":"P1 ...", "end_state":"P1 ...",
 "events":[{"kind":"action","start_frame":0,"end_frame":48,"evidence_frame":24,
 "description":"P1 ..."}]}]}
"""


def source_windows(frame_count, sample_length, context_length, delivered_frames):
    if not 5 <= sample_length <= 362 or sample_length % 17 != 5:
        raise ValueError("Swap plan: sample_length must be the segment planner's 17n+5 length (5..362).")
    if context_length not in (5, 22, 39, 56) or (context_length >= sample_length and delivered_frames > sample_length):
        raise ValueError("Swap plan: context_length must match MC (5/22/39/56) and be shorter than the window.")
    if not 5 <= delivered_frames <= frame_count:
        raise ValueError("Swap plan: delivered_frames must match the segment planner's total_delivered.")
    step = sample_length - context_length
    count = 1 if delivered_frames <= sample_length else 1 + math.ceil((delivered_frames - sample_length) / step)
    windows = []
    for i in range(count):
        start = i * step
        head = context_length if i else 0
        deliver = min(sample_length - head, delivered_frames - start - head)
        windows.append(dict(index=i + 1, start_frame=start, end_frame=start + sample_length,
                            head_trim=head, deliver=deliver))
    return windows


def observation_indices(frame_count, windows, sample_interval):
    if not math.isfinite(sample_interval) or sample_interval <= 0:
        raise ValueError("Swap plan: sample_interval must be a positive number of seconds.")
    step = max(1, round(sample_interval * FPS))
    indices = set(range(0, frame_count, step))
    indices.add(frame_count - 1)
    for w in windows:
        for f in (w['start_frame'], w['start_frame'] + w['head_trim'],
                  w['start_frame'] + w['head_trim'] + w['deliver'] - 1, w['end_frame'] - 1):
            indices.add(min(frame_count - 1, f))
    return sorted(indices)


def reference_images(ref_images):
    pairs = []
    for slot in sorted(ref_images or {}, key=lambda name: int(name.rsplit('_', 1)[-1])):
        value = ref_images[slot]
        if value is None:
            continue
        if len(value) != 1:
            raise ValueError(f"Swap plan: {slot} needs one reference image, not a batch; use additional sockets.")
        frame = value[0]
        if frame.ndim == 4:
            if frame.shape[0] != 1:
                raise ValueError(f"Swap plan: {slot} needs one reference image.")
            frame = frame[0]
        pairs.append((slot, frame))
    return pairs


def _integer(value, label):
    if type(value) is not int:
        raise ValueError(f"Swap plan: {label} must be an integer.")
    return value


def validate_plan(raw, windows, slots, observed, frame_count):
    if not isinstance(raw, dict) or not isinstance(raw.get('unresolved'), list):
        raise ValueError("Swap plan: expected a JSON object with unresolved, cast and segments.")
    if raw['unresolved']:
        raise ValueError("Swap plan needs clearer target identification: " + json.dumps(raw['unresolved'], ensure_ascii=False))
    cast, segments = raw.get('cast'), raw.get('segments')
    if not isinstance(cast, list) or not cast or not isinstance(segments, list):
        raise ValueError("Swap plan: cast must be nonempty and segments must be a list.")
    edits = raw.get('edit_instructions')
    if not isinstance(edits, list) or any(not isinstance(t, str) or not t.strip() or PERSON.search(t) for t in edits):
        raise ValueError('Swap plan: edit_instructions must contain only non-identity user edits.')
    ids, assigned = set(), set()
    for i, person in enumerate(cast, 1):
        if not isinstance(person, dict) or person.get('person_id') != f'P{i}':
            raise ValueError("Swap plan: cast IDs must be unique consecutive P1, P2, ... in first-appearance order.")
        ids.add(person['person_id'])
        if not isinstance(person.get('source_description'), str) or not person['source_description'].strip():
            raise ValueError(f"Swap plan: P{i} has no source identity description.")
        slot = person.get('reference_slot')
        if not isinstance(person.get('replacement_description'), str):
            raise ValueError(f"Swap plan: P{i} replacement_description must be text (empty when preserved).")
        if slot is not None:
            if slot not in slots:
                raise ValueError(f"Swap plan: P{i} references disconnected socket {slot}.")
            if not isinstance(person.get('replacement_description'), str) or not person['replacement_description'].strip():
                raise ValueError(f"Swap plan: P{i} has no replacement identity description.")
            assigned.add(slot)
        evidence = person.get('evidence_frames')
        if not isinstance(evidence, list) or not evidence or any(type(f) is not int or f not in observed for f in evidence):
            raise ValueError(f"Swap plan: P{i} needs observed source-frame evidence.")
    if set(slots) != assigned:
        raise ValueError("Swap plan: unused reference mapping(s): " + ', '.join(sorted(set(slots) - assigned)))
    if len(segments) != len(windows):
        raise ValueError("Swap plan: segment count differs from the current source windows; regenerate the plan.")
    for segment, window in zip(segments, windows):
        if not isinstance(segment, dict):
            raise ValueError("Swap plan: each segment must be an object.")
        for key, value in window.items():
            if _integer(segment.get(key), key) != value:
                raise ValueError(f"Swap plan: segment {window['index']} {key} differs from the current planner.")
        present = segment.get('present_people')
        if not isinstance(present, list) or any(not isinstance(p, str) for p in present) or len(set(present)) != len(present) or not set(present) <= ids:
            raise ValueError("Swap plan: present_people contains an unknown or duplicate person.")
        texts = []
        for field in ('description', 'start_state', 'end_state'):
            value = segment.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Swap plan: segment {window['index']} needs {field}.")
            texts.append(value)
        events = segment.get('events')
        if not isinstance(events, list):
            raise ValueError("Swap plan: events must be a list.")
        for event in events:
            if not isinstance(event, dict) or event.get('kind') not in KINDS:
                raise ValueError("Swap plan: unsupported event kind.")
            start = _integer(event.get('start_frame'), 'event start_frame')
            end = _integer(event.get('end_frame'), 'event end_frame')
            evidence = _integer(event.get('evidence_frame'), 'evidence_frame')
            real_end = min(window['end_frame'], frame_count)
            if not window['start_frame'] <= start < end <= real_end:
                raise ValueError(f"Swap plan: segment {window['index']} event falls outside its real source window.")
            if evidence not in observed or not start <= evidence <= end or evidence >= real_end:
                raise ValueError(f"Swap plan: segment {window['index']} {event['kind']} [{start},{end}) needs a sampled evidence_frame within its observed endpoints and real window; got {evidence}.")
            if not isinstance(event.get('description'), str) or not event['description'].strip():
                raise ValueError("Swap plan: event description is empty.")
            texts.append(event['description'])
        for text in texts:
            mentioned = {f'P{n}' for n in PERSON.findall(text)}
            if not mentioned <= set(present):
                raise ValueError(f"Swap plan: segment {window['index']} describes an absent person: {mentioned - set(present)}.")
            if re.search(r'<(?:Subject|Picture|Video)\s', text, re.I):
                raise ValueError("Swap plan: use P IDs in local descriptions; reference tags are assigned by the renderer.")
    return dict(version=PLAN_VERSION, fps=FPS, frame_count=frame_count, slots=slots,
                observed_frames=observed, cast=cast, segments=segments, edit_instructions=edits, unresolved=[])


def parse_plan(content, windows, slots, observed, frame_count):
    text = content.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    return validate_plan(json.loads(text), windows, slots, observed, frame_count)


def render_segment(plan, segment):
    present = set(segment['present_people'])
    people = {p['person_id']: p for p in plan['cast']}
    slots = plan['slots']
    definitions, retention, edits = [], [], []
    for person in plan['cast']:
        pid = person['person_id']
        if pid not in present:
            continue
        n = int(pid[1:])
        source, replacement = f'<Subject {2*n}>', f'<Subject {2*n-1}>'
        definitions.append(f"{source} is source person {pid}: {person['source_description']}")
        slot = person['reference_slot']
        if slot is None:
            retention.append(f'{source}: fully_preserved, including source appearance and performance.')
        else:
            picture = f'<Picture {slots.index(slot)+1}>'
            definitions.append(f"{replacement} is the replacement identity from {picture}: {person['replacement_description']}")
            edits.append(f'Replace {source} with {replacement} whenever {pid} is visible in this source window.')
            retention.extend([f'{source}: replaced only while present in the source.',
                              f'{replacement}: attribute_transfer from {picture}; inherit the source performance, expression, position and visibility.'])
    def bind(text):
        def substitute(match):
            person = people[f'P{match[1]}']
            n = int(match[1])
            return f'<Subject {2*n-1 if person["reference_slot"] is not None else 2*n}>'
        return PERSON.sub(substitute, text)
    definitions.append('<Video 1> is ONLY the current source window, not the entire film. It supplies motion, expressions, camera, lighting, scene changes, population and timing.')
    details = edits + plan['edit_instructions'] + [
        'Do not show the reference sheet or its background. Reference images supply identities, not additional people.',
        'Keep the source population and exact entrances, exits, occlusion and interactions at every frame. Never add an absent character or transfer identities between people.',
        bind(segment['description']),
        'Observed window opening: ' + bind(segment['start_state']),
        'Observed window ending: ' + bind(segment['end_state']),
    ]
    for event in segment['events']:
        start = (event['start_frame'] - segment['start_frame']) / FPS
        end = (event['end_frame'] - segment['start_frame']) / FPS
        details.append(f'At local {start:.3f}-{end:.3f}s: {bind(event["description"])}')
    details.extend([
        'Follow the exact source performance frame by frame, including expressions, gaze, blinks and mouth movements. Keep replacement identity consistent without freezing expression.',
        'Follow actual source changes in illumination, exposure, color, costume state, camera and environment; do not flatten them into constant conditions.',
        'This window is part of a continuous source edit. Do not add an opening, closing, cut, fade or exposure ramp at a processing boundary. Reproduce such changes only where they are present in this current source window.',
    ])
    if segment['head_trim']:
        details.append(f'The first {segment["head_trim"] / FPS:.3f}s are the previous segment overlap supplied by Motion Context. Continue its identities and motion according to the source; do not restart the action after the overlap.')
    tail = segment['end_frame'] - segment['start_frame'] - segment['head_trim'] - segment['deliver']
    if tail:
        details.append(f'The last {tail / FPS:.3f}s are repeated-source-frame padding outside delivery; hold the source tail naturally without inventing new events.')
    return '\n\n'.join([
        'subject_definitions:\n' + '\n'.join(definitions),
        'summary:\n[video editing + reference generation] ' + (' '.join(edits) if edits else 'Preserve this source window; no replacement target is present.'),
        'retention_analysis:\n' + '\n'.join(retention + ['<Video 1>: partially_preserved; apply only the requested edits.']),
        'detailed_description:\n' + '\n'.join(details),
        'overall_soundscape:\nPreserve the original source audio and synchronization; do not invent dialogue, music or sound from sampled stills.',
        'non_diegetic_music:\nN/A',
    ])


class H3SwapPromptPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = H3SwapPromptCloud.define_schema()
        schema.node_id = 'H3SwapPromptPlan'
        schema.display_name = '[yyp] H3 Swap Plan 全片人物与分段提示词'
        schema.description = '一次分析全片人物与各个 MC 窗口，固定人物编号，输出可校验的分段计划；配合 H3 Swap Segment Prompt 按段取词。'
        schema.inputs.extend([
            io.Int.Input('sample_length', default=141, min=5, max=362, tooltip='接 P1 的 length，必须与实际采样窗一致。'),
            io.Int.Input('context_length', default=22, min=5, max=56, tooltip='接 P1 的 overlap，与 MC 头锚一致。'),
            io.Int.Input('delivered_frames', default=481, min=5, max=100000, tooltip='接 P1 的 total_delivered，包含丢尾后的实际交付帧数。'),
            io.Float.Input('sample_interval', default=2.0, min=0.1, max=5.0, step=0.1, tooltip='每隔多少秒看一帧（24fps）。额外看各段首尾、头锚交界及交付末帧；短暂出入场可减到 0.5s。'),
        ])
        schema.outputs = [io.String.Output(display_name='plan'), io.String.Output(display_name='cast_and_windows')]
        return schema

    @classmethod
    def execute(cls, instruction, model, thinking, temperature, seed, sample_length, context_length,
                delivered_frames, sample_interval=2.0, reasoning_effort='medium', ref_images=None,
                source_frames=None, history_entry='none', api_key='', duration=0.0):
        refs = reference_images(ref_images)
        if source_frames is None or source_frames.ndim != 4:
            raise ValueError('Swap plan: connect the full 24fps source IMAGE batch.')
        frame_count = int(source_frames.shape[0])
        windows = source_windows(frame_count, sample_length, context_length, delivered_frames)
        observed = observation_indices(frame_count, windows, sample_interval)
        slots = [slot for slot, _ in refs]
        user_text = build_swap_user_text(instruction, len(refs), ['source frames listed below'], frame_count / FPS)
        user_text += '\nReference sockets: ' + json.dumps({slot: f'<Picture {i+1}>' for i, slot in enumerate(slots)})
        user_text += '\nExact MC windows: ' + json.dumps(windows)
        user_text += '\nOnly source frames below were observed; unresolved target identities must be reported.'
        images = [image_part(frame) for _, frame in refs] + [image_part(source_frames[index]) for index in observed]
        fingerprint = hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest()
        if history_entry != 'none':
            entry = find_history_entry(history_entry)
            if entry is None or entry.get('kind') != 'swap_plan' or entry.get('input') != instruction.strip():
                raise ValueError('Swap plan: select a matching segmented-swap history entry, not a single-clip prompt.')
            content = entry['output']
            previous = json.loads(content)
            if previous.get('frame_count') != frame_count or previous.get('slots') != slots or previous.get('observed_frames') != observed or previous.get('fingerprint') != fingerprint:
                raise ValueError('Swap plan: history source shape, sockets or observation interval differs; regenerate.')
        else:
            provider, model_id = parse_model_option(model)
            require_vision(provider, model_id, refs)
            user_content = [dict(type='text', text=user_text)]
            for i, (slot, frame) in enumerate(refs):
                user_content.extend([dict(type='text', text=f'<Picture {i+1}> = {slot}: replacement identity only'), images[i]])
            for image_index, index in enumerate(observed, len(refs)):
                memberships = [w['index'] for w in windows if w['start_frame'] <= index < w['end_frame']]
                user_content.extend([dict(type='text', text=f'SOURCE FRAME {index}, t={index / FPS:.3f}s, windows={memberships}'), images[image_index]])
            log(f'swap plan: {frame_count} source frames, {len(windows)} windows, {len(observed)} observations, interval={sample_interval:g}s')
            content = chat(provider, model_id, PLAN_SYSTEM, user_content, temperature=temperature, seed=seed,
                           thinking=thinking == 'enabled', effort_choice=reasoning_effort, api_key=api_key,
                           max_tokens=ENHANCE_MAX_TOKENS, on_text=make_progress_cb(cls.hidden.unique_id))
        plan = parse_plan(content, windows, slots, observed, frame_count)
        for slot, number in re.findall(r'\b(ref_?image_?\d+)\s*(?:替换|replaces?)\s*(?:(?:视频中)?的?)?\s*(?:人物\s*|P)(\d+)\b', instruction, re.I):
            slot = 'ref_image_' + re.search(r'\d+$', slot)[0]
            person = next((p for p in plan['cast'] if p['person_id'] == f'P{number}'), None)
            if person is None or person['reference_slot'] != slot:
                raise ValueError(f'Swap plan: requested {slot} -> P{number} mapping was not respected.')
        plan['instruction'] = instruction.strip()
        plan['fingerprint'] = fingerprint
        encoded = json.dumps(plan, ensure_ascii=False)
        audit = '\n\n'.join([f"{p['person_id']} -> {p['reference_slot'] or 'preserve'}: {p['source_description']}\nReplacement: {p['replacement_description']}" for p in plan['cast']]
                              + [f"Segment {s['index']}: source [{s['start_frame']},{s['end_frame']}), head={s['head_trim']}, deliver={s['deliver']}, present={','.join(s['present_people']) or 'empty'}\n{s['description']}\n" + '\n'.join(f"{e['kind']} [{e['start_frame']},{e['end_frame']}): {e['description']}" for e in s['events']) for s in plan['segments']])
        if history_entry == 'none':
            append_history(dict(ts=datetime.now().isoformat(timespec='seconds'), kind='swap_plan',
                                task_type='cloud/characterswap_segments', model=f'{provider}/{model_id}',
                                thinking=thinking, input=instruction.strip(), output=encoded))
        return io.NodeOutput(encoded, audit)


class H3SwapSegmentPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3SwapSegmentPrompt', display_name='[yyp] H3 Swap Segment Prompt 当前段换人提示词',
                         category=PROMPT_ENHANCE, inputs=[
                             io.String.Input('plan', multiline=True),
                             io.Int.Input('clip_index', default=1, min=1, max=9999),
                             io.Int.Input('start_frame', default=0, min=0, max=100000),
                             io.Int.Input('sample_length', default=141, min=5, max=362),
                             io.Int.Input('head_trim', default=0, min=0, max=56),
                             io.Int.Input('deliver', default=141, min=1, max=362),
                         ], outputs=[io.String.Output(display_name='prompt'), io.String.Output(display_name='segment_check')])

    @classmethod
    def execute(cls, plan, clip_index, start_frame, sample_length, head_trim, deliver):
        data = json.loads(plan)
        if data.get('version') != PLAN_VERSION or data.get('fps') != FPS:
            raise ValueError('Swap segment: unsupported plan version or frame rate.')
        if not 1 <= clip_index <= len(data['segments']):
            raise ValueError('Swap segment: clip_index exceeds the plan.')
        expected = dict(index=clip_index, start_frame=start_frame, end_frame=start_frame + sample_length,
                        head_trim=head_trim, deliver=deliver)
        segment = data['segments'][clip_index - 1]
        if any(segment.get(k) != v for k, v in expected.items()):
            raise ValueError('Swap segment: plan no longer matches the current MC window; regenerate the plan.')
        check = f"Segment {clip_index}: source [{start_frame},{start_frame + sample_length}), present={segment['present_people']}, head={head_trim}, deliver={deliver}"
        return io.NodeOutput(render_segment(data, segment), check)
