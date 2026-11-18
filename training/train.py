"""Multimodal LoRA training on a local Qwen3.5-4B checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_BIAS = 'none'
LORA_LAYER_INDICES = (3, 7, 11, 15, 19, 23, 27, 31)
LORA_PROJECTIONS = ('q_proj', 'v_proj')
DEFAULT_SEED = 20261027
DEFAULT_MAX_SEQUENCE_TOKENS = 5120
IGNORE_INDEX = -100
GRAD_CLIP = 1.0
GRAD_ACCUMULATION = 2
CHAT_TEMPLATE_KWARGS = {'enable_thinking': False}
_IMAGE_REL_RE = re.compile(r'images/[0-9a-f]{64}\.png\Z')
_LANGUAGE_LAYER_RE = re.compile(r'(?:^|\.)language_model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)$')


class TrainError(RuntimeError):
    """Base class for refusals and hard training failures."""


class TrainGateError(TrainError):
    """Pre-flight or post-training gate refused to continue."""


class TrainDependencyError(TrainError):
    """A lazy heavy dependency (torch/transformers/peft) is unavailable."""


class TrainDataError(TrainError):
    """Training rows/templates/masks are invalid; never guessed around."""


class NonFiniteLoss(TrainError):
    """A loss or gradient norm became NaN/inf; stop, never retry silently."""


class SampleTooLong(TrainDataError):
    """One record exceeds the token budget; quarantine it, never truncate."""

    def __init__(self, token_count, max_tokens):
        self.token_count = int(token_count)
        self.max_tokens = int(max_tokens)
        TrainDataError.__init__(
            self, 'sample exceeds token budget: %d > %d' % (self.token_count, self.max_tokens))


def _require(condition, message):
    if not condition:
        raise TrainDataError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _image_digest(relative):
    match = _IMAGE_REL_RE.match(relative)
    if match is None:
        raise TrainDataError(
            'unsafe/noncanonical training image path: %r (expected images/<sha256>.png)'
            % (relative,))
    return relative[len('images/'):-len('.png')]


def _image_part(part, prefix):
    _require(isinstance(part, dict), prefix + 'image content part must be an object')
    path = part.get('path')
    _require(isinstance(path, str) and bool(path.strip()), prefix + 'image part has no path')
    digest = _image_digest(path)
    declared = part.get('sha256')
    if declared is not None:
        _require(declared == digest, prefix + 'image part SHA does not match its path')
    return {'type': 'image', 'path': path, 'sha256': digest}


def _user_parts(content, row, prefix):
    """Normalize the user content list; image parts stay content addressed."""
    images = row.get('images')
    if isinstance(content, str) and content.strip():
        # Legacy pilot rows carried the pictures in ``images`` and one text blob.
        _require(isinstance(images, list) and bool(images),
                 prefix + 'missing policy images for the string user content form')
        parts = [{'type': 'image', 'path': _image_part({'path': item}, prefix)['path']}
                 for item in images]
        parts.append({'type': 'text', 'text': content})
        return parts
    _require(isinstance(content, list) and bool(content),
             prefix + 'user content must be a non-empty text/image list')
    parts = []
    for part in content:
        _require(isinstance(part, dict), prefix + 'user content part must be an object')
        kind = part.get('type')
        if kind == 'image':
            parts.append(_image_part(part, prefix))
        elif kind == 'text':
            _require(isinstance(part.get('text'), str) and bool(part['text'].strip()),
                     prefix + 'empty text content part')
            parts.append({'type': 'text', 'text': part['text']})
        else:
            raise TrainDataError(prefix + 'unsupported user content part type: %r' % (kind,))
    _require(any(part['type'] == 'text' for part in parts),
             prefix + 'user content has no text part')
    _require(any(part['type'] == 'image' for part in parts),
             prefix + 'missing policy images in user content')
    row_images = row.get('images')
    if row_images is not None:
        _require(isinstance(row_images, list) and
                 all(isinstance(item, str) for item in row_images),
                 prefix + 'image list must be a list of relative paths')
        _require(row_images == [part['path'] for part in parts if part['type'] == 'image'],
                 prefix + 'image list does not match the user image parts')
    return parts


def validate_training_row(row, index=None):
    """Validate the system, multimodal user prompt and JSON assistant answer."""
    prefix = 'row %d: ' % index if index is not None else ''
    _require(isinstance(row, dict), prefix + 'training row must be a JSON object')
    messages = row.get('messages')
    _require(isinstance(messages, list) and len(messages) == 3,
             prefix + 'expected exactly system/user/assistant messages')
    for position, role in enumerate(('system', 'user', 'assistant')):
        message = messages[position]
        _require(isinstance(message, dict) and message.get('role') == role,
                 prefix + 'expected %s message at position %d' % (role, position))
    system_content = messages[0].get('content')
    _require(isinstance(system_content, str) and bool(system_content.strip()),
             prefix + 'empty system prompt')
    assistant_content = messages[2].get('content')
    _require(isinstance(assistant_content, str) and bool(assistant_content.strip()),
             prefix + 'empty assistant target')
    try:
        target = json.loads(assistant_content)
    except json.JSONDecodeError as exc:
        raise TrainDataError(prefix + 'assistant target is not JSON: %s' % exc)
    _require(isinstance(target, dict), prefix + 'assistant target must be a JSON object')
    if 'role' in target:
        _require(target['role'] in ('UAV', 'UGV'),
                 prefix + 'assistant target role must be UAV or UGV')
    if 'action' in target:
        _require(isinstance(target['action'], str) and bool(target['action'].strip()),
                 prefix + 'assistant target action must be a non-empty string')
    _user_parts(messages[1].get('content'), row, prefix)
    return row


def row_to_messages(row):
    """Record dict -> chat-template messages; image parts keep their path."""
    validate_training_row(row)
    system, user, assistant = row['messages']
    content = []
    for part in _user_parts(user.get('content'), row, ''):
        if part['type'] == 'text':
            content.append({'type': 'text', 'text': part['text']})
        else:
            content.append({'type': 'image', 'path': part['path']})
    return [
        {'role': 'system', 'content': system['content']},
        {'role': 'user', 'content': content},
        {'role': 'assistant', 'content': assistant['content']},
    ]


def load_row_image(data_root, relative, thumbnail=None):
    """Decode a content-addressed RGB image at its supplied resolution."""
    root = Path(data_root).resolve()
    digest = _image_digest(relative)
    path = (root / relative).resolve()
    _require(path.is_relative_to(root) and path.is_file(),
             'missing/unsafe training image: %s' % relative)
    _require(sha256_bytes(path.read_bytes()) == digest, 'training image SHA mismatch: %s' % relative)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - server dependency
        raise TrainDependencyError('Pillow is required to read training images: %s' % exc)
    with Image.open(path) as image:
        converted = image.convert('RGB')
        if thumbnail is not None:
            converted.thumbnail(tuple(thumbnail), Image.Resampling.LANCZOS)
        _require(converted.width > 0 and converted.height > 0,
                 'empty training image: %s' % relative)
        return converted


def render_messages(messages, *, data_root=None, load_image=None, thumbnail=None):
    """Place decoded images in their recorded prompt slots."""
    rendered = []
    for message in messages:
        if message.get('role') != 'user':
            rendered.append({'role': message['role'], 'content': message['content']})
            continue
        parts = []
        for part in message['content']:
            if part['type'] == 'text':
                parts.append({'type': 'text', 'text': part['text']})
                continue
            if load_image is not None:
                image = load_image(part['path'])
            else:
                if data_root is None:
                    raise TrainDataError('data_root is required to load training images')
                image = load_row_image(data_root, part['path'], thumbnail)
            parts.append({'type': 'image', 'image': image})
        rendered.append({'role': 'user', 'content': parts})
    return rendered


def _ids_list(value, label='input_ids'):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or not value:
        raise TrainDataError('%s must be a non-empty token list, got %r' % (label, value))
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    tokens = []
    for token in value:
        if type(token) is not int:
            raise TrainDataError('%s contains a non-integer token: %r' % (label, token))
        tokens.append(token)
    return tokens


def _apply_chat_template(processor, messages, *, add_generation_prompt):
    try:
        return processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors='pt',
            add_generation_prompt=add_generation_prompt, **CHAT_TEMPLATE_KWARGS)
    except TypeError as exc:
        raise TrainDataError(
            'chat template rejected enable_thinking=False; the frozen eval config '
            'disables thinking, so training refuses to guess: %s' % exc)


def build_training_labels(prompt_ids, full_ids, attention_mask=None):
    """Assistant-only labels with the exact generation prefix.

    Returns ``(labels, supervised_tokens)``.  The prompt/generation-prefix
    tokens and every masked/padding position are ``-100``; if the assistant
    template prefix does not line up token-for-token, or no label survives, the
    sample is hard-rejected instead of being shifted by a guessed offset.
    """
    prompt_ids = [int(token) for token in prompt_ids]
    full_ids = [int(token) for token in full_ids]
    if not prompt_ids:
        raise TrainDataError('empty prompt token sequence')
    if len(prompt_ids) > len(full_ids):
        raise TrainDataError('assistant template prefix mismatch: prompt is longer than full')
    for position in range(len(prompt_ids)):
        if full_ids[position] != prompt_ids[position]:
            raise TrainDataError(
                'assistant template prefix mismatch at token %d' % position)
    labels = [IGNORE_INDEX] * len(full_ids)
    labels[len(prompt_ids):] = list(full_ids[len(prompt_ids):])
    if attention_mask is not None:
        if hasattr(attention_mask, 'tolist'):
            attention_mask = attention_mask.tolist()
        if len(attention_mask) == 1 and isinstance(attention_mask[0], (list, tuple)):
            attention_mask = attention_mask[0]
        attention_mask = [int(value) for value in attention_mask]
        if len(attention_mask) != len(full_ids):
            raise TrainDataError('attention mask length does not match the full sequence')
        for position, keep in enumerate(attention_mask):
            if keep == 0:
                labels[position] = IGNORE_INDEX
    supervised = sum(1 for label in labels if label != IGNORE_INDEX)
    if supervised <= 0:
        raise TrainDataError('no supervised assistant tokens')
    return labels, supervised


def within_token_budget(token_count, max_tokens=DEFAULT_MAX_SEQUENCE_TOKENS):
    """Pure budget predicate; samples are quarantined, never silently cut."""
    if type(token_count) is not int or token_count < 0:
        raise TrainDataError('token_count must be a non-negative integer, got %r' % (token_count,))
    if type(max_tokens) is not int or max_tokens < 1:
        raise TrainDataError('max_tokens must be a positive integer, got %r' % (max_tokens,))
    return token_count <= max_tokens


def encode_training_example(processor, row, *, data_root=None, max_tokens=DEFAULT_MAX_SEQUENCE_TOKENS,
                            thumbnail=None, load_image=None):
    """Tokenize without truncation and supervise the assistant suffix."""
    messages = row_to_messages(row)
    rendered = render_messages(messages, data_root=data_root, load_image=load_image,
                               thumbnail=thumbnail)
    prompt = _apply_chat_template(processor, rendered[:-1], add_generation_prompt=True)
    full = _apply_chat_template(processor, rendered, add_generation_prompt=False)
    prompt_ids = _ids_list(prompt.get('input_ids'), 'prompt input_ids')
    full_ids = _ids_list(full.get('input_ids'), 'full input_ids')
    token_count = len(full_ids)
    if not within_token_budget(token_count, max_tokens):
        raise SampleTooLong(token_count, max_tokens)
    attention_mask = full.get('attention_mask')
    attention_mask = _ids_list(attention_mask, 'attention_mask') if attention_mask is not None else None
    labels, supervised = build_training_labels(prompt_ids, full_ids, attention_mask)
    if attention_mask is None:
        attention_mask = [1] * token_count
    model_inputs = dict(full)
    model_inputs['attention_mask'] = attention_mask
    return {
        'model_inputs': model_inputs,
        'labels': labels,
        'supervised_tokens': supervised,
        'token_count': token_count,
        'prompt_tokens': len(prompt_ids),
    }


def prepare_training_examples(processor, rows, *, data_root=None,
                              max_tokens=DEFAULT_MAX_SEQUENCE_TOKENS,
                              thumbnail=None, load_image=None):
    """Tokenize every row once; oversize rows are quarantined with a reason."""
    prepared = []
    skipped = []
    for index, row in enumerate(rows):
        try:
            example = encode_training_example(
                processor, row, data_root=data_root, max_tokens=max_tokens,
                thumbnail=thumbnail, load_image=load_image)
        except SampleTooLong as exc:
            skipped.append({'row_index': index, 'reason': 'sequence_too_long',
                            'tokens': exc.token_count, 'max_tokens': exc.max_tokens})
            continue
        prepared.append({'row_index': index,
                         'row': row,
                         'token_count': example['token_count'],
                         'prompt_tokens': example['prompt_tokens'],
                         'supervised_tokens': example['supervised_tokens']})
    return prepared, skipped


def select_adapter_targets(named_modules, *, is_linear=None,
                           layers=LORA_LAYER_INDICES, projections=LORA_PROJECTIONS):
    """Select the frozen language-attention q/v modules; require ALL of them.

    A silently empty or partial match is a refusal, not a smaller adapter: the
    counts must be exactly ``len(layers) * len(projections)`` (16 by default).
    """
    layers = tuple(int(layer) for layer in layers)
    projections = tuple(str(projection) for projection in projections)
    expected_keys = {(layer, projection) for layer in layers for projection in projections}
    found = {}
    for name, module in named_modules:
        match = _LANGUAGE_LAYER_RE.search(name)
        if match is None:
            continue
        key = (int(match.group(1)), match.group(2))
        if key not in expected_keys:
            continue
        if is_linear is not None and not is_linear(module):
            continue
        found.setdefault(key, name)
    missing = sorted(expected_keys - set(found))
    if missing or len(found) != len(expected_keys):
        detail = ', '.join('layer %d %s' % key for key in missing) or 'count mismatch'
        raise TrainGateError(
            'LoRA module selection matched %d of %d required language_model modules '
            '(%s); refusing to train a partial adapter'
            % (len(found), len(expected_keys), detail))
    return [found[(layer, projection)] for layer in layers for projection in projections]


def lora_config_kwargs(target_modules):
    """Frozen LoRA hyper-parameters for ``peft.LoraConfig``."""
    targets = list(target_modules)
    expected = len(LORA_LAYER_INDICES) * len(LORA_PROJECTIONS)
    if len(targets) != expected:
        raise TrainGateError(
            'expected exactly %d LoRA target modules, got %d' % (expected, len(targets)))
    return {'r': LORA_R, 'lora_alpha': LORA_ALPHA, 'lora_dropout': LORA_DROPOUT,
            'bias': LORA_BIAS, 'target_modules': targets, 'task_type': 'CAUSAL_LM'}


def check_finite(value, label='loss'):
    number = float(value)
    if not math.isfinite(number):
        raise NonFiniteLoss('%s is not finite: %r' % (label, number))
    return number


def epoch_order(size, epoch, seed=DEFAULT_SEED):
    total = int(size)
    if total < 0:
        raise TrainDataError('cannot order a negative number of samples')
    order = list(range(total))
    random.Random('%d:%d' % (int(seed), int(epoch))).shuffle(order)
    return order


def _to_model_batch(torch, model_inputs, labels, device='cuda:0', dtype=None):
    dtype = dtype or torch.bfloat16
    batch = {}
    for name, value in model_inputs.items():
        tensor = value if hasattr(value, 'to') else torch.as_tensor(value)
        # Per-modality tensors (pixel values / grid) already carry their own leading
        # dimension; every sequence-aligned tensor must be shaped (1, seq) or the
        # multimodal RoPE path in Qwen3.5 receives a rank-2 token-type slice.
        if (name not in ('pixel_values', 'image_grid_thw', 'pixel_values_videos',
                        'video_grid_thw')
                and hasattr(tensor, 'dim') and hasattr(tensor, 'unsqueeze')
                and tensor.dim() == 1):
            tensor = tensor.unsqueeze(0)
        if torch.is_floating_point(tensor):
            tensor = tensor.to(device, dtype=dtype)
        else:
            tensor = tensor.to(device)
        batch[name] = tensor
    batch['labels'] = torch.tensor([list(labels)], dtype=torch.long, device=device)
    return batch


def _optimization_step(torch, model, processor, group, *, data_root, max_tokens, thumbnail,
                       optimizer, parameters):
    optimizer.zero_grad(set_to_none=True)
    weighted_total = None
    token_total = 0
    micro_losses = []
    for entry in group:
        example = encode_training_example(
            processor, entry['row'], data_root=data_root, max_tokens=max_tokens,
            thumbnail=thumbnail)
        labels = example['labels']
        batch = _to_model_batch(torch, example['model_inputs'], labels)
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None:
            raise TrainError('model returned no loss; labels were not forwarded')
        micro_loss = check_finite(loss, 'microbatch loss')
        supervised = int((batch['labels'][:, 1:] != IGNORE_INDEX).sum())
        if supervised != example['supervised_tokens']:
            raise TrainGateError(
                'supervised token count drifted between encode and forward: %d != %d'
                % (supervised, example['supervised_tokens']))
        if supervised <= 0:
            raise TrainDataError('no supervised assistant tokens in the microbatch')
        weighted = loss * supervised
        weighted_total = weighted if weighted_total is None else weighted_total + weighted
        token_total += supervised
        micro_losses.append(micro_loss)
    normalized = weighted_total / token_total
    loss_value = check_finite(normalized, 'token-normalized loss')
    normalized.backward()
    grad_norm = check_finite(torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP),
                             'gradient norm')
    optimizer.step()
    return {'loss': loss_value, 'micro_losses': micro_losses,
            'supervised_tokens': token_total, 'grad_norm': grad_norm}


def _load_base_model(stack, base):
    model = stack['Qwen3_5ForConditionalGeneration'].from_pretrained(
        str(base), local_files_only=True, dtype=stack['torch'].bfloat16,
        attn_implementation='sdpa', device_map={'': 0})
    model.config.use_cache = False
    return model


def adapter_wrapped_targets(named_modules, targets):
    """Return the distinct adapter parents PEFT actually wrapped.

    PEFT 0.21 exposes both ``...q_proj.lora_A`` (the module) and
    ``...q_proj.lora_A.default`` (its adapter), so counting raw matches doubled
    every target.  Parent names are compared after stripping PEFT's own wrapper
    prefix; every selected target must be present and nothing extra.
    """
    targets = list(targets)
    wrapped = {name.rsplit('.lora_A', 1)[0] for name, _ in named_modules
               if '.lora_A' in name}
    normalized = {name[len('base_model.model.'):] if name.startswith('base_model.model.') else name
                  for name in wrapped}
    return normalized, set(targets)


def load_training_rows(data_path):
    """Load and validate a training JSONL export; returns ``(rows, file_sha256)``."""
    data = Path(data_path)
    if not data.is_file():
        raise TrainDataError('training data file not found: %s' % data)
    try:
        raw = data.read_bytes()
    except OSError as exc:
        raise TrainDataError('cannot read training data %s: %s' % (data, exc))
    rows = []
    for number, line in enumerate(raw.decode('utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainDataError('training data line %d is not JSON: %s' % (number, exc))
        rows.append(row)
    if not rows:
        raise TrainDataError('training data has no rows: %s' % data)
    for index, row in enumerate(rows):
        validate_training_row(row, index=index)
    return rows, sha256_bytes(raw)


def train(options):
    from importlib.metadata import version
    try:
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from peft import LoraConfig, PeftModel, get_peft_model
        import accelerate  # required by the single-GPU device map
    except ImportError as exc:
        raise TrainDependencyError(f'Install the train extra and a CUDA PyTorch build: {exc}') from exc
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise TrainGateError('This recipe requires a CUDA device with bfloat16 support')
    if options.out.exists():
        raise TrainGateError(f'Choose a new output directory: {options.out}')
    rows, digest = load_training_rows(options.data)
    data_root = options.data.resolve().parent
    random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    processor = AutoProcessor.from_pretrained(str(options.base), local_files_only=True)
    examples, skipped = prepare_training_examples(processor, rows, data_root=data_root,
                                                  max_tokens=options.max_sequence_tokens)
    if not examples:
        raise TrainDataError('No rows passed the sequence length filter')
    options.out.mkdir(parents=True, exist_ok=False)
    (options.out / 'tokenization.json').write_text(json.dumps({
        'skipped': skipped, 'admitted': [{k: v for k, v in entry.items() if k != 'row'}
                                          for entry in examples]}), encoding='utf-8')
    model = _load_base_model({'torch': torch, 'Qwen3_5ForConditionalGeneration':
                              Qwen3_5ForConditionalGeneration}, options.base)
    targets = select_adapter_targets(model.named_modules(),
                                    is_linear=lambda m: isinstance(m, torch.nn.Linear))
    model = get_peft_model(model, LoraConfig(**lora_config_kwargs(targets)))
    actual, expected = adapter_wrapped_targets(model.named_modules(), targets)
    if actual != expected:
        raise TrainGateError(f'PEFT target mismatch: {actual ^ expected}')
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
    order = epoch_order(len(examples), 0, options.seed)
    samples_consumed = steps = supervised_tokens = 0
    with (options.out / 'steps.jsonl').open('x', encoding='utf-8') as log:
        for start in range(0, len(order), GRAD_ACCUMULATION):
            if options.max_steps is not None and steps >= options.max_steps:
                break
            batch = [examples[i] for i in order[start:start + GRAD_ACCUMULATION]]
            metrics = _optimization_step(torch, model, processor, batch, data_root=data_root,
                                         max_tokens=options.max_sequence_tokens, thumbnail=None,
                                         optimizer=optimizer, parameters=parameters)
            steps += 1
            samples_consumed += len(batch)
            supervised_tokens += metrics['supervised_tokens']
            log.write(json.dumps(dict(metrics, step=steps,
                                      row_indices=[x['row_index'] for x in batch])) + '\n')
            log.flush()
    model.save_pretrained(options.out / 'adapter', safe_serialization=True)
    processor.save_pretrained(options.out / 'processor')
    model.eval()
    probe = encode_training_example(processor, examples[0]['row'], data_root=data_root,
                                    max_tokens=options.max_sequence_tokens)
    batch = _to_model_batch(torch, probe['model_inputs'], probe['labels'])
    with torch.no_grad():
        before = check_finite(model(**batch).loss, 'pre-reload loss')
        base_model = model.unload()
        model = PeftModel.from_pretrained(base_model, options.out / 'adapter', is_trainable=False)
        model.eval()
        after = check_finite(model(**batch).loss, 'reloaded loss')
    if abs(after - before) > 1e-5:
        raise TrainGateError(f'Adapter reload loss mismatch: {before} vs {after}')
    result = {'supplied': len(rows), 'admitted': len(examples), 'skipped': len(skipped),
              'data_sha256': digest, 'samples_consumed': samples_consumed, 'steps': steps,
              'supervised_tokens': supervised_tokens, 'seed': options.seed, 'target_modules': targets,
              'max_sequence_tokens': options.max_sequence_tokens, 'max_steps': options.max_steps,
              'trainable_parameters': sum(p.numel() for p in parameters),
              'base_config_sha256': sha256_bytes((options.base / 'config.json').read_bytes()),
              'dependencies': {name: version(name) for name in ('torch','transformers','peft','accelerate','Pillow')},
              'reload_loss_before': before, 'reload_loss_after': after}
    (options.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Mixed or single-role JSONL with images/ alongside')
    parser.add_argument('--base', type=Path, required=True, help='Local Qwen3.5-4B model and processor directory')
    parser.add_argument('--out', type=Path, required=True, help='New directory for adapter, processor and receipts')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max-sequence-tokens', type=int, default=DEFAULT_MAX_SEQUENCE_TOKENS)
    parser.add_argument('--max-steps', type=int, help='Optional optimizer step cap within the single epoch')
    options = parser.parse_args(argv)
    if options.max_sequence_tokens < 1 or (options.max_steps is not None and options.max_steps < 1):
        parser.error('token and step limits must be positive')
    try:
        result = train(options)
    except (OSError, TrainError) as exc:
        parser.exit(1, f'train: {exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
