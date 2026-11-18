"""Launch a foreground OpenAI-compatible vLLM service for local evaluation."""
import argparse
import json
import os
from pathlib import Path


def config_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'invalid or missing {path}: {exc}') from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f'expected a nonempty JSON configuration at {path}')
    return value


def has_weights(path):
    return path.is_file() and path.stat().st_size > 0


def base_directory(value):
    if not value.strip():
        raise ValueError('--model-dir must be a nonempty directory path')
    folder = Path(value).resolve()
    if not config_json(folder / 'config.json').get('model_type'):
        raise ValueError(f'missing model_type in {folder / "config.json"}')
    for name in ('model.safetensors', 'pytorch_model.bin'):
        index = folder / (name + '.index.json')
        if index.is_file():
            mapping = config_json(index).get('weight_map')
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f'missing weight map in {index}')
            for shard in set(mapping.values()):
                if not isinstance(shard, str) or Path(shard).name != shard or not has_weights(folder / shard):
                    raise ValueError(f'missing or invalid indexed weight shard: {shard!r}')
            return folder
        if has_weights(folder / name):
            return folder
    raise ValueError(f'missing model weights in {folder}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--served-model-name', default='qwen3.5-4b')
    parser.add_argument('--port', default=8114, type=int)
    parser.add_argument('--gpu-memory-utilization', default=.30, type=float)
    parser.add_argument('--adapter', action='append', default=[], metavar='NAME=DIRECTORY')
    parser.add_argument('--dry-run', action='store_true', help='validate local files and print the command')
    a = parser.parse_args()
    adapters = {}
    try:
        base = base_directory(a.model_dir)
        for item in a.adapter:
            name, sep, value = item.partition('=')
            if not name.strip() or name in adapters or name == a.served_model_name or not sep or not value.strip():
                raise ValueError(f'invalid or duplicate LoRA adapter: {item}')
            folder = Path(value).resolve()
            cfg = config_json(folder / 'adapter_config.json')
            if str(cfg.get('peft_type')).upper() != 'LORA' or not any(
                    has_weights(folder / weight) for weight in ('adapter_model.safetensors', 'adapter_model.bin')):
                raise ValueError(f'invalid LoRA configuration or missing adapter weights in {folder}')
            adapters[name] = folder
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    args = ['vllm', 'serve', str(base), '--served-model-name', a.served_model_name,
            '--host', '127.0.0.1', '--port', str(a.port), '--dtype', 'bfloat16', '--max-model-len', '16384',
            '--max-num-seqs', '16', '--gpu-memory-utilization', str(a.gpu_memory_utilization), '--enforce-eager']
    if adapters:
        count = str(len(adapters))
        args.extend(['--enable-lora', '--max-lora-rank', '8', '--max-loras', count, '--max-cpu-loras', count,
                     '--lora-modules', *(f'{name}={folder}' for name, folder in adapters.items())])
    if a.dry_run:
        import shlex
        print(shlex.join(args))
    else:
        os.execvp(args[0], args)


if __name__ == '__main__':
    main()
