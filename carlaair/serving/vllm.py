"""Launch a foreground OpenAI-compatible vLLM service for local evaluation."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--served-model-name', default='qwen3.5-4b')
    parser.add_argument('--port', default=8114, type=int)
    parser.add_argument('--gpu-memory-utilization', default=.30, type=float)
    parser.add_argument('--adapter', action='append', default=[], metavar='NAME=DIRECTORY')
    parser.add_argument('--dry-run', action='store_true', help='validate local paths and print the command')
    a = parser.parse_args()
    if not a.model_dir.is_dir():
        parser.error('model directory does not exist')
    args = ['vllm', 'serve', str(a.model_dir.resolve()), '--served-model-name', a.served_model_name,
            '--host', '127.0.0.1', '--port', str(a.port), '--dtype', 'bfloat16', '--max-model-len', '16384',
            '--max-num-seqs', '16', '--gpu-memory-utilization', str(a.gpu_memory_utilization), '--enforce-eager']
    if a.adapter:
        for item in a.adapter:
            name, sep, folder = item.partition('=')
            if not name or not sep or not (Path(folder) / 'adapter_config.json').is_file():
                parser.error(f'invalid LoRA adapter: {item}')
        args.extend(['--enable-lora', '--max-lora-rank', '8', '--lora-modules', *a.adapter])
    if a.dry_run:
        import shlex
        print(shlex.join(args))
    else:
        os.execvp(args[0], args)


if __name__ == '__main__':
    main()
