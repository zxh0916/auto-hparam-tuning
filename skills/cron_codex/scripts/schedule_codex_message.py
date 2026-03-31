#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import math
import shlex
import subprocess
import sys
import uuid
import os
from pathlib import Path

TIME_FORMATS = (
    '%Y-%m-%d %H:%M',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%dT%H:%M',
    '%Y-%m-%dT%H:%M:%S',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Schedule a payload to be sent to a Codex session later via host cron.'
    )
    target = parser.add_mutually_exclusive_group(required=False)
    target.add_argument('--session', help='Codex session id or thread name.')
    target.add_argument(
        '--last',
        action='store_true',
        help='Target the most recent recorded Codex session.',
    )
    schedule = parser.add_mutually_exclusive_group(required=True)
    schedule.add_argument(
        '--at',
        help="Local target timestamp. Examples: '2026-03-22 21:30', '2026-03-22T21:30:00'.",
    )
    schedule.add_argument(
        '--after-seconds',
        type=int,
        help='Delay in seconds from now before sending the payload.',
    )
    schedule.add_argument(
        '--after-minutes',
        type=int,
        help='Delay in minutes from now before sending the payload.',
    )
    parser.add_argument('--message', help='Message to send.')
    parser.add_argument('--payload', help='Alias of --message.')
    parser.add_argument(
        '--workdir',
        default='.',
        help='Working directory for the codex command. Defaults to the current directory.',
    )
    parser.add_argument(
        '--job-dir',
        default='.cron_codex_jobs',
        help='Directory used to store job metadata and logs.',
    )
    parser.add_argument(
        '--proxy-http',
        default=os.environ.get('HTTP_PROXY', 'http://127.0.0.1:7890'),
        help='HTTP proxy exported into the cron job.',
    )
    parser.add_argument(
        '--proxy-https',
        default=os.environ.get('HTTPS_PROXY', 'http://127.0.0.1:7890'),
        help='HTTPS proxy exported into the cron job.',
    )
    parser.add_argument(
        '--proxy-all',
        default=os.environ.get('ALL_PROXY', 'socks5h://127.0.0.1:7891'),
        help='ALL_PROXY exported into the cron job.',
    )
    parser.add_argument(
        '--proxy-no',
        default=os.environ.get('NO_PROXY', 'localhost,127.0.0.1,::1'),
        help='NO_PROXY exported into the cron job.',
    )
    parser.add_argument(
        '--node-bin',
        default=os.environ.get('CODEX_NODE_BIN', str(Path.home() / '.nvm' / 'versions' / 'node' / 'v24.14.0' / 'bin' / 'node')),
        help='Node binary used to launch the Codex CLI under cron.',
    )
    parser.add_argument(
        '--codex-js',
        default=os.environ.get('CODEX_JS', str(Path.home() / '.nvm' / 'versions' / 'node' / 'v24.14.0' / 'lib' / 'node_modules' / '@openai' / 'codex' / 'bin' / 'codex.js')),
        help='Path to the Codex CLI entrypoint JS used under cron.',
    )
    parser.add_argument(
        '--forward-all-user-ttys',
        action='store_true',
        help='Start a one-shot watcher that forwards the matching final answer to all writable terminals of the current user.',
    )
    parser.add_argument(
        '--forward-tty',
        action='append',
        default=[],
        help='Additional tty device to forward to. Can be repeated.',
    )
    parser.add_argument(
        '--bell',
        action='store_true',
        help='Emit a terminal bell before the forwarded final answer.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate inputs and print the resolved schedule without creating background jobs.',
    )
    return parser.parse_args()


def parse_timestamp(raw: str) -> dt.datetime:
    for fmt in TIME_FORMATS:
        try:
            return dt.datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(
        "Unsupported time format. Use 'YYYY-MM-DD HH:MM[:SS]' or 'YYYY-MM-DDTHH:MM[:SS]'."
    )


def ensure_future(target_time: dt.datetime) -> int:
    now = dt.datetime.now()
    delay = math.ceil((target_time - now).total_seconds())
    if delay <= 0:
        raise ValueError('Scheduled time must be in the future.')
    return delay


def resolve_target_time(args: argparse.Namespace) -> tuple[dt.datetime, int]:
    if args.at:
        target_time = parse_timestamp(args.at)
        return target_time, ensure_future(target_time)

    now = dt.datetime.now()
    if args.after_seconds is not None:
        if args.after_seconds <= 0:
            raise ValueError('--after-seconds must be greater than 0.')
        delay = args.after_seconds
    else:
        if args.after_minutes is None or args.after_minutes <= 0:
            raise ValueError('--after-minutes must be greater than 0.')
        delay = args.after_minutes * 60
    return now + dt.timedelta(seconds=delay), delay


def resolve_message(args: argparse.Namespace) -> str:
    if args.message and args.payload and args.message != args.payload:
        raise ValueError('--message and --payload must match when both are provided.')
    message = args.payload or args.message
    if not message:
        raise ValueError('One of --message or --payload is required.')
    return message


def resolve_session(args: argparse.Namespace) -> str | None:
    if args.last:
        return None
    if args.session:
        return args.session
    env_thread = os.environ.get('CODEX_THREAD_ID')
    if env_thread:
        return env_thread
    raise ValueError('Missing target session. Pass --session, --last, or export CODEX_THREAD_ID.')


def build_codex_command(args: argparse.Namespace, message: str) -> list[str]:
    command = [args.node_bin, args.codex_js, 'exec', '--skip-git-repo-check', '-C', str(Path(args.workdir).resolve()), 'resume']
    if args.last:
        command.append('--last')
    else:
        command.append(resolve_session(args))
    command.append(message)
    return command


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def build_runner_script(
    *,
    target_time: dt.datetime,
    workdir: Path,
    log_file: Path,
    command: list[str],
    marker: str,
    proxy_http: str,
    proxy_https: str,
    proxy_all: str,
    proxy_no: str,
) -> str:
    escaped_log = shlex.quote(str(log_file))
    escaped_command = ' '.join(shlex.quote(part) for part in command)
    target_epoch = math.ceil(target_time.timestamp())
    return (
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'cleanup() {\n'
        '  tmpfile="$(mktemp)"\n'
        '  if crontab -l >"$tmpfile" 2>/dev/null; then\n'
        f'    grep -v {shell_quote(marker)} "$tmpfile" | crontab -\n'
        '  else\n'
        '    : >"$tmpfile"\n'
        '  fi\n'
        '  rm -f "$tmpfile"\n'
        '}\n'
        'trap cleanup EXIT\n'
        f'export HTTP_PROXY={shell_quote(proxy_http)}\n'
        f'export HTTPS_PROXY={shell_quote(proxy_https)}\n'
        f'export ALL_PROXY={shell_quote(proxy_all)}\n'
        f'export NO_PROXY={shell_quote(proxy_no)}\n'
        'unset CODEX_SANDBOX_NETWORK_DISABLED || true\n'
        f'target_epoch={target_epoch}\n'
        'now_epoch="$(date +%s)"\n'
        'if [ "$target_epoch" -gt "$now_epoch" ]; then\n'
        '  sleep "$((target_epoch - now_epoch))"\n'
        'fi\n'
        '{\n'
        '  echo "[$(date \'+%Y-%m-%d %H:%M:%S %z\')] cron start"\n'
        f'  echo "marker={marker}"\n'
        f'  exec {escaped_command}\n'
        '  echo "[$(date \'+%Y-%m-%d %H:%M:%S %z\')] cron end"\n'
        f'}} >> {escaped_log} 2>&1\n'
    )


def start_forward_watcher(args: argparse.Namespace, job_dir: Path, job_id: str) -> dict | None:
    if not args.forward_all_user_ttys and not args.forward_tty:
        args.forward_all_user_ttys = True

    if args.last:
        raise ValueError('Forwarding requires an explicit --session so the watcher can bind to a stable session id.')

    script_path = Path(__file__).resolve().parent / 'watch_codex_session.py'
    log_file = job_dir / f'{job_id}.watch.log'
    command = [
        sys.executable,
        str(script_path),
        '--session-id',
        resolve_session(args),
        '--trigger-exact',
        resolve_message(args),
        '--once',
    ]
    if args.forward_all_user_ttys:
        command.append('--all-user-ttys')
    for tty in args.forward_tty:
        command.extend(['--tty', tty])
    if args.bell:
        command.append('--bell')

    with log_file.open('a', encoding='utf-8') as handle:
        process = subprocess.Popen(
            ['nohup', *command],
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    return {
        'watcher_pid': process.pid,
        'watcher_command': command,
        'watcher_log_file': str(log_file),
    }


def install_crontab_entry(runner_file: Path, marker: str, target_time: dt.datetime) -> str:
    cron_expr = f'{target_time.minute} {target_time.hour} {target_time.day} {target_time.month} *'
    line = f'{cron_expr} {shlex.quote(str(runner_file))} # {marker}\n'

    existing = ''
    result = subprocess.run(
        ['crontab', '-l'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        existing = result.stdout
    elif result.returncode != 1:
        raise RuntimeError(result.stderr.strip() or 'Failed to read current crontab.')

    filtered_lines = [
        item for item in existing.splitlines()
        if marker not in item
    ]
    filtered_lines.append(line.rstrip('\n'))
    new_content = '\n'.join(filtered_lines) + '\n'
    subprocess.run(['crontab', '-'], input=new_content, text=True, check=True)
    return cron_expr


def main() -> int:
    args = parse_args()

    try:
        target_time, delay_seconds = resolve_target_time(args)
        message = resolve_message(args)
        session = resolve_session(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    workdir = Path(args.workdir).resolve()
    job_dir = Path(args.job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex[:12]
    log_file = job_dir / f'{job_id}.log'
    runner_file = job_dir / f'{job_id}.sh'
    meta_file = job_dir / f'{job_id}.json'
    marker = f'CODEX_CRON_{job_id}'

    command = build_codex_command(args, message)
    payload = {
        'job_id': job_id,
        'scheduled_for': target_time.isoformat(sep=' '),
        'delay_seconds': delay_seconds,
        'workdir': str(workdir),
        'target': 'last' if args.last else session,
        'message': message,
        'payload': message,
        'command': command,
        'log_file': str(log_file),
        'runner_file': str(runner_file),
        'marker': marker,
        'forward_all_user_ttys': args.forward_all_user_ttys,
        'forward_tty': args.forward_tty,
        'bell': args.bell,
    }

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0

    runner_file.write_text(
        build_runner_script(
            target_time=target_time,
            workdir=workdir,
            log_file=log_file,
            command=command,
            marker=marker,
            proxy_http=args.proxy_http,
            proxy_https=args.proxy_https,
            proxy_all=args.proxy_all,
            proxy_no=args.proxy_no,
        ),
        encoding='utf-8',
    )
    runner_file.chmod(0o755)

    try:
        watcher_info = start_forward_watcher(args, job_dir, job_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        cron_expr = install_crontab_entry(runner_file, marker, target_time)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if watcher_info:
        payload.update(watcher_info)
    payload['cron_expr'] = cron_expr
    meta_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
