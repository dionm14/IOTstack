#!/usr/bin/env python3
"""
warnhits.py - turn a noisy CMake/GCC build log into a deduplicated warning hit list.

A warning in a header is printed once for every translation unit that includes
it, so a log with 40,000 "warning:" lines may only hold a few hundred distinct
problems. This collapses identical warnings (same file, line, column, flag and
message), counts how many times each one fired, and ranks them so the ones
with the biggest blast radius come first.

Understands:
  * GCC / Clang diagnostics    path:line[:col]: warning: message [-Wflag]
  * -Werror'd warnings          path:line:col: error: message [-Werror=flag]   (with --errors)
  * tool warnings               /usr/bin/ld: warning: ...   cc1plus: warning: ...   ninja: warning: ...
  * CMake configure warnings    CMake Warning (dev) at CMakeLists.txt:12 (project):
  * colour / hyperlink escapes  from -fdiagnostics-color / -fdiagnostics-urls
  * Make (-j) and Ninja logs, gzipped logs, stdin

Python 3.3+, standard library only.
"""

import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
from collections import Counter

EXAMPLES = """
examples:
  %(prog)s build.log
  %(prog)s build.log --root /home/matt/src/proj --exclude third_party/
  %(prog)s build.log --flag conversion --flag sign-conversion --top 50
  %(prog)s build.log --loose                       # fold template-instantiation variants
  %(prog)s build.log --format csv -o warnings.csv  # for a spreadsheet / divvying up work
  %(prog)s build.log --format gcc -o unique.log    # step through in vim (:cfile) or an IDE
  make -j8 2>&1 | %(prog)s -
"""

# ANSI escapes: OSC hyperlinks (-fdiagnostics-urls), CSI colour / erase-line codes, and the
# "]8;;URL<BEL>" remnant Ninja leaves behind when it strips only the ESC byte from a hyperlink.
ANSI_RE = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)'
                     r'|\]8;[^\x07\x1b]*\x07'
                     r'|\x1b\[[0-?]*[ -/]*[@-~]')

# path:line[:col]: warning|error|fatal error: message
DIAG_RE = re.compile(
    r'(?<!\S)(?P<file>[^\s:]+):(?P<line>\d+):(?:(?P<col>\d+):)?\s+'
    r'(?P<kind>warning|error|fatal error):\s+(?P<msg>.*)$')

# tool: warning: message   (ld, cc1plus, collect2, make, ninja, main.c:(.text+0x15), ...)
TOOL_RE = re.compile(
    r'^(?P<file>\S.*?):\s+(?P<kind>[Ww]arning|[Ee]rror|fatal error):\s+(?P<msg>.*)$')

# Trailing [-Wflag] or [-Werror=flag] on a diagnostic.
FLAG_RE = re.compile(r'\s*\[(?P<flag>-W[^\]\s]*)\]\s*$')

# CMake Warning (dev) at CMakeLists.txt:12 (project):   -- message follows on indented lines
CMAKE_RE = re.compile(
    r'(?<!\S)CMake (?P<what>Deprecation Warning|Warning \(dev\)|Warning|Error)'
    r'(?: (?:at|in) (?P<file>\S+?)(?::(?P<line>\d+))?(?: \((?P<cmd>[^)]*)\))?)?:\s*$')
CMAKE_FLAGS = {'Warning': 'cmake', 'Warning (dev)': 'cmake-dev',
               'Deprecation Warning': 'cmake-deprecation', 'Error': 'cmake'}

# GCC's echoed source / caret / fix-it lines ("   42 |   code", "      |   ^~~", "  +++ |+#include").
SOURCE_ECHO_RE = re.compile(r'^\s+(?:\d+|\+\+\+)?\s*\|')

# Common CI timestamp prefixes, stripped automatically unless --strip-prefix is given:
#   2026-11-24T09:15:00.123Z ...   [2026-11-24 09:15:00] ...   [09:15:00] ...
TIMESTAMP_RE = re.compile(
    r'^\[?\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:[.,]\d+)?(?:Z|[+-]\d\d:?\d\d)?\]?[ \t]'
    r'|^\[\d\d:\d\d:\d\d(?:[.,]\d+)?\][ \t]')  # one separator only: CMake's indentation matters

# Per-run noise in tool "locations" that would stop identical warnings from merging.
TOOL_NOISE = ((re.compile(r'\+0x[0-9a-fA-F]+'), '+0x...'),
              (re.compile(r'^make\[\d+\]'), 'make'))


def norm_flag(flag):
    """'-Wconversion', 'Wconversion', 'conversion', '-Wformat=' all -> 'conversion' / 'format'."""
    flag = flag.strip().lstrip('-')
    if flag.startswith('W'):
        flag = flag[1:]
    return flag.rstrip('=')


class HitList(object):
    def __init__(self, opts):
        self.opts = opts
        self.root = os.path.normpath(opts.root) if opts.root else None
        self.entries = {}
        self.lines = 0
        self.found = Counter()      # kind -> occurrences seen in the log
        self.filtered = 0           # occurrences dropped by --exclude/--only/--flag/--skip-flag
        self.want_flags = set(norm_flag(f) for f in opts.flag)
        self.skip_flags = set(norm_flag(f) for f in opts.skip_flag)

    # ---- parsing ---------------------------------------------------------

    def parse(self, path):
        name = '<stdin>' if path == '-' else path
        pending = None  # an open CMake warning block collecting its indented message
        for lineno, raw in enumerate(open_log(path), 1):
            self.lines += 1
            if pending is None and 'arning' not in raw and 'rror' not in raw:
                continue  # fast path: the vast majority of lines
            line = ANSI_RE.sub('', raw).rstrip()
            if self.opts.strip_prefix:
                line = self.opts.strip_prefix.sub('', line, count=1)

            if pending is not None:
                if not line or line[0].isspace():
                    text = line.strip()
                    if text.startswith('Call Stack'):
                        pending['done'] = True
                    elif text and not pending['done']:
                        pending['text'].append(text)
                    continue
                self._finish_cmake(pending)
                pending = None
                if 'arning' not in line and 'rror' not in line:
                    continue

            if SOURCE_ECHO_RE.match(line):
                continue

            m = CMAKE_RE.search(line)
            if m:
                pending = dict(m.groupdict(), text=[], done=False, where='%s:%d' % (name, lineno))
                continue

            m = DIAG_RE.search(line)
            if m:
                self._add(m.group('kind'), self._norm_path(m.group('file')),
                          int(m.group('line')), int(m.group('col')) if m.group('col') else None,
                          m.group('msg'), '%s:%d' % (name, lineno))
                continue

            m = TOOL_RE.match(line)
            if m:
                where = m.group('file')
                for rx, repl in TOOL_NOISE:
                    where = rx.sub(repl, where)
                self._add(m.group('kind').lower(), where, None, None,
                          m.group('msg'), '%s:%d' % (name, lineno))

        if pending is not None:
            self._finish_cmake(pending)

    def _finish_cmake(self, p):
        kind = 'error' if p['what'] == 'Error' else 'warning'
        msg = ' '.join(p['text']) or '(no message)'
        if p['cmd']:
            msg = '%s(): %s' % (p['cmd'], msg)
        msg += ' [-W%s]' % CMAKE_FLAGS[p['what']]  # so it groups under its own "flag"
        self._add(kind, self._norm_path(p['file']) if p['file'] else 'CMake',
                  int(p['line']) if p['line'] else None, None, msg, p['where'])

    def _norm_path(self, p):
        if self.opts.build_dir and not os.path.isabs(p):
            p = os.path.join(self.opts.build_dir, p)
        p = os.path.normpath(p)
        if self.root and (p + os.sep).startswith(self.root + os.sep) and p != self.root:
            p = os.path.relpath(p, self.root)
        return p

    def _add(self, kind, file, line, col, msg, where):
        kind = 'warning' if kind == 'warning' else 'error'
        self.found[kind] += 1
        if kind == 'error' and not self.opts.errors:
            return

        flag = ''
        m = FLAG_RE.search(msg)
        if m:
            flag, msg = m.group('flag'), msg[:m.start()]
            if flag.startswith('-Werror='):
                flag = '-W' + flag[len('-Werror='):]
        msg = msg.strip()

        o = self.opts
        nf = norm_flag(flag)
        if ((o.only and not any(rx.search(file) for rx in o.only))
                or any(rx.search(file) for rx in o.exclude)
                or (self.want_flags and nf not in self.want_flags)
                or nf in self.skip_flags):
            self.filtered += 1
            return

        key = (kind, file, line, col, flag) if o.loose else (kind, file, line, col, flag, msg)
        e = self.entries.get(key)
        if e is None:
            e = self.entries[key] = dict(kind=kind, flag=flag, file=file, line=line, col=col,
                                         count=0, first_seen=where, messages=Counter())
        e['count'] += 1
        e['messages'][msg] += 1

    # ---- results ---------------------------------------------------------

    def ranked(self):
        rows = [e for e in self.entries.values() if e['count'] >= self.opts.min_count]
        rows.sort(key=lambda e: (-e['count'], e['file'], e['line'] or 0, e['col'] or 0))
        return rows[:self.opts.top] if self.opts.top > 0 else rows

    def summary(self, keyfn):
        """[(key, unique, total)] sorted by total desc."""
        uniq, tot = Counter(), Counter()
        for e in self.entries.values():
            k = keyfn(e)
            uniq[k] += 1
            tot[k] += e['count']
        return sorted(((k, uniq[k], tot[k]) for k in tot), key=lambda r: (-r[2], r[0]))


def open_log(path):
    if path == '-':
        return io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    return open(path, 'r', encoding='utf-8', errors='replace')


def location(e):
    s = e['file']
    if e['line'] is not None:
        s += ':%d' % e['line']
        if e['col'] is not None:
            s += ':%d' % e['col']
    return s


def top_message(e):
    return e['messages'].most_common(1)[0][0]


def clip(s, n):
    return s if n <= 0 or len(s) <= n else s[:n - 3] + '...'


# ---- output formats ----------------------------------------------------------

def write_text(out, hl, rows, opts):
    w = out.write
    total = sum(e['count'] for e in hl.entries.values())
    unique = len(hl.entries)
    what = 'warnings + errors' if opts.errors else 'warnings'

    w('Scanned %s lines from %s\n' % (fmt(hl.lines), ', '.join(opts.logs)))
    w('%s %s -> %s unique' % (fmt(total), what, fmt(unique)))
    if total:
        w(' (%s duplicates collapsed, %.0f%%)' % (fmt(total - unique), 100.0 * (total - unique) / total))
    w('\n')
    if hl.filtered:
        w('%s occurrences dropped by --exclude/--only/--flag/--skip-flag\n' % fmt(hl.filtered))
    if hl.found['error'] and not opts.errors:
        w('Note: %s error lines also present (add --errors to list them)\n' % fmt(hl.found['error']))
    if not total:
        return

    by_flag = hl.summary(lambda e: e['flag'] or '(no flag)')
    by_file = hl.summary(lambda e: e['file'])
    width = min(60, max([10] + [len(k) for k, _, _ in by_flag + by_file[:opts.top_files or None]]))

    w('\nBY FLAG%s  %8s %9s\n' % (' ' * (width - 5), 'unique', 'total'))
    for k, u, t in by_flag:
        w('  %-*s  %8s %9s\n' % (width, clip(k, width), fmt(u), fmt(t)))

    shown = by_file[:opts.top_files] if opts.top_files > 0 else by_file
    title = 'TOP FILES' if len(shown) < len(by_file) else 'BY FILE'
    w('\n%s%s  %8s %9s\n' % (title, ' ' * (width + 2 - len(title)), 'unique', 'total'))
    for k, u, t in shown:
        w('  %-*s  %8s %9s\n' % (width, clip(k, width), fmt(u), fmt(t)))
    if len(shown) < len(by_file):
        w('  ... %d more files (--top-files 0 for all)\n' % (len(by_file) - len(shown)))

    w('\nHIT LIST (most occurrences first%s)\n' % (', top %d' % opts.top if opts.top else ''))
    cw = len(fmt(rows[0]['count'])) + 1 if rows else 2
    pad = ' ' * (cw + 5)  # line messages up under the location
    for e in rows:
        head = '%*sx  %s' % (cw, fmt(e['count']), location(e))
        if e['flag']:
            head += '  [%s]' % e['flag']
        if e['kind'] == 'error':
            head += '  ERROR'
        variants = e['messages'].most_common()
        if len(variants) > 1:
            head += '  (%d message variants)' % len(variants)
        w('  ' + head + '\n')
        if len(variants) == 1:
            w('%s%s\n' % (pad, clip(variants[0][0], opts.max_msg)))
        else:
            for msg, n in variants[:3]:
                w('%s%s  (x%s)\n' % (pad, clip(msg, opts.max_msg), fmt(n)))
            if len(variants) > 3:
                w('%s... %d more variants\n' % (pad, len(variants) - 3))


def write_csv(out, hl, rows, opts):
    cw = csv.writer(out)
    cw.writerow(['count', 'kind', 'flag', 'file', 'line', 'column', 'message',
                 'message_variants', 'first_seen_in_log'])
    for e in rows:
        cw.writerow([e['count'], e['kind'], e['flag'], e['file'],
                     '' if e['line'] is None else e['line'], '' if e['col'] is None else e['col'],
                     top_message(e), len(e['messages']), e['first_seen']])


def write_json(out, hl, rows, opts):
    doc = {
        'summary': {
            'lines_scanned': hl.lines,
            'occurrences': sum(e['count'] for e in hl.entries.values()),
            'unique': len(hl.entries),
            'found': dict(hl.found),
            'filtered': hl.filtered,
            'by_flag': [{'flag': k, 'unique': u, 'total': t}
                        for k, u, t in hl.summary(lambda e: e['flag'] or '')],
            'by_file': [{'file': k, 'unique': u, 'total': t}
                        for k, u, t in hl.summary(lambda e: e['file'])],
        },
        'hits': [{'count': e['count'], 'kind': e['kind'], 'flag': e['flag'], 'file': e['file'],
                  'line': e['line'], 'column': e['col'], 'message': top_message(e),
                  'variants': [{'message': m, 'count': n} for m, n in e['messages'].most_common()],
                  'first_seen_in_log': e['first_seen']} for e in rows],
    }
    json.dump(doc, out, indent=2, ensure_ascii=False)
    out.write('\n')


def write_gcc(out, hl, rows, opts):
    """Deduplicated log in compiler format, worst first: works with vim :cfile, Emacs, IDE matchers."""
    for e in rows:
        flag = ' [%s]' % e['flag'] if e['flag'] else ''
        out.write('%s: %s: [x%d] %s%s\n' % (location(e), e['kind'], e['count'], top_message(e), flag))


WRITERS = {'text': write_text, 'csv': write_csv, 'json': write_json, 'gcc': write_gcc}


def fmt(n):
    return '{:,}'.format(n)


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Deduplicate and rank the warnings in a CMake/GCC build log.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('logs', nargs='+', metavar='LOG', help="build log(s); '-' reads stdin; .gz is fine")
    p.add_argument('-f', '--format', choices=sorted(WRITERS), default='text', help='output format (default: text)')
    p.add_argument('-o', '--output', help='write to this file instead of stdout')
    p.add_argument('--root', help='source-tree root to strip from paths, so they print as src/foo.h')
    p.add_argument('--build-dir', help='resolve relative paths against this dir (helps if the log mixes '
                                       'relative and absolute spellings of the same file)')
    p.add_argument('--exclude', action='append', default=[], metavar='REGEX',
                   help='drop warnings whose path matches, e.g. third_party/ (repeatable)')
    p.add_argument('--only', action='append', default=[], metavar='REGEX',
                   help='keep only warnings whose path matches, e.g. ^src/persist/ (repeatable)')
    p.add_argument('--flag', action='append', default=[],
                   help='keep only this warning flag, e.g. conversion or -Wsign-compare (repeatable)')
    p.add_argument('--skip-flag', action='append', default=[], metavar='FLAG',
                   help='drop this warning flag, e.g. unused-parameter (repeatable)')
    p.add_argument('--errors', action='store_true', help='list errors too, not just warnings')
    p.add_argument('--loose', action='store_true',
                   help='merge hits at the same file:line:col + flag even if the message text differs '
                        '(collapses template-instantiation variants)')
    p.add_argument('--min-count', type=int, default=1, metavar='N', help='only list hits seen at least N times')
    p.add_argument('--top', type=int, default=0, metavar='N', help='only list the N most frequent hits')
    p.add_argument('--top-files', type=int, default=20, metavar='N',
                   help='files shown in the text summary (default 20, 0 = all)')
    p.add_argument('--max-msg', type=int, default=400, metavar='N',
                   help='truncate messages longer than N chars in text output (default 400, 0 = never)')
    p.add_argument('--strip-prefix', metavar='REGEX',
                   help=r"strip this from the start of each line first (default: ISO-8601 or [hh:mm:ss] "
                        r"timestamps are stripped automatically); e.g. '^\S+ \S+ '")
    # Let "--flag -Wconversion" work (argparse would otherwise read -Wconversion as an option).
    argv = list(sys.argv[1:] if argv is None else argv)
    for i in range(len(argv) - 2, -1, -1):
        if argv[i] in ('--flag', '--skip-flag') and argv[i + 1].startswith('-W'):
            argv[i:i + 2] = [argv[i] + '=' + argv[i + 1]]
    opts = p.parse_args(argv)

    try:
        opts.exclude = [re.compile(x) for x in opts.exclude]
        opts.only = [re.compile(x) for x in opts.only]
        opts.strip_prefix = re.compile(opts.strip_prefix) if opts.strip_prefix else TIMESTAMP_RE
    except re.error as err:
        p.error('bad regex: %s' % err)

    hl = HitList(opts)
    for path in opts.logs:
        try:
            hl.parse(path)
        except (IOError, OSError) as err:
            p.error(str(err))

    rows = hl.ranked()
    if opts.output:
        with open(opts.output, 'w', encoding='utf-8', newline='' if opts.format == 'csv' else None) as out:
            WRITERS[opts.format](out, hl, rows, opts)
        n = sum(e['count'] for e in hl.entries.values())
        sys.stderr.write('%s occurrences -> %s unique; wrote %d rows to %s\n'
                         % (fmt(n), fmt(len(hl.entries)), len(rows), opts.output))
    else:
        out = sys.stdout
        try:
            out.reconfigure(errors='replace')  # Python 3.7+: don't die on GCC's curly quotes in a C locale
        except AttributeError:
            out = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding or 'utf-8',
                                   errors='replace')
        WRITERS[opts.format](out, hl, rows, opts)
        out.flush()


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:  # e.g. piped into `head`
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
