"""Exact binary-artifact guards, streamed transfers and read-only wheel inspection."""
from email.parser import BytesParser
from email.policy import compat32
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

from packaging.specifiers import SpecifierSet
from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

HOSTS = {'files.pythonhosted.org', 'download.pytorch.org', 'download-r2.pytorch.org'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path, progress_label=None):
    result = hashlib.sha256()
    count, last_notice = 0, time.monotonic()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
            count += len(block)
            if progress_label and time.monotonic() - last_notice >= 20:
                print('  Checksum read:', progress_label, count, 'bytes', flush=True)
                last_notice = time.monotonic()
    return result.hexdigest()


def safe_name(name):
    require(isinstance(name, str) and name not in {'', '.', '..'} and
            PurePosixPath(name).name == name and '\\' not in name,
            'Unsafe artifact filename')
    return name


def public_url(url):
    parsed = urlparse(url)
    require(parsed.scheme == 'https' and parsed.hostname in HOSTS and
            parsed.username is None and parsed.password is None and
            parsed.port in {None, 443}, 'Unexpected wheel publisher URL')
    return url


class PublisherRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_record(record, tags, python_version):
    name = safe_name(record['filename'])
    package, version, _, wheel_tags = parse_wheel_filename(name)
    require(canonicalize_name(package) == canonicalize_name(record['package']) and
            version == Version(record['version']), 'Wheel filename identity differs: ' + name)
    require(bool(wheel_tags & set(tags)), 'Wheel is incompatible with this CPU inspection runtime: ' + name)
    bound = record.get('requires_python')
    require(not bound or SpecifierSet(bound).contains(python_version, prereleases=True),
            'Wheel published Python bound differs: ' + name)
    require(type(record['bytes']) is int and record['bytes'] > 0 and
            re.fullmatch('[0-9a-f]{64}', record['sha256']) is not None,
            'Missing wheel size/SHA-256: ' + name)
    public_url(record['url'])
    if canonicalize_name(package) == 'torch':
        require(version == Version('2.11.0+cu130') and
                urlparse(record['url']).hostname in {'download.pytorch.org', 'download-r2.pytorch.org'},
                'Exact approved CUDA Torch build/publisher required')


def verify_file(path, record):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Wheel missing or symlink: ' + path.name)
    require(path.stat().st_size == record['bytes'], 'Wheel size differs; preserved: ' + path.name)
    require(digest(path, path.name) == record['sha256'], 'Wheel checksum differs; preserved: ' + path.name)
    return {'filename': path.name, 'package': record['package'], 'version': record['version'],
            'bytes': record['bytes'], 'sha256': record['sha256'], 'approved_url': record['url']}


def copy_response(response, stream, record, progress=print):
    """Read at most the approved size plus one byte, preserving incomplete files."""
    public_url(response.url)
    require(response.status == 200, 'Unexpected wheel HTTP status')
    encoding = response.headers.get('Content-Encoding')
    require(encoding in {None, 'identity'}, 'Unexpected wheel content encoding')
    declared = response.headers.get('Content-Length')
    require(declared is None or (declared.isdigit() and int(declared) == record['bytes']),
            'Publisher content length changed')
    result, count, last_notice = hashlib.sha256(), 0, time.monotonic()
    while True:
        chunk = response.read(min(1024 * 1024, record['bytes'] - count + 1))
        if not chunk:
            break
        count += len(chunk)
        require(count <= record['bytes'], 'Wheel response exceeds approved size')
        stream.write(chunk)
        result.update(chunk)
        if time.monotonic() - last_notice >= 20:
            progress('  Downloaded ' + str(count) + '/' + str(record['bytes']) + ' bytes', flush=True)
            last_notice = time.monotonic()
    require(count == record['bytes'] and result.hexdigest() == record['sha256'],
            'Downloaded wheel size/checksum differs; incomplete file preserved')
    return count


def download_one(record, folder, attempt_id):
    folder = Path(folder)
    final = folder / safe_name(record['filename'])
    require(not final.exists() and not final.is_symlink(), 'Complete wheel already exists')
    pending_folder = folder.parent / 'incomplete' / attempt_id
    pending_folder.mkdir(parents=True, exist_ok=True)
    pending = pending_folder / (record['filename'] + '.partial')
    require(not pending.exists() and not pending.is_symlink(), 'Attempt partial already exists')
    request = Request(public_url(record['url']), headers={
        'User-Agent': 'ATML-PA2-Task4-Approved-Wheel-Cache', 'Accept-Encoding': 'identity'})
    opener = build_opener(PublisherRedirects())
    with opener.open(request, timeout=30) as response, pending.open('xb') as stream:
        copy_response(response, stream, record)
        stream.flush()
        os.fsync(stream.fileno())
    # Verify bytes read back from Drive before publishing a complete unit.
    verify_file(pending, record)
    require(not final.exists() and not final.is_symlink(), 'Another writer created a complete wheel')
    os.replace(pending, final)
    return final


def inspect_wheel(path, record, tags, python_version):
    """Read wheel metadata without extracting or executing any package code."""
    validate_record(record, tags, python_version)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        require(len(names) == len(set(names)), 'Duplicate wheel ZIP member')
        for item in entries:
            relative = PurePosixPath(item.filename)
            require(not relative.is_absolute() and '..' not in relative.parts and
                    '\\' not in item.filename and not stat.S_ISLNK(item.external_attr >> 16) and
                    not item.flag_bits & 1, 'Unsafe/encrypted wheel ZIP member')
        candidates = [name for name in names if len(PurePosixPath(name).parts) == 2 and
                      name.endswith('.dist-info/METADATA')]
        require(len(candidates) == 1, 'Wheel must have one top-level METADATA')
        metadata_name = candidates[0]
        directory = metadata_name.rsplit('/', 1)[0]
        wheel_name = directory + '/WHEEL'
        require(wheel_name in names and directory + '/RECORD' in names, 'Wheel format files missing')
        require(archive.getinfo(metadata_name).file_size <= 8 * 1024 * 1024 and
                archive.getinfo(wheel_name).file_size <= 1024 * 1024, 'Wheel metadata is unexpectedly large')
        raw = archive.read(metadata_name)
        raw_wheel = archive.read(wheel_name)
        message = BytesParser(policy=compat32).parsebytes(raw)
        wheel = BytesParser(policy=compat32).parsebytes(raw_wheel)
        for field in ['Name', 'Version']:
            require(len(message.get_all(field, [])) == 1, 'Wheel metadata identity is missing/duplicated')
        require(canonicalize_name(message['Name']) == canonicalize_name(record['package']) and
                Version(message['Version']) == Version(record['version']), 'Wheel METADATA identity differs')
        bound = message.get('Requires-Python')
        require(not bound or SpecifierSet(bound).contains(python_version, prereleases=True),
                'Actual wheel Python bound is incompatible')
        require(wheel.get('Wheel-Version') == '1.0', 'Unsupported wheel format needs review')
        actual_tags = set()
        for tag in wheel.get_all('Tag', []):
            actual_tags.update(parse_tag(tag))
        filename_tags = parse_wheel_filename(record['filename'])[3]
        require(actual_tags == filename_tags and bool(actual_tags & set(tags)),
                'Actual wheel tags differ from the approved filename')
        unpacked = sum(item.file_size for item in entries if not item.is_dir())
        node = {'name': canonicalize_name(message['Name']), 'version': message['Version'],
                'requires_dist': message.get_all('Requires-Dist', []),
                'provides_extra': message.get_all('Provides-Extra', []), 'requires_python': bound}
        return {'filename': record['filename'], 'metadata': node,
                'metadata_sha256': hashlib.sha256(raw).hexdigest(),
                'wheel_metadata_sha256': hashlib.sha256(raw_wheel).hexdigest(),
                'compatible_tags': sorted(str(tag) for tag in actual_tags),
                'zip_member_count': len(entries), 'zip_uncompressed_file_bytes': unpacked,
                'compressed_bytes': record['bytes'],
                'size_meaning': 'Logical ZIP file sizes, not measured installed disk usage; bytecode, filesystem and installer temporary space are additional.'}


def check_closure(inspect_chain, records, pins, target, course_lines, saved_versions):
    nodes = {}
    for name, version in pins.items():
        key = (canonicalize_name(name), Version(version))
        require(key in records, 'Pinned wheel metadata missing: ' + name)
        nodes[canonicalize_name(name)] = records[key]['metadata']
    lookup = nodes.get
    course = inspect_chain(course_lines, lookup, target)
    require(course['required_chain_passed'] and not course['issues'],
            'Actual wheel course dependency closure failed: ' + repr(course['issues']))
    observed = {name: node['version'] for name, node in course['nodes'].items()}
    require(observed == saved_versions, 'Wheel course closure differs from the saved 83-package GPU graph')
    full = inspect_chain([name + '==' + version for name, version in sorted(pins.items())], lookup, target)
    require(full['required_chain_passed'] and not full['issues'] and set(full['nodes']) == set(pins),
            'Full isolated wheel dependency closure failed: ' + repr(full['issues']))
    return course, full
