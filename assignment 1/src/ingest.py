"""Parse a MAL export and create a small, attributed, offline metadata snapshot.

Usage: python src/ingest.py --xml YOUR_EXPORT.xml --metadata FULL_DATABASE.json
The full metadata file is the pinned 2026-27 manami release at SOURCE below.
Inputs are local files. This script makes no network requests. Rebuilding writes
archive.json, animelist.xml, metadata_snapshot.json, and provenance.json into data/.
No external rating, popularity, title text, or viewing behavior becomes a feature.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/manami-project/anime-offline-database/releases/download/2026-27/anime-offline-database-minified.json"


def parse_export(path):
    """Read one XML export into a reproducibly ordered list of dictionaries.

    Args:
        path: Local plain XML file or a gzip-compressed file ending in .gz.

    Returns:
        Rows sorted by integer anime_id. Each row contains the title, exported
        type and episode count, personal score, and viewing status. Score 0 is
        preserved as the export's unrated marker; sample filtering happens in
        the modeling pipeline.

    Raises:
        ValueError: An integer field cannot be parsed, a score is outside 0-10,
            or the export contains duplicate anime IDs. XML parsing and file
            errors propagate so that a malformed archive cannot pass silently.
    """
    # ElementTree decodes the file's declared XML encoding from the byte stream.
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        root = ET.parse(stream).getroot()
    rows = []
    for a in root.findall('anime'):
        # Explicit field projection omits unrelated account and tracking data.
        row = {k: a.findtext(v, '') for k, v in {
            'anime_id': 'series_animedb_id', 'title': 'series_title',
            'xml_type': 'series_type', 'xml_episodes': 'series_episodes',
            'score': 'my_score', 'status': 'my_status'}.items()}
        for k in ['anime_id', 'xml_episodes', 'score']:
            row[k] = int(row[k] or 0)
        if not 0 <= row['score'] <= 10:
            raise ValueError(f"Invalid rating: {row['anime_id']}")
        rows.append(row)
    ids = [r['anime_id'] for r in rows]
    # Reject duplicates before joining: a repeated ID would duplicate training
    # observations and could make a model's apparent support misleading.
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate MAL IDs: resolve before modeling.')
    return sorted(rows, key=lambda r: r['anime_id'])


def make_snapshot(xml_path, metadata_path, destination=ROOT / 'data'):
    """Build the offline subset and label-independent relation components.

    Args:
        xml_path: Personal export accepted by parse_export.
        metadata_path: Complete JSON database from the pinned metadata release.
            All records are needed to resolve links through unlisted titles.
        destination: Directory in which the four derived data files are written.
            Existing files with those names are overwritten.

    Returns:
        (archive_rows, metadata_rows), with one metadata row per archive ID.
        Unmatched IDs remain visible as unmatched singleton groups.

    Personal ratings are read for the archive but never determine graph edges
    or group membership. The saved metadata subset excludes public score fields.
    Full-file hashes identify the precise input bytes. Supplying a redacted XML
    copy records that copy's hash, so it can differ from the original export.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = parse_export(xml_path)
    raw_bytes = Path(metadata_path).read_bytes()
    db = json.loads(raw_bytes)
    records = db['data']
    parent = list(range(len(records)))
    # Union-find represents connected components without building a dense
    # adjacency matrix. Initially every database record is its own component.

    def find(i):
        """Return a record's component root, shortening its parent path."""
        while parent[i] != i:
            # Path halving makes future root lookups traverse fewer links.
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        """Merge two components using the smaller record index as their root."""
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    # All providers' source URLs identify a record; relation edges are resolved
    # across the full database, so unlisted bridge seasons still join groups.
    source_to_index = {s.rstrip('/'): i for i, r in enumerate(records) for s in r['sources']}
    for i, r in enumerate(records):
        for link in r.get('relatedAnime', []):
            j = source_to_index.get(link.rstrip('/'))
            if j is not None:
                union(i, j)
    mal_to_index = {}
    # The full URL index resolves links across providers. This second index
    # extracts MAL identifiers for the exact join with the personal archive.
    for s, i in source_to_index.items():
        m = re.fullmatch(r'https?://myanimelist\.net/anime/(\d+)', s)
        if m:
            mal_to_index[int(m.group(1))] = i
    # Stable component names based on minimum MAL ID, independent of targets.
    component_min = {}
    for aid, i in mal_to_index.items():
        c = find(i)
        component_min[c] = min(component_min.get(c, aid), aid)
    subset = []
    # Project only required fields after resolving the complete relation graph.
    # Filtering the database first would lose paths through unlisted titles.
    for row in rows:
        aid = row['anime_id']
        i = mal_to_index.get(aid)
        if i is None:
            subset.append({'anime_id': aid, 'matched': False,
                           'relation_group': f'unmatched-{aid}', 'tags': [], 'studios': []})
            continue
        r = records[i]
        subset.append({'anime_id': aid, 'matched': True,
                       'relation_group': f"related-{component_min[find(i)]}",
                       'tags': sorted(set(r.get('tags', []))),
                       'studios': sorted(set(r.get('studios', []))),
                       'year': r.get('animeSeason', {}).get('year'),
                       'type': r.get('type'), 'episodes': r.get('episodes'),
                       'sources': r['sources']})
    provenance = {'source_url': SOURCE, 'database_date': db['lastUpdate'],
                  'retrieved_date': '2026-10-18', 'license': db['license'],
                  'database_sha256': hashlib.sha256(raw_bytes).hexdigest(),
                  'original_xml_sha256': hashlib.sha256(Path(xml_path).read_bytes()).hexdigest(),
                  'grouping': 'Undirected connected components over all full-database relatedAnime edges; all provider URLs resolved. Components may include crossovers or adaptations.'}
    # The retrieval date identifies the bundled snapshot. Update this provenance
    # deliberately when changing to a different downloaded release.
    (destination / 'archive.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    # Preserve the raw XML structure needed by the parser while omitting the
    # account identifier and unused personal tracking fields from publication.
    xml_root = ET.Element('myanimelist')
    names = {'anime_id': 'series_animedb_id', 'title': 'series_title',
             'xml_type': 'series_type', 'xml_episodes': 'series_episodes',
             'score': 'my_score', 'status': 'my_status'}
    for row in rows:
        node = ET.SubElement(xml_root, 'anime')
        for key, tag in names.items():
            ET.SubElement(node, tag).text = str(row[key])
    ET.indent(xml_root)
    ET.ElementTree(xml_root).write(destination / 'animelist.xml', encoding='utf-8', xml_declaration=True)
    (destination / 'metadata_snapshot.json').write_text(json.dumps(subset, indent=2, ensure_ascii=False) + '\n')
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return rows, subset


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml', required=True)
    p.add_argument('--metadata', required=True)
    args = p.parse_args()
    rows, metadata = make_snapshot(args.xml, args.metadata)
    print(f'{len(rows)} archive rows; {sum(r["matched"] for r in metadata)} metadata matches.')
