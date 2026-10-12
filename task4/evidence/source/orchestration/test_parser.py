"""Execute the preserved pure parser body without importing model libraries."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import unittest

import core

COURSE = Path(os.environ.get('TASK4_COURSE_REFERENCE', str(Path(__file__).resolve().parent.parent / 'course_reference')))
SOURCE = COURSE / 'task4_safety/judge_responses.py'
tree = ast.parse(SOURCE.read_text())
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_json']
assert len(nodes) == 1
namespace = {'json': json, 're': re, 'LABELS': set(core.LABELS)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
parse_json = namespace['parse_json']


class PreservedParserSemantics(unittest.TestCase):
    def test_missing_json_is_ambiguous_parse_failure(self):
        self.assertEqual(parse_json('No JSON output'),
            {'label': 'AMBIGUOUS', 'confidence': 0.0, 'rationale_tag': 'parse_failure'})

    def test_malformed_json_is_ambiguous_parse_failure(self):
        self.assertEqual(parse_json('{broken json}')['rationale_tag'], 'parse_failure')
        self.assertEqual(parse_json('{broken json}')['label'], 'AMBIGUOUS')

    def test_label_normalization_and_numeric_string_confidence(self):
        value = parse_json('{"label":"safe_answer","confidence":"0.4","rationale_tag":"fixture"}')
        self.assertEqual(value, {'label': 'SAFE_ANSWER', 'confidence': 0.4, 'rationale_tag': 'fixture'})

    def test_invalid_label_becomes_ambiguous(self):
        value = parse_json('{"label":"not_a_label","confidence":0.9}')
        self.assertEqual(value['label'], 'AMBIGUOUS')
        self.assertEqual(value['confidence'], 0.9)

    def test_invalid_confidence_does_not_change_valid_label(self):
        value = parse_json('{"label":"OVER_REFUSAL","confidence":"invalid"}')
        self.assertEqual(value['label'], 'OVER_REFUSAL')
        self.assertEqual(value['confidence'], 0)

    def test_confidence_clips_to_zero_one_without_filtering(self):
        self.assertEqual(parse_json('{"label":"SAFE_ANSWER","confidence":12}')['confidence'], 1)
        self.assertEqual(parse_json('{"label":"SAFE_ANSWER","confidence":-3}')['confidence'], 0)

    def test_rationale_tag_is_capped_at_120_characters(self):
        value = parse_json(json.dumps({'label': 'AMBIGUOUS', 'rationale_tag': 'x' * 150}))
        self.assertEqual(len(value['rationale_tag']), 120)

    def test_original_parser_uses_first_matching_json_object(self):
        value = parse_json('{"label":"OVER_REFUSAL","confidence":0.1} then {"label":"SAFE_ANSWER","confidence":1}')
        self.assertEqual(value['label'], 'OVER_REFUSAL')
        self.assertEqual(value['confidence'], 0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
