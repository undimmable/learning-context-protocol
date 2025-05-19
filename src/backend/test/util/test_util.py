from unittest import TestCase

from src.util.util import split_lines, timestamp_string


class TestUtil(TestCase):
    def test_split_lines_basic(self):
        s = "line1\nline2\nline3"
        result = split_lines(s)
        self.assertEqual(result, ["line1", "line2", "line3"])

    def test_split_lines_with_leading_and_trailing_whitespace(self):
        s = "\n\nline1\nline2\nline3\n\n"
        result = split_lines(s)
        self.assertEqual(result, ["line1", "line2", "line3"])

    def test_split_lines_single_line(self):
        s = "single line"
        result = split_lines(s)
        self.assertEqual(result, ["single line"])

    def test_split_lines_empty_string(self):
        s = ""
        result = split_lines(s)
        self.assertEqual(result, [""])

    def test_timestamp_string_format(self):
        ts = timestamp_string()
        # ISO 8601 basic assertion: should end with 'Z' or have time zone offset and the format YYYY-MM-DDTHH:MM:SS...
        # Because it uses ISO format with UTC tzinfo, usually ends with 'Z' or +00:00
        # In python, isoformat may produce something like '2024-06-07T14:20:30.123456+00:00'
        iso8601_regex = (
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00|Z)$"
        )
        # timestamp_string uses datetime.datetime.now(datetime.UTC).isoformat()
        # The isoformat() of a datetime with timezone UTC usually ends with +00:00 (Python 3.12)
        # so we adjust the regex check accordingly (without 'Z')
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00)$")
