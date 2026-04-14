from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from thesis_bot.config import DEFAULT_CORE_THESES
from thesis_bot.config import UNSORTED_CORE_THESIS
from thesis_bot.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_default_core_theses_include_unsorted_at_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "EXTRACTION_SOURCE=local\n"
                "REVIEW_OUTPUT_DESTINATION=local\n"
                "REVIEW_INPUT_SOURCE=local\n",
                encoding="utf-8",
            )
            settings = load_settings(env_path=env_path, override=True)
        self.assertEqual(settings.core_theses, DEFAULT_CORE_THESES)

    def test_unsorted_is_normalized_to_final_position(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "CORE_THESES=Unsorted,Criteria,Construction,Unsorted\n"
                "EXTRACTION_SOURCE=local\n"
                "REVIEW_OUTPUT_DESTINATION=local\n"
                "REVIEW_INPUT_SOURCE=local\n",
                encoding="utf-8",
            )
            settings = load_settings(env_path=env_path, override=True)
        self.assertEqual(
            settings.core_theses,
            ("Criteria", "Construction", UNSORTED_CORE_THESIS),
        )

    def test_invalid_mode_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "EXTRACTION_SOURCE=s3\n"
                "REVIEW_OUTPUT_DESTINATION=local\n"
                "REVIEW_INPUT_SOURCE=local\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "EXTRACTION_SOURCE"):
                load_settings(env_path=env_path, override=True)


if __name__ == "__main__":
    unittest.main()
