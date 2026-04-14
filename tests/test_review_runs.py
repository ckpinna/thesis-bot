from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from thesis_bot.config import load_settings
from thesis_bot.io.review_runs import bucket_review_filename
from thesis_bot.io.review_runs import expected_review_bucket_paths
from thesis_bot.io.review_runs import write_review_bucket_csvs
from thesis_bot.pipelines.extract_for_review import split_review_dataframe_by_core_thesis
from thesis_bot.pipelines.load_reviewed_theses import load_reviewed_dataframe
from thesis_bot.schemas import REVIEWED_CSV_COLUMNS


def build_review_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Thesis Number": 1,
                "Thesis Statement": "AI will improve underwriting.",
                "Title": "AI Improves Underwriting Speed",
                "Description": "A thesis about underwriting efficiency.",
                "Supports Thesis Numbers": "",
                "Core Thesis": "Artificial Intelligence",
                "Source File": "deck-a.pdf",
            },
            {
                "Thesis Number": 2,
                "Thesis Statement": "Construction software reduces waste.",
                "Title": "Construction Software Reduces Waste",
                "Description": "A thesis about operational savings.",
                "Supports Thesis Numbers": "1",
                "Core Thesis": "Construction",
                "Source File": "deck-b.pptx",
            },
        ],
        columns=REVIEWED_CSV_COLUMNS,
    )


class ReviewRunTests(unittest.TestCase):
    def test_bucket_split_emits_all_configured_buckets(self) -> None:
        dataframe = build_review_dataframe()
        buckets = split_review_dataframe_by_core_thesis(
            dataframe,
            ("Artificial Intelligence", "Construction", "Biotech", "Criteria", "Unsorted"),
        )
        self.assertEqual(set(buckets), {
            "Artificial Intelligence",
            "Construction",
            "Biotech",
            "Criteria",
            "Unsorted",
        })
        self.assertEqual(len(buckets["Artificial Intelligence"]), 1)
        self.assertEqual(len(buckets["Biotech"]), 0)

    def test_local_review_run_roundtrip_loads_all_bucket_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            env_path = temp_root / ".env"
            output_dir = temp_root / "outputs"
            env_path.write_text(
                "\n".join(
                    [
                        "CORE_THESES=Artificial Intelligence,Construction,Biotech,Criteria,Unsorted",
                        "EXTRACTION_SOURCE=local",
                        "REVIEW_OUTPUT_DESTINATION=local",
                        f"LOCAL_REVIEW_OUTPUT_DIR={output_dir}",
                        "REVIEW_INPUT_SOURCE=local",
                        f"LOCAL_REVIEWED_RUN_DIR={output_dir / 'review_run_fixture'}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            settings = load_settings(env_path=env_path, override=True)

            run_id = "review_run_fixture"
            bucket_dataframes = split_review_dataframe_by_core_thesis(
                build_review_dataframe(),
                settings.core_theses,
            )
            output_paths = write_review_bucket_csvs(settings, bucket_dataframes, run_id=run_id)

            self.assertEqual(
                Path(output_paths["Artificial Intelligence"]).name,
                bucket_review_filename("Artificial Intelligence"),
            )
            self.assertEqual(
                expected_review_bucket_paths(settings)["Criteria"],
                settings.local_reviewed_run_dir / bucket_review_filename("Criteria"),
            )

            loaded = load_reviewed_dataframe(settings=settings)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(set(loaded["Core Thesis"]), {"Artificial Intelligence", "Construction"})


if __name__ == "__main__":
    unittest.main()
