import csv

from scripts import prepare_data


def test_main_writes_processed_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(prepare_data, "OUT_DIR", tmp_path / "processed")
    monkeypatch.setattr(
        prepare_data,
        "read_conllu",
        lambda path: [["xin chào", "."], ["sinh viên"]],
    )

    prepare_data.main()

    with (tmp_path / "processed" / "train.csv").open(
        encoding="utf-8-sig", newline=""
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert set(prepare_data.SPLITS) == {"train", "dev", "test"}
    for split in prepare_data.SPLITS:
        split_path = tmp_path / "processed" / f"{split}.csv"
        assert split_path.is_file()
        with split_path.open(encoding="utf-8-sig", newline="") as csv_file:
            split_rows = list(csv.DictReader(csv_file))
        assert len(split_rows) == 2
        assert split_rows[0]["text"] == "xin chào ."
        assert split_rows[0]["segmented"] == "xin_chào ."
        assert split_rows[0]["labels"] == "B I B"
