"""Unit tests for kaggle_ml_toolkit.loader module."""

import pytest
import pandas as pd

from kaggle_ml_toolkit.loader import load_csv, load_competition_data, DataBundle


class TestLoadCsv:
    """Tests for the load_csv function."""

    def test_load_csv_valid_file(self, tmp_path):
        """Loading a valid CSV returns a DataFrame with correct shape and columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,score\nAlice,30,85.5\nBob,25,90.0\n")

        df = load_csv(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ["name", "age", "score"]

    def test_load_csv_file_not_found(self, tmp_path):
        """Non-existent path raises FileNotFoundError with path in message."""
        fake_path = str(tmp_path / "nonexistent.csv")

        with pytest.raises(FileNotFoundError, match=fake_path.replace("\\", "\\\\")):
            load_csv(fake_path)

    def test_load_csv_empty_file(self, tmp_path):
        """CSV with headers only raises ValueError with 'no data rows' context."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("col_a,col_b,col_c\n")

        with pytest.raises(ValueError, match="[Nn]o data rows"):
            load_csv(str(csv_file))

    def test_load_csv_preserves_columns(self, tmp_path):
        """Original column names are preserved exactly, including unusual characters."""
        csv_file = tmp_path / "cols.csv"
        csv_file.write_text("First Name,Last-Name,score_v2,ID #\nJane,Doe,99.1,1\n")

        df = load_csv(str(csv_file))

        assert list(df.columns) == ["First Name", "Last-Name", "score_v2", "ID #"]


class TestLoadCompetitionData:
    """Tests for the load_competition_data function."""

    def test_load_competition_data_valid(self, tmp_path):
        """Valid directory with train.csv and test.csv returns DataBundle."""
        train_file = tmp_path / "train.csv"
        test_file = tmp_path / "test.csv"
        train_file.write_text("id,target,feature\n1,0,3.5\n2,1,4.2\n")
        test_file.write_text("id,feature\n3,5.1\n4,6.0\n")

        bundle = load_competition_data(str(tmp_path))

        assert isinstance(bundle, DataBundle)
        assert isinstance(bundle.train, pd.DataFrame)
        assert isinstance(bundle.test, pd.DataFrame)
        assert bundle.train.shape == (2, 3)
        assert bundle.test.shape == (2, 2)
        assert "target" in bundle.train.columns
        assert "id" in bundle.test.columns

    def test_load_competition_data_missing_train(self, tmp_path):
        """Directory without train.csv raises FileNotFoundError mentioning train.csv."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("id,feature\n1,2.0\n")

        with pytest.raises(FileNotFoundError, match="train.csv"):
            load_competition_data(str(tmp_path))

    def test_load_competition_data_missing_test(self, tmp_path):
        """Directory without test.csv raises FileNotFoundError mentioning test.csv."""
        train_file = tmp_path / "train.csv"
        train_file.write_text("id,target\n1,0\n")

        with pytest.raises(FileNotFoundError, match="test.csv"):
            load_competition_data(str(tmp_path))

    def test_load_competition_data_missing_dir(self, tmp_path):
        """Non-existent directory raises FileNotFoundError."""
        fake_dir = str(tmp_path / "no_such_dir")

        with pytest.raises(FileNotFoundError):
            load_competition_data(fake_dir)
