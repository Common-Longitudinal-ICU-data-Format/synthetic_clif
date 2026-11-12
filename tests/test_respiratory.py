"""Tests for respiratory support generator."""

import pytest
import pandas as pd

from synthetic_clif.generators.respiratory import RespiratoryGenerator


class TestRespiratoryGenerator:
    """Tests for RespiratoryGenerator."""

    def test_generate_basic(self, hospitalizations_df, seed, mcide):
        """Test basic respiratory support generation."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert len(df) > 0
        assert "hospitalization_id" in df.columns
        assert "recorded_dttm" in df.columns
        assert "device_category" in df.columns
        assert "mode_category" in df.columns
        assert "fio2_set" in df.columns
        assert "peep_set" in df.columns
        assert "tracheostomy" in df.columns

    def test_imv_has_ventilator_settings(self, hospitalizations_df, seed, mcide):
        """Test that IMV records have ventilator settings."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        imv_records = df[df["device_category"] == "IMV"]
        if len(imv_records) > 0:
            assert imv_records["tidal_volume_set"].notna().any()
            assert imv_records["resp_rate_set"].notna().any()

    def test_tracheostomy_flag(self, hospitalizations_df, seed, mcide):
        """Test that tracheostomy flag is integer (0 or 1)."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert set(df["tracheostomy"].unique()).issubset({0, 1})

    def test_device_appropriate_settings(self, hospitalizations_df, seed, mcide):
        """Test that settings are appropriate for device type."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        # Room air should have FiO2 of 0.21
        room_air = df[df["device_category"] == "Room Air"]
        if len(room_air) > 0:
            assert (room_air["fio2_set"].dropna() == 0.21).all()

        # High flow NC charts its flow in liters per minute (CLIF lpm_set)
        hfnc = df[df["device_category"] == "High Flow NC"]
        if len(hfnc) > 0:
            assert hfnc["lpm_set"].between(30, 60).all()
            assert hfnc["flow_rate_set"].isna().all()
