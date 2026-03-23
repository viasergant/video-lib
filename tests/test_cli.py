from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scanner.cli import main


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_video_files(tmp_path: Path, names: list[str]) -> list[Path]:
    files = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"fake-video")
        files.append(p)
    return files


def _base_patches(
    *,
    ollama_list_side_effect=None,
    sidecar_exists_return=False,
    process_side_effect=None,
):
    """Return a dict of patch targets → MagicMock configs."""
    mock_ollama_client = MagicMock()
    if ollama_list_side_effect is not None:
        mock_ollama_client.list.side_effect = ollama_list_side_effect
    else:
        mock_ollama_client.list.return_value = []

    mock_whisper = MagicMock()
    mock_processor = MagicMock()
    if process_side_effect is not None:
        mock_processor.process.side_effect = process_side_effect

    return {
        "ollama_client": mock_ollama_client,
        "whisper": mock_whisper,
        "processor": mock_processor,
        "sidecar_exists_return": sidecar_exists_return,
    }


# ---------------------------------------------------------------------------
# Helper: run CLI with common mocks applied
# ---------------------------------------------------------------------------

def _run(args: list[str], mocks: dict, *, catch_exceptions: bool = False) -> object:
    runner = CliRunner()

    # ollama and faster_whisper are lazily imported inside main(); we patch
    # the modules themselves so the import inside the function body resolves
    # to our mocks.
    import types

    mock_ollama_mod = types.ModuleType("ollama")
    mock_ollama_mod.Client = MagicMock(return_value=mocks["ollama_client"])

    mock_fw_mod = types.ModuleType("faster_whisper")
    mock_fw_mod.WhisperModel = MagicMock(return_value=mocks["whisper"])

    with (
        patch.dict("sys.modules", {"ollama": mock_ollama_mod, "faster_whisper": mock_fw_mod}),
        patch("scanner.cli.VideoProcessor", return_value=mocks["processor"]),
        patch("scanner.cli.sidecar_exists", return_value=mocks["sidecar_exists_return"]),
    ):
        return runner.invoke(main, args, catch_exceptions=catch_exceptions)


# ---------------------------------------------------------------------------
# Ollama connectivity check
# ---------------------------------------------------------------------------

class TestOllamaConnectivity:
    def test_exits_with_error_when_ollama_unreachable(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mocks = _base_patches(ollama_list_side_effect=ConnectionRefusedError("refused"))
        result = _run(["--folder", str(tmp_path)], mocks)
        assert result.exit_code == 1
        assert "Cannot connect to Ollama" in result.output

    def test_proceeds_when_ollama_reachable(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mocks = _base_patches()
        result = _run(["--folder", str(tmp_path)], mocks)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------------

class TestVideoDiscovery:
    def test_exits_cleanly_when_no_videos_found(self, tmp_path):
        mocks = _base_patches()
        result = _run(["--folder", str(tmp_path)], mocks)
        assert result.exit_code == 0
        assert "No video files found" in result.output

    def test_discovers_supported_extensions(self, tmp_path):
        _make_video_files(tmp_path, ["a.mp4", "b.mkv", "c.txt"])
        mocks = _base_patches()
        result = _run(["--folder", str(tmp_path)], mocks)
        assert result.exit_code == 0
        # processor.process should only be called for video files
        assert mocks["processor"].process.call_count == 2

    def test_non_recursive_does_not_descend(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "video.mp4").write_bytes(b"fake")
        mocks = _base_patches()
        result = _run(["--folder", str(tmp_path)], mocks)
        # no videos at top level → "No video files found"
        assert "No video files found" in result.output
        assert mocks["processor"].process.call_count == 0

    def test_recursive_flag_descends_into_subdirs(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "video.mp4").write_bytes(b"fake")
        mocks = _base_patches()
        result = _run(["--folder", str(tmp_path), "--recursive"], mocks)
        assert mocks["processor"].process.call_count == 1


# ---------------------------------------------------------------------------
# Sidecar skip logic
# ---------------------------------------------------------------------------

class TestSidecarSkip:
    def test_skips_video_when_sidecar_exists(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mocks = _base_patches(sidecar_exists_return=True)
        result = _run(["--folder", str(tmp_path)], mocks)
        assert result.exit_code == 0
        mocks["processor"].process.assert_not_called()

    def test_processes_video_when_no_sidecar(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mocks = _base_patches(sidecar_exists_return=False)
        result = _run(["--folder", str(tmp_path)], mocks)
        assert mocks["processor"].process.call_count == 1


# ---------------------------------------------------------------------------
# WhisperModel loading
# ---------------------------------------------------------------------------

class TestWhisperModelLoading:
    def _run_with_whisper_cls(self, tmp_path, args, mock_whisper_cls):
        import types
        runner = CliRunner()
        mock_ollama_client = MagicMock()
        mock_processor = MagicMock()

        mock_ollama_mod = types.ModuleType("ollama")
        mock_ollama_mod.Client = MagicMock(return_value=mock_ollama_client)

        mock_fw_mod = types.ModuleType("faster_whisper")
        mock_fw_mod.WhisperModel = mock_whisper_cls

        with (
            patch.dict("sys.modules", {"ollama": mock_ollama_mod, "faster_whisper": mock_fw_mod}),
            patch("scanner.cli.VideoProcessor", return_value=mock_processor),
            patch("scanner.cli.sidecar_exists", return_value=False),
        ):
            runner.invoke(main, args)

    def test_whisper_model_loaded_with_correct_name(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mock_whisper_cls = MagicMock(return_value=MagicMock())
        self._run_with_whisper_cls(
            tmp_path, ["--folder", str(tmp_path), "--whisper-model", "small"], mock_whisper_cls
        )
        call_args = mock_whisper_cls.call_args
        assert call_args[0][0] == "small"

    def test_whisper_model_default_is_base(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mock_whisper_cls = MagicMock(return_value=MagicMock())
        self._run_with_whisper_cls(
            tmp_path, ["--folder", str(tmp_path)], mock_whisper_cls
        )
        call_args = mock_whisper_cls.call_args
        assert call_args[0][0] == "base"


# ---------------------------------------------------------------------------
# VideoProcessor construction
# ---------------------------------------------------------------------------

class TestVideoProcessorConstruction:
    def _run_with_processor_cls(self, tmp_path, args, mock_processor_cls):
        import types
        runner = CliRunner()
        mock_ollama_client = MagicMock()

        mock_ollama_mod = types.ModuleType("ollama")
        mock_ollama_mod.Client = MagicMock(return_value=mock_ollama_client)

        mock_fw_mod = types.ModuleType("faster_whisper")
        mock_fw_mod.WhisperModel = MagicMock(return_value=MagicMock())

        with (
            patch.dict("sys.modules", {"ollama": mock_ollama_mod, "faster_whisper": mock_fw_mod}),
            patch("scanner.cli.VideoProcessor", mock_processor_cls),
            patch("scanner.cli.sidecar_exists", return_value=False),
        ):
            runner.invoke(main, args)

    def test_processor_created_with_correct_model(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mock_processor_cls = MagicMock(return_value=MagicMock())
        self._run_with_processor_cls(
            tmp_path, ["--folder", str(tmp_path), "--model", "custom:7b"], mock_processor_cls
        )
        _, kwargs = mock_processor_cls.call_args
        assert kwargs["model"] == "custom:7b"

    def test_processor_created_with_correct_prompt(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mock_processor_cls = MagicMock(return_value=MagicMock())
        custom_prompt = "What objects are visible?"
        self._run_with_processor_cls(
            tmp_path, ["--folder", str(tmp_path), "--prompt", custom_prompt], mock_processor_cls
        )
        _, kwargs = mock_processor_cls.call_args
        assert kwargs["prompt"] == custom_prompt


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

class TestSummaryOutput:
    def test_summary_shows_processed_count(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mocks = _base_patches()
        result = _run(["--folder", str(tmp_path)], mocks)
        assert "processed=1" in result.output

    def test_summary_shows_skipped_count(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mocks = _base_patches(sidecar_exists_return=True)
        result = _run(["--folder", str(tmp_path)], mocks)
        assert "skipped=1" in result.output

    def test_summary_shows_failed_count(self, tmp_path):
        _make_video_files(tmp_path, ["clip.mp4"])
        mocks = _base_patches(process_side_effect=RuntimeError("boom"))
        result = _run(["--folder", str(tmp_path)], mocks)
        assert "failed=1" in result.output


# ---------------------------------------------------------------------------
# Required options
# ---------------------------------------------------------------------------

class TestRequiredOptions:
    def test_missing_folder_shows_error(self):
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "missing" in result.output.lower()
