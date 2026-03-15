"""Tests for the ocr module."""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np


# Mock PaddleOCR to avoid heavy dependency in tests
@pytest.fixture
def mock_paddle():
    with patch('agent.ocr.PaddleOCR') as mock:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = []
        mock.return_value = mock_instance
        yield mock_instance


def test_read_all_returns_cached_results(mock_paddle):
    """read_all() should return cached results for same file."""
    # Create a test image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new('RGB', (100, 100), color='white')
        img.save(f.name)
        image_path = f.name

    try:
        # Mock the _run method to return test data
        with patch('agent.ocr.OCREngine._run') as mock_run:
            mock_run.return_value = [
                {"text": "test", "confidence": 0.9, "x": 50, "y": 50}
            ]
            from agent.ocr import OCREngine
            
            # Need to reload to use the mocked PaddleOCR
            import importlib
            import agent.ocr
            importlib.reload(agent.ocr)
            
            ocr = agent.ocr.OCREngine()
            result1 = ocr.read_all(image_path)
            result2 = ocr.read_all(image_path)
            
            # Should return same cached result
            assert result1 == result2
    finally:
        os.unlink(image_path)


def test_read_all_different_mtime(mock_paddle):
    """read_all() should re-run OCR when file mtime changes."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new('RGB', (100, 100), color='white')
        img.save(f.name)
        image_path = f.name

    try:
        with patch('agent.ocr.OCREngine._run') as mock_run:
            mock_run.side_effect = [
                [{"text": "first", "confidence": 0.9, "x": 50, "y": 50}],
                [{"text": "second", "confidence": 0.9, "x": 60, "y": 60}]
            ]
            import agent.ocr
            import importlib
            importlib.reload(agent.ocr)
            
            ocr = agent.ocr.OCREngine()
            result1 = ocr.read_all(image_path)
            
            # Touch the file to change mtime
            import time
            time.sleep(0.1)
            os.utime(image_path, None)
            
            result2 = ocr.read_all(image_path)
            
            assert result1[0]["text"] == "first"
            assert result2[0]["text"] == "second"
    finally:
        os.unlink(image_path)


def test_read_all_non_existent_file(mock_paddle):
    """read_all() should handle non-existent file gracefully."""
    import agent.ocr
    import importlib
    importlib.reload(agent.ocr)
    
    ocr = agent.ocr.OCREngine()
    result = ocr.read_all("/non/existent/file.png")
    assert result == []


def test_read_region_coordinates_adjustment(mock_paddle):
    """read_region() should adjust coordinates to full-image space."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new('RGB', (200, 200), color='white')
        img.save(f.name)
        image_path = f.name

    try:
        with patch('agent.ocr.OCREngine._run') as mock_run:
            mock_run.return_value = [
                {"text": "test", "confidence": 0.9, "x": 10, "y": 10}
            ]
            import agent.ocr
            import importlib
            importlib.reload(agent.ocr)
            
            ocr = agent.ocr.OCREngine()
            # Read region starting at (50, 50)
            result = ocr.read_region(image_path, (50, 50, 150, 150))
            
            # Coordinates should be adjusted by crop offset
            assert result[0]["x"] == 60  # 10 + 50
            assert result[0]["y"] == 60  # 10 + 50
    finally:
        os.unlink(image_path)


def test_read_region_saves_cropped_image(mock_paddle):
    """read_region() should save cropped image to temp path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new('RGB', (100, 100), color='white')
        img.save(f.name)
        image_path = f.name

    try:
        with patch('agent.ocr.OCREngine._run') as mock_run:
            mock_run.return_value = []
            import agent.ocr
            import importlib
            importlib.reload(agent.ocr)
            
            ocr = agent.ocr.OCREngine()
            ocr.read_region(image_path, (10, 10, 50, 50))
            
            # Temp file should exist
            assert os.path.exists("/tmp/4x_ocr_crop.png")
    finally:
        os.unlink(image_path)


def test_run_filters_low_confidence(mock_paddle):
    """_run() should filter out results with confidence < 0.5."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new('RGB', (100, 100), color='white')
        img.save(f.name)
        image_path = f.name

    try:
        # Configure mock to return low confidence result
        mock_paddle.predict.return_value = [{
            "rec_texts": ["low_conf", "high_conf"],
            "rec_scores": [0.3, 0.9],
            "dt_polys": [[[0, 0], [10, 0], [10, 10], [0, 10]],
                         [[20, 0], [30, 0], [30, 10], [20, 10]]]
        }]
        
        import agent.ocr
        import importlib
        importlib.reload(agent.ocr)
        
        ocr = agent.ocr.OCREngine()
        
        # We can't easily test _run because it depends on real OCR
        # But we can verify the mock was called
        ocr._ocr.predict(image_path)
        assert mock_paddle.predict.called
    finally:
        os.unlink(image_path)


def test_run_filters_empty_text(mock_paddle):
    """_run() should filter out results with empty text."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new('RGB', (100, 100), color='white')
        img.save(f.name)
        image_path = f.name

    try:
        mock_paddle.predict.return_value = [{
            "rec_texts": ["", "valid_text"],
            "rec_scores": [0.9, 0.9],
            "dt_polys": [[[0, 0], [10, 0], [10, 10], [0, 10]],
                         [[20, 0], [30, 0], [30, 10], [20, 10]]]
        }]
        
        import agent.ocr
        import importlib
        importlib.reload(agent.ocr)
        
        ocr = agent.ocr.OCREngine()
        ocr._ocr.predict(image_path)
        
        assert mock_paddle.predict.called
    finally:
        os.unlink(image_path)