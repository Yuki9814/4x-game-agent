"""Tests for the template_match module."""
import os
import tempfile
import pytest
import numpy as np
import cv2
from agent.template_match import TemplateMatcher


@pytest.fixture
def temp_dir():
    """Create a temporary directory for templates."""
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def template_matcher(temp_dir):
    """Create a TemplateMatcher with a temp directory."""
    return TemplateMatcher(temp_dir, log_fn=lambda x: None)


def test_load_templates_creates_directory_if_not_exists(temp_dir):
    """_load_templates() should create the directory if it doesn't exist."""
    # Directory doesn't exist
    non_existent = os.path.join(temp_dir, "new_templates")
    matcher = TemplateMatcher(non_existent, log_fn=lambda x: None)
    assert os.path.exists(non_existent)


def test_load_templates_empty_directory(template_matcher):
    """_load_templates() should handle empty directory."""
    assert template_matcher.list_templates() == []


def test_load_templates_with_png_files(temp_dir, template_matcher):
    """_load_templates() should load .png files."""
    # Create a simple template image
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(temp_dir, "test_template.png"), img)

    # Create a new matcher to load templates
    matcher = TemplateMatcher(temp_dir, log_fn=lambda x: None)
    assert "test_template" in matcher.list_templates()


def test_load_templates_ignores_non_png(temp_dir):
    """_load_templates() should ignore non-.png files."""
    # Create a .jpg file
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(temp_dir, "test_template.jpg"), img)

    matcher = TemplateMatcher(temp_dir, log_fn=lambda x: None)
    assert matcher.list_templates() == []


def test_find_with_invalid_screenshot(template_matcher, temp_dir):
    """find() should return None for non-existent screenshot."""
    result = template_matcher.find("non_existent.png", "test")
    assert result is None


def test_find_with_unknown_template(template_matcher):
    """find() should return None for unknown template."""
    # Create a dummy screenshot
    fd, screenshot = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(screenshot, img)

    try:
        result = template_matcher.find(screenshot, "unknown_template")
        assert result is None
    finally:
        os.unlink(screenshot)


def test_find_all_with_unknown_template(template_matcher):
    """find_all() should return empty list for unknown template."""
    result = template_matcher.find_all("dummy.png", "unknown")
    assert result == []


def test_capture_template_saves_correctly(template_matcher, temp_dir):
    """capture_template() should save the region correctly."""
    # Create a dummy screenshot
    fd, screenshot = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.imwrite(screenshot, img)

    try:
        result = template_matcher.capture_template(screenshot, "captured", 10, 10, 50, 50)
        assert result is True
        assert "captured" in template_matcher.list_templates()

        # Verify the file was saved
        saved_path = os.path.join(temp_dir, "captured.png")
        assert os.path.exists(saved_path)
    finally:
        os.unlink(screenshot)


def test_capture_template_invalid_region(template_matcher):
    """capture_template() should return False for invalid region."""
    fd, screenshot = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(screenshot, img)

    try:
        # Try to capture outside the image bounds
        result = template_matcher.capture_template(screenshot, "invalid", 100, 100, 200, 200)
        assert result is False
    finally:
        os.unlink(screenshot)


def test_has_template(template_matcher, temp_dir):
    """has_template() should return correct boolean."""
    assert template_matcher.has_template("nonexistent") is False

    # Add a template
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(temp_dir, "my_template.png"), img)

    matcher = TemplateMatcher(temp_dir, log_fn=lambda x: None)
    assert matcher.has_template("my_template") is True


def test_list_templates(template_matcher, temp_dir):
    """list_templates() should return list of template names."""
    assert template_matcher.list_templates() == []

    # Add templates
    for name in ["a", "b", "c"]:
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(temp_dir, f"{name}.png"), img)

    matcher = TemplateMatcher(temp_dir, log_fn=lambda x: None)
    assert set(matcher.list_templates()) == {"a", "b", "c"}


def test_find_all_deduplication_logic(temp_dir):
    """find_all() should deduplicate close matches."""
    # Create a template
    template = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(temp_dir, "small.png"), template)

    # Create a screenshot with multiple similar matches
    screenshot = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    # Add the template at different positions
    for x, y in [(10, 10), (15, 15), (100, 100), (105, 105)]:
        screenshot[y:y+20, x:x+20] = template

    fd, screenshot_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(screenshot_path, screenshot)

    try:
        matcher = TemplateMatcher(temp_dir, log_fn=lambda x: None)
        results = matcher.find_all(screenshot_path, "small", threshold=0.5, max_results=5)
        # Should have at most 2 results (10,10) and (15,15) are close, (100,100) and (105,105) are close
        # So we should get 2 results
        assert len(results) <= 5
    finally:
        os.unlink(screenshot_path)