"""The template and the stylesheet must not disagree.

index.html carried a 95-line inline <style> block alongside a <link> to
styles.css. The inline rules won and hardcoded light colours, so the dark-mode
toggle changed the page background and left every card white.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "templates", "index.html")
STYLESHEET = os.path.join(ROOT, "static", "css", "styles.css")


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_template_carries_no_inline_style_block():
    assert "<style>" not in read(TEMPLATE)


def test_template_does_not_hardcode_a_theme():
    """A hardcoded class on <body> overrides prefers-color-scheme."""
    assert 'class="light"' not in read(TEMPLATE)
    assert 'class="dark"' not in read(TEMPLATE)


def test_every_class_used_has_a_rule():
    template, css = read(TEMPLATE), read(STYLESHEET)
    used = set()
    for value in re.findall(r'class="([^"{}]+)"', template):
        used.update(value.split())
    # severity-<x> is built at render time from the severity value
    used.discard("severity-{{")
    missing = [c for c in used if f".{c}" not in css]
    assert not missing, f"classes with no rule: {missing}"


def test_severity_variants_are_all_styled():
    css = read(STYLESHEET)
    for level in ("high", "medium", "low"):
        assert f".severity-{level}" in css


def test_dark_theme_is_defined_for_both_toggle_and_system():
    css = read(STYLESHEET)
    assert "prefers-color-scheme: dark" in css
    assert "body.dark" in css


def test_colours_come_from_variables_not_literals_in_components():
    """Component rules should read tokens, so both themes stay consistent."""
    css = read(STYLESHEET)
    body_block = css[css.index("body {"):css.index(".mode-toggle")]
    assert "var(--bg)" in body_block and "var(--text)" in body_block
