"""Shared Textual widgets and screens for this repo's TUIs.

Conventions (issue #43):
- All record-derived text is rendered with ``markup=False`` — otherwise
  bracketed text like ``[f]ilter`` or user content is parsed as Rich markup.
- Single-keypress menus (the curses ``getch()`` prompt idiom) use
  :class:`KeyMenuScreen`; the :data:`CANCEL` sentinel distinguishes
  "dismissed without choosing" from a chosen value of ``None`` (= clear).
"""

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

CANCEL = "cancel"  # dismissal sentinel distinct from None (None = "clear")


class KeyMenuScreen(ModalScreen):
    """Single-keypress menu rendered as a one-line prompt at the bottom.

    Dismisses with ``keymap[key]`` for a mapped key (case-insensitive),
    or with :data:`CANCEL` for any other key — mirroring the curses
    "press one key, anything else cancels" prompt idiom.
    """

    DEFAULT_CSS = """
    KeyMenuScreen {
        align: left bottom;
        background: $background 0%;
    }
    KeyMenuScreen > Static {
        width: 100%;
        background: $accent;
        color: $text;
    }
    """

    def __init__(self, prompt: str, keymap: dict) -> None:
        super().__init__()
        self._prompt = prompt
        self._keymap = keymap

    def compose(self) -> ComposeResult:
        yield Static(self._prompt, id="key-menu-prompt", markup=False)

    def on_key(self, event) -> None:
        event.stop()
        self.dismiss(self._keymap.get(event.key.lower(), CANCEL))
