"""Shared Textual widgets and screens for this repo's TUIs.

Conventions (issue #43):
- All record-derived text is rendered with ``markup=False`` — otherwise
  bracketed text like ``[f]ilter`` or user content is parsed as Rich markup.
- Single-keypress menus (the curses ``getch()`` prompt idiom) use
  :class:`KeyMenuScreen`; the :data:`CANCEL` sentinel distinguishes
  "dismissed without choosing" from a chosen value of ``None`` (= clear).
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Input, ListView, Static

CANCEL = "cancel"  # dismissal sentinel distinct from None (None = "clear")


class PageListView(ListView):
    """ListView whose PgUp/PgDn move the cursor by a page, not just the scroll.

    Stock ListView inherits the scroll-container page bindings, which scroll
    the viewport but leave the cursor behind — the curses TUIs this replaces
    always paged the cursor.
    """

    BINDINGS = [
        Binding("pageup", "cursor_page_up", "Page up", show=False),
        Binding("pagedown", "cursor_page_down", "Page down", show=False),
    ]

    def _page(self) -> int:
        return max(1, self.size.height)

    def action_cursor_page_up(self) -> None:
        if len(self):
            self.index = max(0, (self.index or 0) - self._page())

    def action_cursor_page_down(self) -> None:
        if len(self):
            self.index = min(len(self) - 1, (self.index or 0) + self._page())


class BottomModal(ModalScreen):
    """Base for one-line bottom prompts that replace the curses bottom row."""

    DEFAULT_CSS = """
    BottomModal {
        align: left bottom;
        background: $background 0%;
    }
    BottomModal > Static {
        width: 100%;
        background: $accent;
        color: $text;
    }
    """


class HintScreen(BottomModal):
    """Blocking notice (used for errors); any key dismisses."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        yield Static(f"{self._message}  (press any key)", markup=False)

    def on_key(self, event) -> None:
        event.stop()
        self.dismiss(None)


class PromptLineScreen(BottomModal):
    """One-line text prompt with prefill. Dismisses stripped text, None on Esc.

    Prefills with *initial* (edit the current value instead of retyping it
    blind); Input rejects control characters, so a stray Esc can't pollute
    the golden set.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, prompt: str, initial: str = "") -> None:
        super().__init__()
        self._prompt = prompt
        self._initial = initial

    def compose(self) -> ComposeResult:
        yield Static(self._prompt, markup=False)
        yield Input(value=self._initial, id="prompt-input")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


class KeyMenuScreen(BottomModal):
    """Single-keypress menu rendered as a one-line prompt at the bottom.

    Dismisses with ``keymap[key]`` for a mapped key (case-insensitive),
    or with :data:`CANCEL` for any other key — mirroring the curses
    "press one key, anything else cancels" prompt idiom.
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
