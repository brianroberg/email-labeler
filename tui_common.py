"""Shared Textual widgets and screens for this repo's TUIs.

Conventions (issue #43):
- All record-derived text is rendered with ``markup=False`` — otherwise
  bracketed text like ``[f]ilter`` or user content is parsed as Rich markup.
- Single-keypress menus (the ``getch()`` prompt idiom from the old curses
  TUIs) use :class:`KeyMenuScreen`; the :data:`CANCEL` sentinel distinguishes
  "dismissed without choosing" from a chosen value of ``None`` (= clear).
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, ListView, Static


class _Cancel:
    """Dismissal sentinel distinct from None (None = "clear").

    A unique object, not a string, so user-typed text (e.g. a sender filter
    of literally "cancel") can never compare equal to it.
    """

    def __repr__(self) -> str:  # helps debugging dismiss results
        return "CANCEL"


CANCEL = _Cancel()


class PageListView(ListView):
    """ListView with cursor-moving PgUp/PgDn/Home/End (and ^B/^F aliases).

    Stock ListView inherits the scroll-container page bindings, which scroll
    the viewport but leave the cursor behind — the curses TUIs this replaces
    always paged the cursor. Every cursor movement (including the inherited
    up/down) posts :class:`PageListView.UserNavigated` so screens can react
    to *user* navigation without being confused by programmatic index churn
    during list rebuilds.
    """

    BINDINGS = [
        Binding("pageup,ctrl+b", "cursor_page_up", "Page up", show=False),
        Binding("pagedown,ctrl+f", "cursor_page_down", "Page down", show=False),
        Binding("home", "cursor_home", "First", show=False),
        Binding("end", "cursor_end", "Last", show=False),
    ]

    class UserNavigated(Message):
        """The user moved the cursor with a navigation key."""

    def _page(self) -> int:
        return max(1, self.size.height)

    def action_cursor_up(self) -> None:
        super().action_cursor_up()
        self.post_message(self.UserNavigated())

    def action_cursor_down(self) -> None:
        super().action_cursor_down()
        self.post_message(self.UserNavigated())

    def action_cursor_page_up(self) -> None:
        if len(self):
            self.index = max(0, (self.index or 0) - self._page())
        self.post_message(self.UserNavigated())

    def action_cursor_page_down(self) -> None:
        if len(self):
            self.index = min(len(self) - 1, (self.index or 0) + self._page())
        self.post_message(self.UserNavigated())

    def action_cursor_home(self) -> None:
        if len(self):
            self.index = 0
        self.post_message(self.UserNavigated())

    def action_cursor_end(self) -> None:
        if len(self):
            self.index = len(self) - 1
        self.post_message(self.UserNavigated())


class BottomModal(ModalScreen):
    """Base for one-line bottom prompts that replace the curses bottom row.

    Subclasses must dismiss through :meth:`dismiss_once` — key auto-repeat
    (or a fast double-tap) queues a second event behind the dismissal, and
    a raw second ``dismiss()`` pops whatever screen is underneath (or
    crashes the app when the stack empties).
    """

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

    _dismissed = False

    def dismiss_once(self, result) -> None:
        if not self._dismissed:
            self._dismissed = True
            self.dismiss(result)


class HintScreen(BottomModal):
    """Blocking notice (used for errors); any key dismisses."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        yield Static(f"{self._message}  (press any key)", markup=False)

    def on_key(self, event) -> None:
        event.stop()
        self.dismiss_once(None)


class PromptLineScreen(BottomModal):
    """One-line text prompt with prefill. Dismisses stripped text, None on Esc.

    Prefills with *initial* (edit the current value instead of retyping it
    blind). Control characters are stripped on submit — typed ones never
    reach the Input, but a paste can carry them — so a stray Esc or ANSI
    sequence can't pollute the golden set.
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
        clean = "".join(ch for ch in event.value if ch.isprintable())
        self.dismiss_once(clean.strip())

    def action_cancel(self) -> None:
        self.dismiss_once(None)


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
        self.dismiss_once(self._keymap.get(event.key.lower(), CANCEL))
