"""ASCII diamond startup banner for The RAGulator (Psypher Labs).

Renders a neon matrix-green diamond with the tool name and tagline. Uses
``rich`` for truecolor output when available, and falls back to raw ANSI escape
codes so the banner still appears (in colour on most terminals) without any
extra dependency. Printing the banner never raises; display is best-effort.
"""

from __future__ import annotations

# Neon "matrix" green used across Psypher Labs branding.
NEON_GREEN = "#00FF41"
_ANSI_NEON = "\033[38;2;0;255;65m"  # truecolor equivalent of #00FF41
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"

# The diamond. Kept symmetric; "RAG" sits on the widest row so the brand reads
# straight out of the logo.
_DIAMOND = r"""
               *
              ***
             *****
            *******
           *********
          ***********
         **** RAG ****
          ***********
           *********
            *******
             *****
              ***
               *
"""


def render_banner_text() -> str:
    """Return the banner as a plain (uncoloured) string, for logs/tests."""
    return (
        f"{_DIAMOND}\n"
        "                 The RAGulator\n"
        "          it regulates.  -  Psypher Labs\n"
    )


def show_banner() -> None:
    """Print the neon-green diamond banner. Best-effort; never raises."""
    try:
        from rich.console import Console
        from rich.text import Text

        console = Console()
        banner = Text()
        for line in _DIAMOND.strip("\n").splitlines():
            banner.append(line + "\n", style=NEON_GREEN)
        banner.append("\n")
        banner.append("           The RAGulator\n", style=f"bold {NEON_GREEN}")
        banner.append("   it regulates.  ", style=f"{NEON_GREEN}")
        banner.append("\u25C6 ", style=f"bold {NEON_GREEN}")  # ◆ diamond glyph
        banner.append("Psypher Labs\n", style="bold white")
        console.print(banner)
    except Exception:
        # Fallback: raw ANSI so we still get the green diamond without rich.
        try:
            print(f"{_ANSI_NEON}{_DIAMOND}{_ANSI_RESET}")
            print(f"{_ANSI_BOLD}{_ANSI_NEON}           The RAGulator{_ANSI_RESET}")
            print(f"{_ANSI_NEON}   it regulates.  \u25C6 Psypher Labs{_ANSI_RESET}\n")
        except Exception:
            # Last resort: plain text, no colour.
            print(render_banner_text())


if __name__ == "__main__":
    show_banner()
