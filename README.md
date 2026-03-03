# anki-gen

Generate Anki flashcards from Markdown notes using an LLM.

Local model support soon. Currently supports API usage for OpenAI and Anthropic

## Features

- Parses `.md` files and generates flashcards via OpenAI or Anthropic
- Three card types: **basic**, **basic (reversed)**, **definition**
- Interactive TUI (`--confirm`) to review, edit, delete, tag, and mark concepts for reversal before generating cards
- Exports to `.apkg` or pushes directly into a running Anki via **AnkiConnect**
- MathJax support for mathematical notation
- Syntax-highlighted code blocks

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

Copy `.env.example` to `.env` and fill in your API key(s):

```bash
cp .env.example .env
```

## Usage

```
anki-gen [OPTIONS] PATH [PATH ...]
```

### Basic — generate and export to `.apkg`

```bash
anki-gen notes.md
anki-gen lectures/          # recursively finds all .md files
```

### Push directly to Anki (requires AnkiConnect add-on)

```bash
anki-gen notes.md --push
anki-gen notes.md --push --deck "My Deck"
```

### Interactive concept review

```bash
anki-gen notes.md --confirm
```

Opens a TUI after concept extraction so you can review the list before cards are generated.

**TUI keys:**

| Key | Action |
|-----|--------|
| `j` / `↓` | Move down |
| `k` / `↑` | Move up |
| `d` | Toggle delete on current item |
| `e` | Edit current item inline |
| `r` | Toggle reversal (`⇄`) on current concept |
| `R` | Toggle reversal on **all** non-deleted concepts |
| `t` | Edit per-concept tags (separate field; concept text is read-only) |
| `a` | Add a new concept |
| `n` | Add a note / instruction for the LLM |
| `Enter` | Confirm and generate cards |
| `q` | Abort |

Concepts marked `⇄` generate **basic (reversed)** cards — Anki tests both directions.

### Global tags

Apply tags to every generated card:

```bash
anki-gen notes.md --tags math,calculus,exam-prep
```

Spaces within a tag are replaced with hyphens automatically.

### Other options

```
-o, --output FILE     Write to a specific .apkg path
-d, --deck NAME       Override the deck name
-n, --max-cards N     Cap the number of cards generated per file
-v, --verbose         Show concept lists, card-type summary, and tag breakdown
--provider            openai (default) or anthropic
-m, --model MODEL_ID  Override the model (e.g. gpt-4o-mini)
--color COLOR         Animation colour: blue (default), indigo, orange, red, green
```

## Requirements

- OpenAI or Anthropic API key
- For `--push`: [AnkiConnect](https://ankiweb.net/shared/info/2055492159) add-on with Anki running
- For `.apkg` export: `genanki` (installed automatically)
