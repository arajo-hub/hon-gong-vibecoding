# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

A collection of small, independent coding studies, each in its own numbered `Study-NN/` directory. `Study-01` is a handwritten digit recognizer built from scratch with NumPy (no ML framework). `Study-02` is a personal to-do list app in plain HTML/CSS/JS.

## Commands (Study-01)

Study-01 uses a local virtualenv at `Study-01/.venv` (gitignored, not committed).

```bash
# One-time setup
python3 -m venv Study-01/.venv
Study-01/.venv/bin/pip install numpy pillow

# Train the model — downloads MNIST (mnist.npz, gitignored) on first run,
# then overwrites Study-01/weights.json with freshly trained weights
cd Study-01 && ../Study-01/.venv/bin/python train_mnist.py
```

`web_version/` and `desktop_version/` each have their own `CLAUDE.md` with the commands to run that version — see those before working inside either directory. There is no test suite, linter, or build step in this repository.

## Commands (Study-02)

No build step or dependencies — open `Study-02/index.html` directly in a browser, or serve the folder locally (e.g. `python3 -m http.server` from `Study-02/`) if you want to test it the way a real deployment would be accessed.

## Architecture (Study-01)

- `train_mnist.py` — trains a 784 → 128 (ReLU) → 10 (softmax) MLP with a hand-written forward/backward pass and mini-batch SGD + momentum (no TensorFlow/PyTorch). Downloads MNIST from a fixed Google-hosted URL, then exports the trained weights to `weights.json`, rounded to 4 decimals to keep the file small.
- `weights.json` — the single source of truth for trained weights, shared by every consumer: `desktop_version/model.py` and the `web_version/app.js` browser demo (which fetches it at runtime rather than embedding it, and requires a local HTTP server — see `web_version/CLAUDE.md`). Retraining overwrites this one file; nothing else needs to change. `w1` is shaped `(784, 128)` indexed `[input][hidden]`; `w2` is shaped `(128, 10)` indexed `[hidden][output]`. Any reimplementation of the forward pass (Python, JS, or otherwise) must stay numerically consistent with `MLP.forward` in `train_mnist.py`.
- `web_version/` — browser UI with a drawing canvas; runs the forward pass client-side in vanilla JS. Also published as a standalone Claude Artifact (see the link in `README.md`) that embeds its own copy of `weights.json` inline instead of fetching it.
- `desktop_version/` — native Tkinter UI with a drawing canvas (`draw_app.py`) plus a CLI for classifying image files (`predict.py`); both share preprocessing/inference code via `model.py`.

## Architecture (Study-02)

- `todo-app-prd.md` — the product requirements doc the app was built against; consult it for the intended behavior of any feature before changing it.
- `app.js` — all app state and logic. Todos live in the module-level `todos` array and are persisted as a JSON array under the `todos` key in `localStorage` via `saveTodos`/`loadTodos` (both wrapped in `try/catch` so a corrupted value or an unavailable `localStorage` degrades to an empty in-memory list instead of crashing). Every mutating function (`addTodo`, `toggleComplete`, `handleDelete`, `commitEdit`) calls `saveTodos()` then `renderTodos()` — there is no other path that changes state, so a new feature that mutates `todos` should follow the same pattern rather than patching the DOM directly.
- `renderTodos()` is the single re-render entry point: it recomputes progress (`updateProgress()`), applies the current category filter and completed-last sort (`getVisibleTodos()`), and rebuilds `#todoList` from scratch (including the inline edit row when `editingId` matches). `currentFilter` and `editingId` are the only other pieces of UI state, and both trigger a full `renderTodos()` call when changed rather than being read reactively elsewhere.
- Data model: `{ id, text, category: "work" | "personal" | "study", completed, createdAt, updatedAt }`. `CATEGORY_LABELS` and `PROGRESS_ELEMENT_IDS` are the two lookup tables keyed by category value — extending the category set means updating both, `index.html`'s filter/select markup, and the `category-tag--*`/`category-progress-item--*` CSS variants in `style.css`.
