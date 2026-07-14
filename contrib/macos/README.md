# Right-click to rename, on macOS

This sets up a Finder **Quick Action**, so you can select one or more PDFs,
right-click, and rename them. No Terminal, no commands to remember.

Once installed, the menu item lives under **Quick Actions** in the right-click
menu of any PDF.

## Before you start

You need the tool itself installed. This part does need the Terminal, but only
once:

```bash
pip install rename-academic-pdf
```

Then find out where it landed, because the Quick Action needs the full path:

```bash
which rename-academic-pdf
```

Copy the path it prints. It will look something like
`/opt/homebrew/bin/rename-academic-pdf` or `/opt/anaconda3/bin/rename-academic-pdf`.

## Setting it up

1. Open **Automator** (press Cmd+Space, type "Automator", hit Return).
2. Choose **New Document**, then pick **Quick Action**, then **Choose**.
3. At the top of the right-hand pane, set the two dropdowns:
   - *Workflow receives current* → **PDF files**
   - *in* → **Finder**
4. In the search box on the left, type `Run Shell Script`. Drag it into the big
   empty area on the right.
5. On the Run Shell Script action, set:
   - *Shell* → **/bin/zsh**
   - *Pass input* → **as arguments**  ← this one matters, see below
6. Delete whatever is in the script box, and paste in the contents of
   [`quick-action.sh`](quick-action.sh).
7. Change the `RENAME=` line at the top to the path you copied earlier.
8. Save with Cmd+S. Name it **Rename Academic PDF**.

That's it. Right-click any PDF in Finder and look under **Quick Actions**.

## The first time you run it

macOS will almost certainly show a permission prompt, because a script is
asking to touch files in Downloads or Desktop. **Click OK.** If you dismiss it
by accident, nothing will happen when you use the Quick Action, and you will
get no explanation.

To fix it after the fact, go to **System Settings → Privacy & Security → Files
and Folders** and grant access, or **Full Disk Access** if it is still stuck.

## If nothing happens

Open `~/Library/Logs/rename-academic-pdf.log`. Every run appends to it,
including the full error if something failed. That is the whole reason the
script logs instead of discarding errors.

Common causes, in rough order of likelihood:

**Nothing in the log at all.** The script never ran. That is the permission
prompt above.

**"No such file or directory".** The `RENAME=` path at the top of the script is
wrong. Run `which rename-academic-pdf` again and correct it.

**"OPENAI_API_KEY environment variable is required".** You are using the LLM
fallback, and your key is in `~/.zshrc`. A Quick Action runs a non-interactive
shell, which never reads `~/.zshrc`. Move the line to `~/.zshenv` instead:

```bash
export OPENAI_API_KEY=sk-...
```

`~/.zshenv` is read by every zsh, interactive or not, so both your Terminal and
your Quick Action will see it.

**"already exists".** A PDF with the correct name is already sitting in that
folder, usually because you downloaded the same paper twice and got a
`something (1).pdf`. The Quick Action passes `--skip-existing`, so it leaves
both files alone rather than overwriting. Delete the duplicate yourself, or
swap `--skip-existing` for `--force` in the script if you would rather it
overwrite.

## Why "as arguments" matters

If *Pass input* is set to **to stdin** instead of **as arguments**, the PDF
paths get piped into the script rather than handed to it as `"$@"`, the loop
runs zero times, and the Quick Action silently does nothing at all. It is an
easy setting to miss and it produces no error.
