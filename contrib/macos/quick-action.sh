#!/bin/zsh
#
# Finder Quick Action: rename the selected academic PDFs.
#
# This is the script that goes inside the Automator "Run Shell Script" action.
# See contrib/macos/README.md for how to set it up.
#
# Input:  one or more PDF paths, passed by Finder as arguments ("$@").
# Output: each PDF is renamed in place. Progress and errors are written to
#         ~/Library/Logs/rename-academic-pdf.log, and a notification appears
#         if anything did not get renamed.

# 1. Where the tool is installed.
#    Find your own path by running this in Terminal:  which rename-academic-pdf
#    Common locations:
#      /opt/homebrew/bin/rename-academic-pdf   (Homebrew, Apple Silicon)
#      /usr/local/bin/rename-academic-pdf      (Homebrew, Intel)
#      /opt/anaconda3/bin/rename-academic-pdf  (Anaconda)
RENAME="/opt/anaconda3/bin/rename-academic-pdf"

# 2. API keys, only needed if you use the LLM fallback (--llm).
#    A Quick Action runs a NON-INTERACTIVE shell, and that never reads ~/.zshrc.
#    So keys exported in ~/.zshrc are invisible here. Move them to ~/.zshenv,
#    which is read by every zsh, and this line will pick them up.
[ -f "$HOME/.zshenv" ] && source "$HOME/.zshenv"

# 3. Log to a file. Never send errors to /dev/null: if something goes wrong you
#    want to be able to read why, otherwise a failure just looks like "nothing
#    happened when I clicked".
LOG="$HOME/Library/Logs/rename-academic-pdf.log"
mkdir -p "$HOME/Library/Logs"

renamed=0
skipped=0

for pdf in "$@"; do
    echo "===== $(date): $pdf" >>"$LOG"

    # --skip-existing: if a file with the target name is already there, leave it
    # alone and move on, rather than stopping to ask a question that nobody is
    # sitting there to answer.
    if "$RENAME" --skip-existing "$pdf" >>"$LOG" 2>&1; then
        renamed=$((renamed + 1))
    else
        skipped=$((skipped + 1))
    fi
done

# 4. Tell the user if anything did not work. Finder already shows the renames.
if [ "$skipped" -gt 0 ]; then
    osascript -e "display notification \"$renamed renamed, $skipped could not be renamed. See Console > Log Reports, or ~/Library/Logs/rename-academic-pdf.log\" with title \"Rename Academic PDF\""
fi
