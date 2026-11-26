# make_serve_template.py -- strip llama.cpp-only tags from a HF chat_template.jinja
# so the stock Jinja2 used by Strata's native serve can render it. The Inkling-2
# template carries its tool-call instructions (content_thinking,
# content_invoke_tool_json, effort_map, tools) -- those are what make
# frontend.parser_for_template() select InklingOutputParser. Only the
# {% generation %}/{% endgeneration %} markers (llama.cpp context-window
# control, irrelevant to Jinja rendering) are removed; semantics are unchanged.
#
# Usage: python make_serve_template.py <input.jinja> <output.jinja>

import pathlib
import re
import sys

if len(sys.argv) != 3:
    sys.exit("usage: make_serve_template.py <input.jinja> <output.jinja>")

src = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
# generation and endgeneration can appear with or without surrounding whitespace.
out = re.sub(r"\{%-?\s*end?generation\s*-?%\}", "", src)
out = re.sub(r"\{%-?\s*generation\s*-?%\}", "", out)
pathlib.Path(sys.argv[2]).write_text(out, encoding="utf-8", newline="\n")

# Fail early if the result is still not renderable.
try:
    import jinja2
    jinja2.Environment().from_string(out)
except ImportError:
    pass  # jinja2 optional at build time
print("serve template written: %s (%d bytes)" % (sys.argv[2], len(out.encode("utf-8"))))
