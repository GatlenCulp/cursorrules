# cursorrules

Shared rules for AI agents interacting with my projects.

<!-- For self: https://github.com/GatlenCulp/gatlens_prompt_engineering/tree/main/src/content -->

## Installation options

To copy everything into an existing project while prompting for overwrites, run the following:

```bash
git clone https://github.com/GatlenCulp/cursorrules.git
# -r (recursive)
# -n (no-clobber): Don't overwrite existing files OR -i (interactive): Ask to overwrite
cp -ri cursorrules/.cursor /path/to/your/project/
```

## Descriptions

For more information about the cursor features and what I have defined, see [`.cursor/README.md`](.cursor/README.md)

<!-- TODO: [cursorrules] Tell cursor never to write ascii characters (perhaps? Idk.) -->

<!-- TODO: [cursorrules] Never write the full docstring (e.g. arguments) unless asked as arguments may also change. Only add args that may be confusing. It clutters everything. Instead just write a one-line docstring.  -->

<!-- TODO: [cursorrules] Always ask that in pydantic, the created time be separate from the id, run_id, etc. -->

// TODO: [cursorrules] let todo structure be TODO(person1, person2): [tag1, tag2] <unique-title> Description. All in kebab-case
// TODO: [cursorrules] Use {I am a long name, yes I am, name3} in text to better denote I am a long name / yes I am / name3 in text because english syntax is stupid
// TODO: [cursorrules] Always ask to select a "canon" name and footnote alternatives names and explanations behind the names if not obvious. Or look into glossorium options? Also always start footnotes on newline. Also always indent when making a list.
// TODO: [cursorrules] Define the #%% syntax I use for python notebooks and how best to do that
// TODO: [cursorrules] Tell cursor to enforce the role of kebab-case for private modules / notebooks (meaning no others import)? Idk if a good idea. (also why am I being so OCD I have to stop.)

<!-- TODO: [cursorrules] Only ever include notational details at the end.-->

<!--

-->

<!--
TODO: [cursorrules] command generate notation:

At the end of the document, generate a summary of all notation used throughout the document.
-->

<!-- TODO: [cursorrules] describe ruff config setup. -->
