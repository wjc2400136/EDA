# Script Index

[English](SCRIPT_INDEX.md) | [Simplified Chinese](SCRIPT_INDEX.zh-CN.md)

The anonymous package exposes one experiment entry point:

| Script | Purpose |
|---|---|
| experiments/run_multi_source.py | Generate and evaluate the six compared attacks for the principal CNN-source transfer tables. |

Use --mode generate, --mode eval, or --mode both to select the workflow.
Use --attack for one method or --attacks for a comma-separated method list.
Run python experiments/run_multi_source.py --help for all options.
