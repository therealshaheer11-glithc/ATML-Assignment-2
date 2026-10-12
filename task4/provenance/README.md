# Historical materials

`releases/` preserves all 19 exact source ZIPs used during preparation and generation. They contain code and small metadata, not wheel caches or weights. `launchers/` holds original standalone Colab helpers. `history/` records contemporaneous observations, approvals and stage instructions; pending statuses describe the time of writing. `planning/` preserves the original requirements review and deliberate-defect distinction.

`EXPORT_PATH_MAP.json` maps four case-colliding export names to distinct worker filenames without changing their bytes or original manifest. `ORIGINAL_HELPERS.json` maps every included original helper/note filename to its canonical publication path and SHA-256. Byte-identical files already in the frozen export are referenced there instead of copied again. Historical scripts keep their original Drive paths, sealed hashes and restart guards. Do not run all historical repair steps blindly on a new runtime.

The current entry points are ../verify_package.py and ../reproduce.py. The current completion state and limitations are in ../README.md and ../RESULTS.md.
