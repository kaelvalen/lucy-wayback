# Save-format fixtures

One frozen save per supported format, so every migration stays tested
(`save::tests::fixtures_migrate_and_continue_like_their_own_binary`). Never regenerate an old
fixture: it stands for tissues that already exist on disk.

- `corpus.txt`: synthetic CHILDES-like dialogue (40 KB, generated with `random.Random(7)`; no
  corpus text).
- `vNN-params.json`: the rule constants of the fixture run. They exercise depth 64, classes and
  pruning, and `ctx_mix` from format 5 on.
- `vNN.bin`: a save in format NN, written by the binary of that format.
- `vNN-continue.jsonl`: what that same binary wrote when it resumed the save.

The test loads each `vNN.bin` with the current code (migrating it), continues it, and requires
exactly these metrics lines, minus wall clock.

Recipe, with `B` = a `lucy` binary built from a commit of format NN (format 4: `857a46f`):

    B run --corpus corpus.txt --ticks 160000 --metrics-every 16000 --params vNN-params.json \
      --metrics /tmp/first.jsonl --save vNN.bin
    : > vNN-continue.jsonl
    B run --corpus corpus.txt --ticks 320000 --metrics-every 16000 --resume vNN.bin \
      --metrics vNN-continue.jsonl

When a layout changes (format N → N+1), add the frozen structs and the migration (see the
header of `crates/lucy-core/src/save.rs`), then add `v(N+1)` here with the recipe above.
`vN` stays as it is.
