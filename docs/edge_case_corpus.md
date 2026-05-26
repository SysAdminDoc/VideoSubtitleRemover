# Edge-case corpus -- community contribution guide

RM-55: the regression harness at `tests/test_reference_clips.py` ships
with eight deterministic *synthetic* clips. To strengthen the
harness, we want a curated bank of *real* short clips covering edge
cases the synthetic generator can't reproduce (compression-specific
artefacts, OCR fonts, mixed-language tickers, etc).

This document is the contributor handbook for that corpus.

## What counts as a useful clip

- **Short.** 8-15 seconds, encoded at the source's native resolution
  (do NOT re-encode for upload -- the encode-specific noise is part
  of the test signal).
- **CC0 / public domain only.** Each clip must come from a source
  the contributor has the right to redistribute under the project's
  MIT license. Examples that work: Wikimedia / Library of Congress
  / NASA archive clips; YouTube CC-BY uploads (rare); your own
  capture.
- **Single failure mode.** The clip should isolate ONE thing the
  current pipeline gets wrong. Mixing multiple failure modes makes
  the regression harness noisy.
- **Settings + a description.** Include the exact ProcessingConfig
  (CLI flag string or settings.json snippet) that produced the
  before-screenshot.

## Edge cases we are most interested in

| Category | What we want to cover |
|---|---|
| OCR misreads | Decorative fonts, motion-blur during pan, anti-aliased credits |
| Karaoke | Per-syllable highlight that fuses incorrectly under `--karaoke-grouping` |
| Chyron vs subtitle | News tickers + dialogue in the same lower-third |
| Vertical CJK | Right-edge tategaki on letterboxed footage |
| Thin Latin | <2 px stroke fonts on photographic backgrounds |
| Dissolves | Long dissolve where the histogram detector misses the cut |
| Logo / watermark | Static corner logo (current TBE struggles vs OCR cascade) |
| HDR | bt2020 / smpte2084 source where the SDR tone-map currently lands |

## Submission flow

1. Open a GitHub Discussion in the "Edge cases" category with:
   - The short clip (attached file, or a public CC0 URL).
   - The settings string / JSON.
   - A 1-2 frame before/after screenshot showing the failure.
   - The license declaration ("CC0", "I shot this and grant CC0",
     "Wikimedia Commons + URL", etc.).
2. Maintainers triage. Anything we ingest gets added to a curated
   `tests/clips/` directory and wired into a `ReferenceClipFixture`
   class that loads + processes + asserts a PSNR/SSIM floor.
3. The contributor's GitHub handle is added to a "Corpus credits"
   section in the next release notes.

## What we will NOT add

- Clips from streaming services where the contributor doesn't have
  redistribution rights (Netflix rips, Bluray rips of commercial
  films, etc.). Use a CC0 or public-domain analogue.
- Long clips. Anything past 20 seconds inflates the test suite
  runtime past the CI budget.
- Clips bound to a particular OCR engine's quirks (i.e. the bug
  isn't a VSR bug). File those upstream.

## Compiling baselines

Once a clip is committed, the maintainer runs:

```
python -m backend.processor -i tests/clips/<NAME>.mp4 \
   -o tests/clips/_baselines/<NAME>_cleaned.mp4 \
   --config tests/clips/<NAME>.json --quality-report
```

The reported PSNR / SSIM become the regression floors with a
generous tolerance (+/-0.5 dB PSNR, +/-0.01 SSIM). Future PRs that
breach those bounds fail CI.
