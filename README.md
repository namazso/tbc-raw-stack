# tbc-raw-stack

A median filter for TBCs that don't have VBI frame numbers, such as outputs of vhs-decode and cvbs-decode.

## Purpose

**tbc-raw-stack** is a tool for applying median filter to "raw" TBC sources, such as tape captures via **vhs-decode**. Median filter (aka "stacking") can be used to combine multiple captures of the same media (possibly from a different physical copy) to improve quality and reduce dropouts. It is similar to **ld-disc-stacker** however that tool relies on frame indices extracted from the VBI area of LaserDiscs, while **tbc-raw-stack** relies on the user to line up the starting points, then on heuristics to detect desyncs.

## Usage

### 1. Capture multiple copies

**tbc-raw-stack** can stack a minimum of 3 and a maximum of 15 captures. Using multiple copies or capture devices (like VCRs) may or may not help, but it is recommended that all hardware configurations are represented the same weight. So if you have two VCRs and two tapes, you'd want to do 4, 8, or 12 captures. 

### 2. Decode the captures

Decode your captures with the tool appropriate for the format. If using **vhs-decode**, you must use version 0.3.1 or later. **tbc-raw-stack** expects that the decoder will output fields roughly linear to the capture, i.e., undetected fields are decoded into garbage, not dropped. Field duplicates are the only exception from this, and are gracefully handled.

### 3. Line up start frames

Captures are imperfect, and the starting frames often don't match. Use **ld-analyse** to find the same field in all the captures, and write down its index. Be aware that sometimes the field order is also incorrect if the decoder picks up a bottom field as first. This is supported, you can pass an even number as starting field (although finding it in **ld-analyse** is harder in this case).

### 4. Start stacking

Now, you can run the stacker tool with the earlier information:

```text
tbc-raw-stack --output-basename <OUTPUT_BASENAME>
    --input-basename <INPUT_1_BASENAME> --start-field <INPUT_1_START>
    --input-basename <INPUT_2_BASENAME> --start-field <INPUT_2_START>
    --input-basename <INPUT_3_BASENAME> --start-field <INPUT_3_START>
```

Keep in mind that the first input is special, as most of the metadata is kept from that input. This metadata can be used to align audio, among other things. Please make sure that the first input has the correct field order, as otherwise desyncs will happen.

Once it's complete, you should have the stacked output as `<OUTPUT_BASENAME>`

### 5. Possible problems

#### High MSE warning

If you receive this warning at the beginning of the stacking process, you most likely picked the starting field index incorrectly for the input mentioned in the warning.

If you receive this warning later during stacking, it's likely that the inputs desynchronized unexpectedly. This may be a stacker bug, or a decoder bug. Please submit an issue!

#### Dupe on input / Dupe written

Decode tools may write out duplicate fields if two first or two second fields are found in a row. **tbc-raw-stack** warns you when it happens, and only writes out the earliest dupe, swallowing the dupes of the other inputs.

The `--dupes-to-drops` flag turns dupes into frame drops (by dropping the duped field and the next one). This may be preferred if dupes are happening between clips.

### 6. Advanced usage

Use `tbc-raw-stack --help` to get a full listing of options.

#### Quality metrics

The `--metrics-csv` option, when provided, creates a file with MSE metrics for each field of each input. This can be used to track down desyncs, or to weed out low quality inputs.
