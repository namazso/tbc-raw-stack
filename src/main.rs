//  This Source Code Form is subject to the terms of the Mozilla Public
//  License, v. 2.0. If a copy of the MPL was not distributed with this
//  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]

mod median;
mod simd_x86_128;
mod simd_x86_256;
mod simd_x86_512;
mod tbc_metadata;

use crate::tbc_metadata::{System, TbcMetadata, VitsMetrics};
use clap::Parser;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, span, trace, warn, Level};
use tracing_subscriber::EnvFilter;

/// Stack multiple tapes
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input basenames
    #[arg(short, long)]
    input_basename: Vec<String>,

    /// Field index to start with, for each input (1-based)
    #[arg(short, long)]
    start_field: Vec<usize>,

    /// Output basename
    #[arg(short, long)]
    output_basename: String,

    /// How many fields to process (0 = all)
    #[arg(short = 'c', long, default_value_t = 0)]
    max_fields: usize,

    /// How many inputs should agree on having a dropout to mark it as such [default: ceil(inputs_count / 2)]
    #[arg(short, long)]
    dropout_threshold: Option<usize>,

    /// Convert duplicated frames to drops
    #[arg(long, default_value_t = false)]
    dupes_to_drops: bool,

    /// If provided, write field mappings
    #[arg(long)]
    fieldmap_csv: Option<PathBuf>,

    /// If provided, write RMSE pSNR
    #[arg(long)]
    metrics_csv: Option<PathBuf>,
}

struct InputTbc {
    index: usize,
    metadata: TbcMetadata,
    tbc: BufReader<File>,
    chroma: BufReader<File>,
    field_index: usize,
    dupe_count: usize,
    last_seq_no: usize,
}

unsafe fn to_bytes<T>(input: &[T]) -> &[u8] {
    let ptr = input as *const [T] as *const u8; // Cast slice of T to a slice of u8
    let len = input.len() * size_of::<T>(); // Calculate the length in bytes
    std::slice::from_raw_parts(ptr, len) // Create a slice of u8 from the raw pointer
}
unsafe fn to_bytes_mut<T>(input: &mut [T]) -> &mut [u8] {
    let ptr = input as *mut [T] as *mut u8; // Cast slice of T to a mutable slice of u8
    let len = input.len() * size_of::<T>(); // Calculate the length in bytes
    std::slice::from_raw_parts_mut(ptr, len) // Create a mutable slice of u8 from the raw pointer
}

const MAX_SAMPLES_PER_FIELD: usize = 0x57000;
const MIN_INPUT_STREAMS: usize = 3;
const MAX_INPUT_STREAMS: usize = 15;

const RMSE_WARN_THRESHOLD: usize = 30;

// 355 255 PAL samples * 512 * 2 channels = ~347 MB per input
// 347 MB * (15 input + 1 output) = 5.552 GB total memory usage
// since 512 is also the default sector size, it may help with storage stuff too...
const IO_BUFFER_MULTIPLIER: usize = 512;

struct SystemConstants {
    /// Start sample for calculating black pSNR
    black_start_sample: usize,

    /// End sample for calculating black pSNR
    black_end_sample: usize,

    /// Start sample for calculating RMSE pSNR
    useful_start_sample: usize,

    /// End sample for calculating RMSE pSNR
    useful_end_sample: usize,

    /// Difference between black and white
    psnr_scale: f32,
}

impl SystemConstants {
    fn error_to_psnr(&self, error: f32) -> f32 {
        20. * (self.psnr_scale / error).log10()
    }
}

const SYSTEM_PAL: SystemConstants = SystemConstants {
    black_start_sample: 24048,
    black_end_sample: 24928,    // 24 935 originally but we pick a nicer number
    useful_start_sample: 61312, // line 55
    useful_end_sample: 258752,  // line 229
    psnr_scale: 0.7 * (0xD300 - 0x0100) as f32,
};

const SYSTEM_NTSC: SystemConstants = SystemConstants {
    black_start_sample: 144,    // 143 originally
    black_end_sample: 432,      // 429 originally
    useful_start_sample: 27328, // line 31
    useful_end_sample: 209280,  // line 231
    psnr_scale: 0.75 * (0xC800 - 0x0400) as f32,
};

fn calculate_bpsnr(field: &[u16], constants: &SystemConstants) -> f32 {
    let region = &field[constants.black_start_sample..constants.black_end_sample];
    let len = region.len();
    assert_eq!(len % 16, 0);
    let mut sum = 0u32;
    for chunk in region.chunks_exact(16) {
        let chunk: &[u16; 16] = chunk.try_into().unwrap();
        for v in chunk {
            sum += *v as u32;
        }
    }
    let mean = sum as f32 / len as f32;
    let mut variance = 0f32;
    for chunk in region.chunks_exact(16) {
        let chunk: &[u16; 16] = chunk.try_into().unwrap();
        for v in chunk {
            let dev = *v as f32 - mean;
            variance += dev * dev;
        }
    }
    let stddev = (variance / len as f32).sqrt();
    constants.error_to_psnr(stddev)
}

#[repr(align(64))]
#[derive(Copy, Clone)]
struct FieldBuffer([u16; MAX_SAMPLES_PER_FIELD]);

impl Default for FieldBuffer {
    fn default() -> Self {
        FieldBuffer([0; MAX_SAMPLES_PER_FIELD]) // Initialize the array with zeros
    }
}

fn main() {
    let level = std::env::var("RUST_LOG").unwrap_or_else(|_| {
        format!("{}=info", env!("CARGO_PKG_NAME").replace("-", "_")).to_string()
    });
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(level.as_str()))
        .init();

    let args = Args::parse();

    if !(MIN_INPUT_STREAMS..MAX_INPUT_STREAMS).contains(&args.input_basename.len()) {
        panic!(
            "Invalid number of inputs, must be between {MIN_INPUT_STREAMS} and {MAX_INPUT_STREAMS}"
        );
    }

    if args.input_basename.len() != args.start_field.len() {
        panic!("Count of input parameters and start field parameters is not equal!");
    }

    let mut inputs = args
        .input_basename
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let json = p.clone() + ".tbc.json";
            let tbc = p.clone() + ".tbc";
            let chroma = p.clone() + "_chroma.tbc";
            let start_field = args.start_field[i] - 1;

            let metadata: TbcMetadata =
                serde_json::from_reader(File::open(json).expect("Cannot open input JSON metadata"))
                    .expect("Cannot parse JSON metadata");
            let field_size =
                metadata.video_parameters.field_height * metadata.video_parameters.field_width;
            let field_bytes = field_size * 2;
            let tbc_file = File::open(tbc).expect("Cannot open tbc file");
            let mut tbc_file =
                BufReader::with_capacity(field_size * IO_BUFFER_MULTIPLIER, tbc_file);
            tbc_file
                .seek(SeekFrom::Start((field_bytes * start_field) as u64))
                .expect("Cannot seek to start field");
            let chroma_file = File::open(chroma).expect("Cannot open chroma file");
            let mut chroma_file =
                BufReader::with_capacity(field_size * IO_BUFFER_MULTIPLIER, chroma_file);
            chroma_file
                .seek(SeekFrom::Start((field_bytes * start_field) as u64))
                .expect("Cannot seek to start field");

            InputTbc {
                index: i,
                metadata,
                tbc: tbc_file,
                chroma: chroma_file,
                field_index: start_field,
                dupe_count: start_field % 2,
                last_seq_no: 0,
            }
        })
        .collect::<Vec<_>>();

    if inputs[0].dupe_count != 0 {
        panic!("The first input must have correct field order!")
    }

    let system = inputs[0].metadata.video_parameters.system.clone();
    let sys = if system == System::Pal {
        &SYSTEM_PAL
    } else {
        &SYSTEM_NTSC
    };

    let dropout_threshold = args.dropout_threshold.unwrap_or(inputs.len().div_ceil(2));

    let field_width = inputs[0].metadata.video_parameters.field_width;
    let field_height = inputs[0].metadata.video_parameters.field_height;
    let field_size = field_width * field_height;
    let field_size_rounded = field_size.div_ceil(32) * 32;

    let max_fields = args.max_fields;

    let mut out_luma = {
        let path = args.output_basename.clone() + ".tbc";
        let file = File::create_new(path).expect("Cannot create tbc file");
        BufWriter::with_capacity(field_size * IO_BUFFER_MULTIPLIER, file)
    };
    let mut out_chroma = {
        let path = args.output_basename.clone() + "_chroma.tbc";
        let file = File::create_new(path).expect("Cannot create tbc file");
        BufWriter::with_capacity(field_size * IO_BUFFER_MULTIPLIER, file)
    };
    let mut out_fields: Vec<tbc_metadata::Field> = Vec::new();
    let mut out_metrics = args.metrics_csv.map(|f| {
        let file = File::create_new(f).expect("Cannot open metrics file");
        BufWriter::new(file)
    });
    let mut out_fieldmap = args.fieldmap_csv.map(|f| {
        let file = File::create_new(f).expect("Cannot open metrics file");
        BufWriter::new(file)
    });

    let mut dupes_written = 0usize;

    let mut new_luma = Box::new(<FieldBuffer>::default());
    let new_luma = &mut new_luma.0.as_mut_slice()[0..field_size_rounded];
    let mut new_chroma = Box::new(<FieldBuffer>::default());
    let new_chroma = &mut new_chroma.0.as_mut_slice()[0..field_size_rounded];
    let mut new_field = inputs[0].metadata.fields[inputs[0].field_index].clone();

    let mut in_luma = vec![<FieldBuffer>::default(); inputs.len()];
    let mut in_luma = in_luma.iter_mut().map(|f| f.0.as_mut()).collect::<Vec<_>>();
    let mut in_chroma = vec![<FieldBuffer>::default(); inputs.len()];
    let mut in_chroma = in_chroma
        .iter_mut()
        .map(|f| f.0.as_mut())
        .collect::<Vec<_>>();

    let mut sse_luma = vec![0u64; inputs.len()];
    let mut sse_luma_edge = vec![0u64; inputs.len()];
    let mut sse_chroma = vec![0u64; inputs.len()];
    let mut rmse_bad_in_a_row = vec![0usize; inputs.len()];

    let now = Instant::now();

    let mut drop_next = false;

    loop {
        let new_field_idx = out_fields.len();

        let _span = span!(Level::INFO, "field", idx = new_field_idx + 1).entered();

        if max_fields != 0 && out_fields.len() == max_fields {
            // we exported the requested count of fields
            break;
        }

        if inputs
            .iter()
            .any(|i| i.field_index == i.metadata.fields.len())
        {
            // one of the inputs ended
            break;
        }

        let mut should_write_dupe = false;
        for f in &mut inputs {
            if f.metadata.fields[f.field_index].seq_no <= f.last_seq_no {
                warn!(
                    "Dupe in input #{}, at field {}",
                    f.index + 1,
                    f.field_index + 1
                );
                if f.dupe_count % 2 == dupes_written % 2 {
                    // we only actually write out a dupe if it looks "new"
                    should_write_dupe = true;
                }
                f.dupe_count += 1;
                f.field_index += 1;
                f.tbc.seek_relative((field_size * 2) as i64).unwrap();
                f.chroma.seek_relative((field_size * 2) as i64).unwrap();
            }
        }

        // let's check it again after the dupe skipping
        if inputs
            .iter()
            .any(|i| i.field_index == i.metadata.fields.len())
        {
            break;
        }

        if should_write_dupe {
            dupes_written += 1;
            if args.dupes_to_drops {
                warn!("Dropping dupe field and the following one");
                drop_next = true;
                continue;
            } else {
                warn!("Writing out dupe");
            }
        } else {
            {
                new_field.seq_no = new_field_idx + 1;
                let str = inputs
                    .iter()
                    .map(|i| (i.field_index + 1).to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                trace!("Generating from fields {}", str);
                if let Some(fieldmap) = out_fieldmap.as_mut() {
                    fieldmap
                        .write_all(format!("{},{}\n", new_field_idx + 1, str).as_bytes())
                        .unwrap();
                }
            }

            new_field = inputs[0].metadata.fields[inputs[0].field_index].clone();

            for i in 0..inputs.len() {
                inputs[i]
                    .tbc
                    .read_exact(unsafe { to_bytes_mut(&mut in_luma[i][0..field_size]) })
                    .unwrap();
                inputs[i]
                    .chroma
                    .read_exact(unsafe { to_bytes_mut(&mut in_chroma[i][0..field_size]) })
                    .unwrap();
            }

            // We calculate median luma in 3 parts, because we only want the SSE of the middle bits.
            // The rest may be garbage due to head switch, and we don't want it to skew the numbers.
            median::batch_n(
                &mut new_luma[0..sys.useful_start_sample],
                in_luma
                    .iter()
                    .map(|f| &(**f)[0..sys.useful_start_sample])
                    .collect::<Vec<_>>()
                    .as_slice(),
                &mut sse_luma_edge[..],
            );
            median::batch_n(
                &mut new_luma[sys.useful_start_sample..sys.useful_end_sample],
                in_luma
                    .iter()
                    .map(|f| &(**f)[sys.useful_start_sample..sys.useful_end_sample])
                    .collect::<Vec<_>>()
                    .as_slice(),
                &mut sse_luma[..],
            );
            median::batch_n(
                &mut new_luma[sys.useful_end_sample..field_size_rounded],
                in_luma
                    .iter()
                    .map(|f| &(**f)[sys.useful_end_sample..field_size_rounded])
                    .collect::<Vec<_>>()
                    .as_slice(),
                &mut sse_luma_edge[..],
            );

            median::batch_n(
                new_chroma,
                in_chroma
                    .iter()
                    .map(|f| &(**f)[0..field_size_rounded])
                    .collect::<Vec<_>>()
                    .as_slice(),
                &mut sse_chroma[..],
            );

            new_field.vits_metrics = Some(VitsMetrics {
                bpsnr: calculate_bpsnr(&new_luma[0..field_size], sys) as f64,
                other: Default::default(),
            });

            #[derive(PartialEq, Eq)]
            enum Dropout {
                Start,
                End,
            }

            let mut flat_dropouts = inputs
                .iter()
                .flat_map(|i| {
                    if let Some(dropouts) = &i.metadata.fields[i.field_index].drop_outs {
                        let mut out = vec![];
                        for j in 0..dropouts.field_line.len() {
                            let line = dropouts.field_line[j];
                            if line >= field_height {
                                continue; // WTF?
                            }
                            let startx = dropouts.startx[j];
                            let endx = dropouts.endx[j];
                            out.push((line * field_width + startx, Dropout::Start));
                            out.push((line * field_width + endx, Dropout::End));
                        }
                        out
                    } else {
                        vec![]
                    }
                })
                .collect::<Vec<_>>();
            flat_dropouts.sort_unstable_by(|a, b| a.0.cmp(&b.0));

            new_field.drop_outs = if flat_dropouts.is_empty() {
                None
            } else {
                let mut out_dropouts = tbc_metadata::DropOuts {
                    field_line: vec![],
                    startx: vec![],
                    endx: vec![],
                };
                let mut depth = 0usize;
                let mut start = 0usize;
                for (sample, do_type) in flat_dropouts {
                    if do_type == Dropout::Start {
                        depth += 1;
                        if depth == dropout_threshold {
                            start = sample;
                        }
                    } else {
                        if depth == dropout_threshold {
                            let line = start / field_width;
                            let startx = start - line * field_width;
                            let endx = sample - line * field_width;
                            out_dropouts.field_line.push(line);
                            out_dropouts.startx.push(startx);
                            out_dropouts.endx.push(endx);
                        }
                        depth -= 1;
                    }
                }
                Some(out_dropouts)
            };

            for i in &mut inputs {
                i.last_seq_no = i.metadata.fields[i.field_index].seq_no;
                i.field_index += 1;
            }
        }

        if drop_next {
            drop_next = false;
            continue;
        }

        {
            let useful_size = sys.useful_end_sample - sys.useful_start_sample;
            let rmse_psnr = sse_luma
                .iter()
                .map(|f| sys.error_to_psnr((*f as f32 / useful_size as f32).sqrt()))
                .collect::<Vec<_>>();

            let str = rmse_psnr
                .iter()
                .map(|v| format!("{}", v))
                .collect::<Vec<_>>()
                .join(",");
            trace!("RMSE pSNR: {}", str);
            if let Some(metrics) = out_metrics.as_mut() {
                metrics
                    .write_all(format!("{},{}\n", new_field_idx + 1, str).as_bytes())
                    .unwrap();
            }
            let sum = rmse_psnr.iter().sum::<f32>();
            for (i, &v) in rmse_psnr.iter().enumerate() {
                let avg_of_others = (sum - v) / ((inputs.len() - 1) as f32);
                if v < 32. && v < avg_of_others - 5. {
                    rmse_bad_in_a_row[i] += 1;
                    if rmse_bad_in_a_row[i] % RMSE_WARN_THRESHOLD == 0 {
                        warn!(
                        "RMSE pSNR on input #{} has been very high for {} fields: {}. Bad source or desync?",
                        i + 1,
                            rmse_bad_in_a_row[i],
                        v
                    );
                    }
                } else {
                    rmse_bad_in_a_row[i] = 0;
                }
            }
        }

        out_luma
            .write_all(unsafe { to_bytes(&new_luma[0..field_size]) })
            .unwrap();
        out_chroma
            .write_all(unsafe { to_bytes(&new_chroma[0..field_size]) })
            .unwrap();
        out_fields.push(new_field.clone());
    }

    let frames = out_fields.len() / 2;
    let secs = now.elapsed().as_secs_f64();
    let fps = frames as f64 / secs;
    info!("Processed {frames} frames in {secs}s ({fps} FPS)");

    for (idx, field) in out_fields.iter_mut().enumerate() {
        field.is_first_field = idx % 2 == 0;
    }

    let mut out_meta = inputs[0].metadata.clone();
    out_meta.video_parameters.number_of_sequential_fields = out_fields.len();
    out_meta.fields = out_fields;

    let meta_str = serde_json::to_string(&out_meta).unwrap();
    let mut meta_file = File::create_new(args.output_basename.clone() + ".tbc.json")
        .expect("Can't create metadata file");
    meta_file
        .write_all(meta_str.as_bytes())
        .expect("Can't write to metadata file");
}
