//  This Source Code Form is subject to the terms of the Mozilla Public
//  License, v. 2.0. If a copy of the MPL was not distributed with this
//  file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Functional tests for the SIMD median routines: the sorting networks fully
//! sort, the `avg`/`sse` helpers match a scalar reference, and the end-to-end
//! `batch_n` matches a scalar median + sum-of-squared-errors reference. Every
//! check runs over each supported element type via the [`TestScalar`] harness.

use super::{avg, batch_n, sse, Net, Nets, Scalar, BLOCK_BYTES};

/// Tiny deterministic xorshift64 PRNG.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
}

/// Per-type test glue: how to draw random values, the reference squared error,
/// and how to compare accumulators (exact for integers, tolerant for the
/// non-associative float reduction).
trait TestScalar: Scalar + PartialOrd + core::fmt::Debug {
    /// A random value. `wide` draws from a broad range (full compare width,
    /// kept small enough that the squared-error sum still fits the accumulator);
    /// `!wide` draws from a tiny range to force many lane-wise ties.
    fn rand(rng: &mut Rng, wide: bool) -> Self;
    /// Reference squared error of `m` against `x`, computed independently of the
    /// kernel (in `i128`/`f64`) so it actually cross-checks `sse_step`.
    fn ref_sse(m: Self, x: Self) -> Self::Acc;
    /// Whether two accumulators agree: exact for `u64`, relative-tolerant for
    /// `f64` (block-wise float summation reorders the adds).
    fn acc_close(got: Self::Acc, want: Self::Acc) -> bool;
}

macro_rules! test_int {
    ($t:ty, $wide:expr) => {
        impl TestScalar for $t {
            fn rand(rng: &mut Rng, wide: bool) -> $t {
                let r = rng.next();
                if wide {
                    $wide(r)
                } else {
                    (r & 0xF) as $t
                }
            }
            fn ref_sse(m: $t, x: $t) -> u64 {
                let d = m as i128 - x as i128;
                (d * d) as u64
            }
            fn acc_close(got: u64, want: u64) -> bool {
                got == want
            }
        }
    };
}

// Unsigned/small types take their full width; the 32-bit types are clamped to
// ~20 bits so the per-input squared-error sum stays inside `u64`.
test_int!(u8, |r| r as u8);
test_int!(i8, |r| r as i8);
test_int!(u16, |r| r as u16);
test_int!(i16, |r| r as i16);
test_int!(u32, |r| r >> 12);
test_int!(i32, |r| (r as i32) >> 12);

macro_rules! test_float {
    ($t:ty) => {
        impl TestScalar for $t {
            fn rand(rng: &mut Rng, wide: bool) -> $t {
                let r = rng.next();
                // Integer-valued floats: the network sorts them exactly and the
                // median's rounding average lands on representable values, so the
                // output can be compared for exact equality.
                if wide {
                    (((r >> 12) as i32) - (1 << 19)) as $t
                } else {
                    (r & 0xF) as $t
                }
            }
            fn ref_sse(m: $t, x: $t) -> f64 {
                let d = (m - x) as f64;
                d * d
            }
            fn acc_close(got: f64, want: f64) -> bool {
                (got - want).abs() <= 1e-9 * got.abs().max(want.abs()).max(1.0)
            }
        }
    };
}

test_float!(f32);
test_float!(f64);

/// A random block of `L` lanes.
fn rand_v<T: TestScalar, const L: usize>(rng: &mut Rng, wide: bool) -> [T; L] {
    std::array::from_fn(|_| T::rand(rng, wide))
}

/// Asserts that, for every lane, the values across the given vectors are in
/// non-decreasing order.
fn check_sorted_lanes<T: PartialOrd + Copy + core::fmt::Debug, const L: usize>(vs: &[[T; L]]) {
    for lane in 0..L {
        for k in 1..vs.len() {
            assert!(
                vs[k - 1][lane] <= vs[k][lane],
                "lane {lane}: {:?} > {:?} at position {k}",
                vs[k - 1][lane],
                vs[k][lane]
            );
        }
    }
}

/// Runs the `N`-input sorting network over random blocks and checks it fully
/// sorts every lane.
fn check_sort<T: TestScalar, const N: usize, const L: usize>(seed: u64)
where
    Nets: Net<N>,
{
    let mut rng = Rng::new(seed);
    for &wide in &[true, false] {
        for _ in 0..2000 {
            let mut arr: [[T; L]; N] = std::array::from_fn(|_| rand_v::<T, L>(&mut rng, wide));
            <Nets as Net<N>>::sort::<T, L>(&mut arr);
            check_sorted_lanes(&arr);
        }
    }
}

/// Exercises every supported stream count `3..=15` of the sorting network for
/// one element type.
fn check_all_sorts<T: TestScalar, const L: usize>() {
    check_sort::<T, 3, L>(0x9E3779B97F4A7C15);
    check_sort::<T, 4, L>(0x9E3779B97F4A7C16);
    check_sort::<T, 5, L>(0x9E3779B97F4A7C17);
    check_sort::<T, 6, L>(0x9E3779B97F4A7C18);
    check_sort::<T, 7, L>(0x9E3779B97F4A7C19);
    check_sort::<T, 8, L>(0x9E3779B97F4A7C1A);
    check_sort::<T, 9, L>(0x9E3779B97F4A7C1B);
    check_sort::<T, 10, L>(0x9E3779B97F4A7C1C);
    check_sort::<T, 11, L>(0x9E3779B97F4A7C1D);
    check_sort::<T, 12, L>(0x9E3779B97F4A7C1E);
    check_sort::<T, 13, L>(0x9E3779B97F4A7C1F);
    check_sort::<T, 14, L>(0x9E3779B97F4A7C20);
    check_sort::<T, 15, L>(0x9E3779B97F4A7C21);
}

/// Scalar reference median of a column, matching the SIMD semantics (rounding
/// average of the two middle elements for even counts) by reusing
/// [`Scalar::avg`].
fn reference_median<T: TestScalar>(col: &mut [T]) -> T {
    col.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = col.len();
    if n % 2 == 1 {
        col[n / 2]
    } else {
        T::avg(col[n / 2 - 1], col[n / 2])
    }
}

/// End-to-end check of `batch_n` against a scalar median + SSE reference for one
/// element type, across all stream counts.
fn check_batch<T: TestScalar, const L: usize>(seed: u64) {
    let mut rng = Rng::new(seed);
    let len = L * 7; // must be a multiple of L
    for n in 3..=15usize {
        for &wide in &[true, false] {
            let inputs: Vec<Vec<T>> = (0..n)
                .map(|_| (0..len).map(|_| T::rand(&mut rng, wide)).collect())
                .collect();
            let slices: Vec<&[T]> = inputs.iter().map(|v| v.as_slice()).collect();

            let mut out = inputs[0].clone(); // reused as scratch of the right type/len
            let mut sse_acc = vec![T::Acc::default(); n];
            batch_n(&mut out, &slices, &mut sse_acc);

            // Reference median per sample.
            for i in 0..len {
                let mut col: Vec<T> = (0..n).map(|k| inputs[k][i]).collect();
                let expected = reference_median(&mut col);
                assert!(
                    out[i] == expected,
                    "median mismatch n={n} wide={wide} sample={i}: got {:?} want {:?}",
                    out[i],
                    expected
                );
            }

            // Reference sum of squared errors per input.
            let mut ref_sse = vec![T::Acc::default(); n];
            for (k, input) in inputs.iter().enumerate() {
                for i in 0..len {
                    ref_sse[k] += T::ref_sse(out[i], input[i]);
                }
            }
            for k in 0..n {
                assert!(
                    T::acc_close(sse_acc[k], ref_sse[k]),
                    "sse mismatch n={n} wide={wide} input={k}: got {:?} want {:?}",
                    sse_acc[k],
                    ref_sse[k]
                );
            }
        }
    }
}

macro_rules! type_suite {
    ($mod:ident, $t:ty, $lanes:literal) => {
        mod $mod {
            use super::*;

            #[test]
            fn lane_count() {
                assert_eq!(<$t as Scalar>::LANES, $lanes);
                assert_eq!(
                    <$t as Scalar>::LANES,
                    BLOCK_BYTES / core::mem::size_of::<$t>()
                );
            }

            #[test]
            fn sorts() {
                check_all_sorts::<$t, $lanes>();
            }

            #[test]
            fn batch_matches_reference() {
                check_batch::<$t, $lanes>(0xC0FFEE123);
            }
        }
    };
}

type_suite!(t_u8, u8, 64);
type_suite!(t_i8, i8, 64);
type_suite!(t_u16, u16, 32);
type_suite!(t_i16, i16, 32);
type_suite!(t_u32, u32, 16);
type_suite!(t_i32, i32, 16);
type_suite!(t_f32, f32, 16);
type_suite!(t_f64, f64, 8);

#[test]
fn avg_rounds_without_overflow() {
    // Rounding half-up with no overflow at the top.
    let cases = [
        (0u16, 0u16),
        (1, 2),
        (2, 3),
        (65535, 65535),
        (65534, 1),
        (65535, 0),
        (100, 201),
        (40000, 40001),
    ];
    for &(x, y) in &cases {
        let got = avg::<u16, 32>([x; 32], [y; 32])[0];
        let exp = ((x as u32 + y as u32 + 1) >> 1) as u16;
        assert_eq!(got, exp, "avg({x}, {y})");
    }
}

#[test]
fn sse_matches_scalar() {
    let mut rng = Rng::new(0xDEADBEEF);
    for _ in 0..2000 {
        let a = rand_v::<u16, 32>(&mut rng, true);
        let b = rand_v::<u16, 32>(&mut rng, true);
        let got = sse::<u16, 32>(a, b);
        let mut exp = 0u64;
        for i in 0..32 {
            let d = a[i] as i64 - b[i] as i64;
            exp += (d * d) as u64;
        }
        assert_eq!(got, exp);
    }
}

/// Throughput benchmark for `batch_n`. Run with:
///   cargo test --release -- --ignored --nocapture bench_batch_n
/// Uses cache-resident buffers so it measures compute throughput (the best
/// case for wider vectors). Build with `-C target-cpu=x86-64-v2/v3/v4` to
/// compare microarchitecture levels.
#[test]
#[ignore]
fn bench_batch_n() {
    bench_type::<u8>("u8");
    bench_type::<i8>("i8");
    bench_type::<u16>("u16");
    bench_type::<i16>("i16");
    bench_type::<u32>("u32");
    bench_type::<i32>("i32");
    bench_type::<f32>("f32");
    bench_type::<f64>("f64");
}

#[cfg(test)]
fn bench_type<T: TestScalar>(name: &str) {
    use std::hint::black_box;
    use std::time::Instant;

    const BYTES: usize = 16 * 1024; // per input, stays in L1/L2
    let len = BYTES / core::mem::size_of::<T>();
    const ITERS: u64 = 100_000;

    println!(
        "\n[{name}]  LANES = {}, LEN = {len}, ITERS = {ITERS}",
        T::LANES
    );
    for &n in &[3usize, 5, 8, 15] {
        let mut rng = Rng::new(0x1234 + n as u64);
        let inputs: Vec<Vec<T>> = (0..n)
            .map(|_| (0..len).map(|_| T::rand(&mut rng, true)).collect())
            .collect();
        let slices: Vec<&[T]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut out = inputs[0].clone();
        let mut sse_acc = vec![T::Acc::default(); n];

        for _ in 0..2000 {
            batch_n(
                black_box(out.as_mut_slice()),
                black_box(slices.as_slice()),
                &mut sse_acc,
            );
        }

        // Best of 5 runs to suppress scheduling/turbo noise.
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                batch_n(
                    black_box(out.as_mut_slice()),
                    black_box(slices.as_slice()),
                    &mut sse_acc,
                );
            }
            let secs = t0.elapsed().as_secs_f64();
            black_box(&out);
            black_box(&sse_acc);
            best = best.min(secs);
        }

        let total = len as f64 * ITERS as f64;
        println!(
            "  n={n:2}  {:.3} ns/elem  {:7.0} Melem/s",
            best / total * 1e9,
            total / best / 1e6,
        );
    }
}
