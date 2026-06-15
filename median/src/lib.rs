//  This Source Code Form is subject to the terms of the Mozilla Public
//  License, v. 2.0. If a copy of the MPL was not distributed with this
//  file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! SIMD median filter that auto-vectorizes.
//!
//! [`batch_n`] takes `N` equal-length input streams and computes, for each
//! sample position, the median across the `N` streams, plus each input's sum of
//! squared errors against that median. The median is the middle value for odd
//! `N`, or the rounding average of the two middle values for even `N`.
//!
//! Work proceeds in fixed [`BLOCK_BYTES`]-byte blocks (`L = BLOCK_BYTES /
//! size_of::<T>()` lanes per block), each lowering to native packed
//! instructions.
//!
//! The element type is abstracted by the [`Scalar`] trait. Supported types are
//! the signed and unsigned integers up to 32 bits and both IEEE floats
//! (`u8`/`i8`/`u16`/`i16`/`u32`/`i32`/`f32`/`f64`); 64-bit integers are not
//! supported.

use core::ops::AddAssign;

/// Block size in bytes the kernels work in (64). Input slice lengths must be
/// multiples of `BLOCK_BYTES / size_of::<T>()` lanes (e.g. multiples of 32 for
/// `u16`).
pub const BLOCK_BYTES: usize = 64;

/// Element types the median kernels support:
/// `u8`/`i8`/`u16`/`i16`/`u32`/`i32`/`f32`/`f64`.
///
/// Each method operates on a single lane.
pub trait Scalar: Copy {
    /// Accumulator for the sum of squared errors: `u64` for integers, `f64` for
    /// floats.
    type Acc: Copy + Default + AddAssign + core::fmt::Debug;

    /// Lanes per block for this type: `BLOCK_BYTES / size_of::<Self>()`.
    const LANES: usize;

    /// Lane-wise minimum.
    fn vmin(a: Self, b: Self) -> Self;
    /// Lane-wise maximum.
    fn vmax(a: Self, b: Self) -> Self;
    /// Rounding average, used for the even-`N` median. Integers compute
    /// `(a + b + 1) >> 1` without overflow; floats compute `(a + b) * 0.5`.
    fn avg(a: Self, b: Self) -> Self;
    /// Accumulate the squared error of median `m` against original `x` into the
    /// per-input accumulator.
    fn sse_step(acc: &mut Self::Acc, m: Self, x: Self);

    /// Run the median kernel for this type: write each sample's median across
    /// the `N` inputs to `out` and each input's sum of squared errors to `sse_`.
    fn batch<const N: usize>(out: &mut [Self], sse_: &mut [Self::Acc; N], a: &[&[Self]; N])
    where
        Nets: Net<N>;
}

/// Compare-exchange: leaves the lane-wise minimum in `a` and maximum in `b`.
#[inline]
fn sort2<T: Scalar, const L: usize>(a: &mut [T; L], b: &mut [T; L]) {
    for i in 0..L {
        let x = a[i];
        let y = b[i];
        a[i] = T::vmin(x, y);
        b[i] = T::vmax(x, y);
    }
}

/// Rounding average of two vectors, lane-wise.
#[inline]
fn avg<T: Scalar, const L: usize>(a: [T; L], b: [T; L]) -> [T; L] {
    let mut out = a;
    for i in 0..L {
        out[i] = T::avg(a[i], b[i]);
    }
    out
}

/// Sum of squared errors between two vectors, lane-wise.
#[inline(never)]
fn sse<T: Scalar, const L: usize>(m: [T; L], x: [T; L]) -> T::Acc {
    let mut sum = T::Acc::default();
    for i in 0..L {
        T::sse_step(&mut sum, m[i], x[i]);
    }
    sum
}

/// The median kernel for a fixed stream count `N`, generic over the element
/// type `T` and lane count `L`.
pub trait Net<const N: usize> {
    /// Writes each sample's median across the `N` inputs to `out` and
    /// accumulates each input's sum of squared errors against the median into
    /// `sse_`.
    fn run<T: Scalar, const L: usize>(out: &mut [T], sse_: &mut [T::Acc; N], a: &[&[T]; N]);

    /// Fully sorts `N` vectors lane-wise with the same network `run` uses.
    /// Test-only.
    #[cfg(test)]
    fn sort<T: Scalar, const L: usize>(a: &mut [[T; L]; N]);
}

/// Zero-sized type carrying the per-`N` [`Net`] implementations.
pub struct Nets;

/// Runs the median kernel for element type `T`, lane count `L` and stream
/// count `N`.
#[inline(never)]
fn batch_median<T: Scalar, const L: usize, const N: usize>(
    out: &mut [T],
    sse_: &mut [T::Acc; N],
    a: &[&[T]; N],
) where
    Nets: Net<N>,
{
    <Nets as Net<N>>::run::<T, L>(out, sse_, a);
}

/// Implements [`Scalar`] for an integer type. `$wide` is the wider type the
/// rounding average computes in. The squared error is accumulated by `$sse`:
/// `sse_narrow` for ≤ 16-bit types, `sse_wide` for 32-bit types.
macro_rules! impl_int_scalar {
    ($t:ty, $wide:ty, $sse:ident) => {
        impl Scalar for $t {
            type Acc = u64;
            const LANES: usize = BLOCK_BYTES / core::mem::size_of::<$t>();

            #[inline]
            fn vmin(a: Self, b: Self) -> Self {
                a.min(b)
            }
            #[inline]
            fn vmax(a: Self, b: Self) -> Self {
                a.max(b)
            }
            #[inline]
            fn avg(a: Self, b: Self) -> Self {
                ((a as $wide + b as $wide + 1) >> 1) as $t
            }
            #[inline]
            fn sse_step(acc: &mut u64, m: Self, x: Self) {
                impl_int_scalar!(@$sse acc, m, x);
            }
            #[inline]
            fn batch<const N: usize>(out: &mut [Self], sse_: &mut [u64; N], a: &[&[Self]; N])
            where
                Nets: Net<N>,
            {
                batch_median::<Self, { BLOCK_BYTES / core::mem::size_of::<$t>() }, N>(out, sse_, a)
            }
        }
    };
    (@sse_narrow $acc:ident, $m:ident, $x:ident) => {
        let diff = $m as i32 - $x as i32;
        *$acc += (diff as i64 * diff as i64) as u64;
    };
    (@sse_wide $acc:ident, $m:ident, $x:ident) => {
        let d = (($m as i64) - ($x as i64)).unsigned_abs();
        *$acc += d * d;
    };
}

/// Implements [`Scalar`] for a float type: `vmin`/`vmax` via compare-select,
/// `avg` as the plain midpoint, and the squared error accumulated in `f64`.
macro_rules! impl_float_scalar {
    ($t:ty) => {
        impl Scalar for $t {
            type Acc = f64;
            const LANES: usize = BLOCK_BYTES / core::mem::size_of::<$t>();

            #[inline]
            fn vmin(a: Self, b: Self) -> Self {
                if a < b {
                    a
                } else {
                    b
                }
            }
            #[inline]
            fn vmax(a: Self, b: Self) -> Self {
                if a < b {
                    b
                } else {
                    a
                }
            }
            #[inline]
            fn avg(a: Self, b: Self) -> Self {
                (a + b) * 0.5
            }
            #[inline]
            fn sse_step(acc: &mut f64, m: Self, x: Self) {
                let d = (m - x) as f64;
                *acc += d * d;
            }
            #[inline]
            fn batch<const N: usize>(out: &mut [Self], sse_: &mut [f64; N], a: &[&[Self]; N])
            where
                Nets: Net<N>,
            {
                batch_median::<Self, { BLOCK_BYTES / core::mem::size_of::<$t>() }, N>(out, sse_, a)
            }
        }
    };
}

// The supported element types.
impl_int_scalar!(u8, u16, sse_narrow);
impl_int_scalar!(i8, i16, sse_narrow);
impl_int_scalar!(u16, u32, sse_narrow);
impl_int_scalar!(i16, i32, sse_narrow);
impl_int_scalar!(u32, u64, sse_wide);
impl_int_scalar!(i32, i64, sse_wide);
impl_float_scalar!(f32);
impl_float_scalar!(f64);

/// Generates a `Net` implementation per entry plus the runtime `batch_n`
/// dispatcher. Each entry is
///
/// ```text
/// N => ([lane indices 0..N-1], [median slot(s)], [compare-exchange network])
/// ```
///
/// The median is the middle sorted lane for odd `N`, or the rounding average of
/// the two middle lanes for even `N`.
macro_rules! medians {
    (
        $(
            $n:literal => (
                [ $($lane:literal),+ $(,)? ],
                [ $mid0:literal $(, $midr:literal)* ],
                [ $( ($x:literal, $y:literal) ),+ $(,)? ]
            )
        ),+ $(,)?
    ) => {
        $(
            impl Net<$n> for Nets {
                #[inline]
                fn run<T: Scalar, const L: usize>(
                    out: &mut [T],
                    sse_: &mut [T::Acc; $n],
                    a: &[&[T]; $n],
                ) {
                    ::paste::paste! {
                        // Bind each input slice to a local.
                        $( let [<a $lane>] = a[$lane]; )+
                        let len = out.len();
                        assert_eq!(len % L, 0);
                        $( assert_eq!(len, [<a $lane>].len()); )+
                        sse_.fill(T::Acc::default());
                        for (i, outc) in out.chunks_exact_mut(L).enumerate() {
                            let base = i * L;
                            // Originals, kept for the squared-error accumulation.
                            $( let [<va $lane>]: [T; L] = [<a $lane>][base..base + L].try_into().unwrap(); )+
                            // Working copies the network sorts in place.
                            $( let mut [<s $lane>] = [<va $lane>]; )+
                            $( sort2(&mut [<s $x>], &mut [<s $y>]); )+
                            // Median: middle local (odd) or rounding avg of the
                            // two middle locals (even).
                            let m = [<s $mid0>];
                            $( let m = avg(m, [<s $midr>]); )*
                            $( sse_[$lane] += sse(m, [<va $lane>]); )+
                            outc.copy_from_slice(&m);
                        }
                    }
                }

                #[cfg(test)]
                fn sort<T: Scalar, const L: usize>(a: &mut [[T; L]; $n]) {
                    $(
                        let (mut lo, mut hi) = (a[$x], a[$y]);
                        sort2(&mut lo, &mut hi);
                        a[$x] = lo;
                        a[$y] = hi;
                    )+
                }
            }
        )+

        /// Computes the per-sample median across the input streams `a`, writing
        /// each median to `out` and each input's sum of squared errors against
        /// the median to `sse_`. All slices must have the same length, a
        /// multiple of `T::LANES`; `sse_` has one entry per input. Panics if the
        /// number of inputs is unsupported.
        pub fn batch_n<T: Scalar>(out: &mut [T], a: &[&[T]], sse_: &mut [T::Acc]) {
            match a.len() {
                $(
                    $n => T::batch::<$n>(
                        out,
                        sse_.try_into().unwrap(),
                        a.try_into().unwrap(),
                    ),
                )+
                _ => panic!(),
            }
        }
    };
}

// Sorting networks as compare-exchange index pairs, one entry per supported
// stream count.
medians! {
    3  => ([0, 1, 2], [1], [(0, 2), (0, 1), (1, 2)]),
    4  => ([0, 1, 2, 3], [1, 2], [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]),
    5  => ([0, 1, 2, 3, 4], [2], [(0, 3), (1, 4), (0, 2), (1, 3), (0, 1), (2, 4), (1, 2), (3, 4), (2, 3)]),
    6  => ([0, 1, 2, 3, 4, 5], [2, 3], [(0, 5), (1, 3), (2, 4), (1, 2), (3, 4), (0, 3), (2, 5), (0, 1), (2, 3), (4, 5), (1, 2), (3, 4)]),
    7  => ([0, 1, 2, 3, 4, 5, 6], [3], [(0, 6), (2, 3), (4, 5), (0, 2), (1, 4), (3, 6), (0, 1), (2, 5), (3, 4), (1, 2), (4, 6), (2, 3), (4, 5), (1, 2), (3, 4), (5, 6)]),
    8  => ([0, 1, 2, 3, 4, 5, 6, 7], [3, 4], [(0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7), (0, 1), (2, 3), (4, 5), (6, 7), (2, 4), (3, 5), (1, 4), (3, 6), (1, 2), (3, 4), (5, 6)]),
    9  => ([0, 1, 2, 3, 4, 5, 6, 7, 8], [4], [(0, 3), (1, 7), (2, 5), (4, 8), (0, 7), (2, 4), (3, 8), (5, 6), (0, 2), (1, 3), (4, 5), (7, 8), (1, 4), (3, 6), (5, 7), (0, 1), (2, 4), (3, 5), (6, 8), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6)]),
    10 => ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5], [(0, 1), (2, 5), (3, 6), (4, 7), (8, 9), (0, 6), (1, 8), (2, 4), (3, 9), (5, 7), (0, 2), (1, 3), (4, 5), (6, 8), (7, 9), (0, 1), (2, 7), (3, 5), (4, 6), (8, 9), (1, 2), (3, 4), (5, 6), (7, 8), (1, 3), (2, 4), (5, 7), (6, 8), (2, 3), (4, 5), (6, 7)]),
    11 => ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5], [(0, 9), (1, 6), (2, 4), (3, 7), (5, 8), (0, 1), (3, 5), (4, 10), (6, 9), (7, 8), (1, 3), (2, 5), (4, 7), (8, 10), (0, 4), (1, 2), (3, 7), (5, 9), (6, 8), (0, 1), (2, 6), (4, 5), (7, 8), (9, 10), (2, 4), (3, 6), (5, 7), (8, 9), (1, 2), (3, 4), (5, 6), (7, 8), (2, 3), (4, 5), (6, 7)]),
    12 => ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [5, 6], [(0, 8), (1, 7), (2, 6), (3, 11), (4, 10), (5, 9), (0, 2), (1, 4), (3, 5), (6, 8), (7, 10), (9, 11), (0, 1), (2, 9), (4, 7), (5, 6), (10, 11), (1, 3), (2, 7), (4, 9), (8, 10), (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (1, 2), (3, 5), (6, 8), (9, 10), (2, 4), (3, 6), (5, 8), (7, 9), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]),
    13 => ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [6], [(0, 11), (1, 7), (2, 4), (3, 5), (8, 9), (10, 12), (0, 2), (3, 6), (4, 12), (5, 7), (8, 10), (0, 8), (1, 3), (2, 5), (4, 9), (6, 11), (7, 12), (0, 1), (2, 10), (3, 8), (4, 6), (9, 11), (1, 3), (2, 4), (5, 10), (6, 8), (7, 9), (11, 12), (1, 2), (3, 4), (5, 8), (6, 9), (7, 10), (2, 3), (4, 7), (5, 6), (8, 11), (9, 10), (4, 5), (6, 7), (8, 9), (10, 11), (3, 4), (5, 6), (7, 8), (9, 10)]),
    14 => ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [6, 7], [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (0, 2), (1, 3), (4, 8), (5, 9), (10, 12), (11, 13), (0, 10), (1, 6), (2, 11), (3, 13), (5, 8), (7, 12), (1, 4), (2, 8), (3, 6), (5, 11), (7, 10), (9, 12), (0, 1), (3, 9), (4, 10), (5, 7), (6, 8), (12, 13), (1, 5), (2, 4), (3, 7), (6, 10), (8, 12), (9, 11), (1, 2), (3, 5), (4, 6), (7, 9), (8, 10), (11, 12), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (3, 4), (5, 6), (7, 8), (9, 10)]),
    15 => ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [7], [(0, 6), (1, 10), (2, 14), (3, 9), (4, 12), (5, 13), (7, 11), (0, 7), (2, 5), (3, 4), (6, 11), (8, 10), (9, 12), (13, 14), (1, 13), (2, 3), (4, 6), (5, 9), (7, 8), (10, 14), (11, 12), (0, 3), (1, 4), (5, 7), (6, 13), (8, 9), (10, 11), (12, 14), (0, 2), (1, 5), (3, 8), (4, 6), (7, 10), (9, 11), (12, 13), (0, 1), (2, 5), (3, 10), (4, 8), (6, 7), (9, 12), (11, 13), (1, 2), (3, 4), (5, 6), (7, 9), (8, 10), (11, 12), (3, 5), (4, 6), (7, 8), (9, 10), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]),
}

#[cfg(test)]
mod tests;
