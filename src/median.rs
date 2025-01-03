//  This Source Code Form is subject to the terms of the Mozilla Public
//  License, v. 2.0. If a copy of the MPL was not distributed with this
//  file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::{simd_x86_128, simd_x86_256, simd_x86_512};

pub fn batch_n(out: &mut [u16], a: &[&[u16]], sse: &mut [u64]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512bw") {
            return simd_x86_512::batch_median_n(out, a, sse);
        }
        if is_x86_feature_detected!("avx2") {
            return simd_x86_256::batch_median_n(out, a, sse);
        }
        if is_x86_feature_detected!("sse4.1") {
            return simd_x86_128::batch_median_n(out, a, sse);
        }
    }
    unimplemented!();
}
