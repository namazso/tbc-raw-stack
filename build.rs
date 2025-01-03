//  This Source Code Form is subject to the terms of the Mozilla Public
//  License, v. 2.0. If a copy of the MPL was not distributed with this
//  file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::env;
use std::io::Write;
use std::path::Path;

struct SimdNames {
    feature: &'static str,
    type_name: &'static str,
    mm_prefix: &'static str,
    mm_loadu: &'static str,
    mm_storeu: &'static str,
}

fn write_simd<W: Write>(w: &mut W, simd: &SimdNames) {
    let sort_net: [&[(i32, i32)]; 16] = [
        &[(0i32, 0i32); 0][..],
        &[(0i32, 0i32); 0][..],
        &[(0i32, 0i32); 0][..],
        &[(0, 2), (0, 1), (1, 2)][..],
        &[(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)][..],
        &[
            (0, 3),
            (1, 4),
            (0, 2),
            (1, 3),
            (0, 1),
            (2, 4),
            (1, 2),
            (3, 4),
            (2, 3),
        ][..],
        &[
            (0, 5),
            (1, 3),
            (2, 4),
            (1, 2),
            (3, 4),
            (0, 3),
            (2, 5),
            (0, 1),
            (2, 3),
            (4, 5),
            (1, 2),
            (3, 4),
        ][..],
        &[
            (0, 6),
            (2, 3),
            (4, 5),
            (0, 2),
            (1, 4),
            (3, 6),
            (0, 1),
            (2, 5),
            (3, 4),
            (1, 2),
            (4, 6),
            (2, 3),
            (4, 5),
            (1, 2),
            (3, 4),
            (5, 6),
        ][..],
        &[
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (2, 4),
            (3, 5),
            (1, 4),
            (3, 6),
            (1, 2),
            (3, 4),
            (5, 6),
        ][..],
        &[
            (0, 3),
            (1, 7),
            (2, 5),
            (4, 8),
            (0, 7),
            (2, 4),
            (3, 8),
            (5, 6),
            (0, 2),
            (1, 3),
            (4, 5),
            (7, 8),
            (1, 4),
            (3, 6),
            (5, 7),
            (0, 1),
            (2, 4),
            (3, 5),
            (6, 8),
            (2, 3),
            (4, 5),
            (6, 7),
            (1, 2),
            (3, 4),
            (5, 6),
        ][..],
        &[
            (0, 1),
            (2, 5),
            (3, 6),
            (4, 7),
            (8, 9),
            (0, 6),
            (1, 8),
            (2, 4),
            (3, 9),
            (5, 7),
            (0, 2),
            (1, 3),
            (4, 5),
            (6, 8),
            (7, 9),
            (0, 1),
            (2, 7),
            (3, 5),
            (4, 6),
            (8, 9),
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (1, 3),
            (2, 4),
            (5, 7),
            (6, 8),
            (2, 3),
            (4, 5),
            (6, 7),
        ][..],
        &[
            (0, 9),
            (1, 6),
            (2, 4),
            (3, 7),
            (5, 8),
            (0, 1),
            (3, 5),
            (4, 10),
            (6, 9),
            (7, 8),
            (1, 3),
            (2, 5),
            (4, 7),
            (8, 10),
            (0, 4),
            (1, 2),
            (3, 7),
            (5, 9),
            (6, 8),
            (0, 1),
            (2, 6),
            (4, 5),
            (7, 8),
            (9, 10),
            (2, 4),
            (3, 6),
            (5, 7),
            (8, 9),
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (2, 3),
            (4, 5),
            (6, 7),
        ][..],
        &[
            (0, 8),
            (1, 7),
            (2, 6),
            (3, 11),
            (4, 10),
            (5, 9),
            (0, 2),
            (1, 4),
            (3, 5),
            (6, 8),
            (7, 10),
            (9, 11),
            (0, 1),
            (2, 9),
            (4, 7),
            (5, 6),
            (10, 11),
            (1, 3),
            (2, 7),
            (4, 9),
            (8, 10),
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (1, 2),
            (3, 5),
            (6, 8),
            (9, 10),
            (2, 4),
            (3, 6),
            (5, 8),
            (7, 9),
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
        ][..],
        &[
            (0, 11),
            (1, 7),
            (2, 4),
            (3, 5),
            (8, 9),
            (10, 12),
            (0, 2),
            (3, 6),
            (4, 12),
            (5, 7),
            (8, 10),
            (0, 8),
            (1, 3),
            (2, 5),
            (4, 9),
            (6, 11),
            (7, 12),
            (0, 1),
            (2, 10),
            (3, 8),
            (4, 6),
            (9, 11),
            (1, 3),
            (2, 4),
            (5, 10),
            (6, 8),
            (7, 9),
            (11, 12),
            (1, 2),
            (3, 4),
            (5, 8),
            (6, 9),
            (7, 10),
            (2, 3),
            (4, 7),
            (5, 6),
            (8, 11),
            (9, 10),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
        ][..],
        &[
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (12, 13),
            (0, 2),
            (1, 3),
            (4, 8),
            (5, 9),
            (10, 12),
            (11, 13),
            (0, 10),
            (1, 6),
            (2, 11),
            (3, 13),
            (5, 8),
            (7, 12),
            (1, 4),
            (2, 8),
            (3, 6),
            (5, 11),
            (7, 10),
            (9, 12),
            (0, 1),
            (3, 9),
            (4, 10),
            (5, 7),
            (6, 8),
            (12, 13),
            (1, 5),
            (2, 4),
            (3, 7),
            (6, 10),
            (8, 12),
            (9, 11),
            (1, 2),
            (3, 5),
            (4, 6),
            (7, 9),
            (8, 10),
            (11, 12),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
        ][..],
        &[
            (0, 6),
            (1, 10),
            (2, 14),
            (3, 9),
            (4, 12),
            (5, 13),
            (7, 11),
            (0, 7),
            (2, 5),
            (3, 4),
            (6, 11),
            (8, 10),
            (9, 12),
            (13, 14),
            (1, 13),
            (2, 3),
            (4, 6),
            (5, 9),
            (7, 8),
            (10, 14),
            (11, 12),
            (0, 3),
            (1, 4),
            (5, 7),
            (6, 13),
            (8, 9),
            (10, 11),
            (12, 14),
            (0, 2),
            (1, 5),
            (3, 8),
            (4, 6),
            (7, 10),
            (9, 11),
            (12, 13),
            (0, 1),
            (2, 5),
            (3, 10),
            (4, 8),
            (6, 7),
            (9, 12),
            (11, 13),
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 9),
            (8, 10),
            (11, 12),
            (3, 5),
            (4, 6),
            (7, 8),
            (9, 10),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
        ][..],
    ];

    let feature = simd.feature;
    let type_name = simd.type_name;
    let mm_prefix = simd.mm_prefix;
    let mm_loadu = simd.mm_loadu;
    let mm_storeu = simd.mm_storeu;

    // SSE - rustc is pretty good at vectorizing this
    w.write_all(
        format!(
            r#"
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "{feature}")]
#[inline]
unsafe fn sse(a: {type_name}, b: {type_name}) -> u64 {{
    let a: [u16; size_of::<{type_name}>() / 2] = std::mem::transmute(a);
    let b: [u16; size_of::<{type_name}>() / 2] = std::mem::transmute(b);
    let mut sum: u64 = 0;
    for i in 0..a.len() {{
        let diff = a[i] as i32 - b[i] as i32;
        let sq = diff as i64 * diff as i64;
        sum += sq as u64;
    }}
    sum
}}
"#
        )
        .as_bytes(),
    )
    .unwrap();

    // SORT2
    w.write_all(
        format!(
            r#"
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "{feature}")]
#[inline]
unsafe fn sort2(a0: &mut {type_name}, a1: &mut {type_name}) {{
    let min = {mm_prefix}_min_epu16(*a0, *a1);
    let max = {mm_prefix}_max_epu16(*a0, *a1);
    *a0 = min;
    *a1 = max;
}}
"#
        )
        .as_bytes(),
    )
    .unwrap();

    for i in 3..=15 {
        // SORTn
        w.write_all(
            format!(
                r#"
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "{feature}")]
#[inline]
unsafe fn sort{i}("#
            )
            .as_bytes(),
        )
        .unwrap();
        for j in 0..i {
            w.write_all(format!("a{j}: &mut {type_name},").as_bytes())
                .unwrap();
        }
        w.write_all(") {\n".as_bytes()).unwrap();
        let sort = sort_net[i]
            .iter()
            .map(|(a, b)| format!("sort2(a{a}, a{b});"))
            .collect::<Vec<_>>()
            .join("\n");
        w.write_all(sort.as_bytes()).unwrap();
        w.write_all("}\n".as_bytes()).unwrap();

        // MEDIANn
        w.write_all(
            format!(
                r#"
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "{feature}")]
#[inline]
unsafe fn median{i}("#
            )
            .as_bytes(),
        )
        .unwrap();
        for j in 0..i {
            w.write_all(format!("a{j}: {type_name},").as_bytes())
                .unwrap();
        }
        w.write_all(format!(") -> {type_name} {{\n").as_bytes())
            .unwrap();
        for j in 0..i {
            w.write_all(format!("let mut a{j} = a{j};\n").as_bytes())
                .unwrap();
        }
        w.write_all(format!("sort{i}(\n").as_bytes()).unwrap();
        for j in 0..i {
            w.write_all(format!("&mut a{j},\n").as_bytes()).unwrap();
        }
        w.write_all(");\n".as_bytes()).unwrap();
        if i % 2 == 1 {
            w.write_all(format!("a{}\n", i / 2).as_bytes()).unwrap();
        } else {
            w.write_all(format!("{mm_prefix}_avg_epu16(a{}, a{})\n", i / 2 - 1, i / 2).as_bytes())
                .unwrap();
        }
        w.write_all("}\n".as_bytes()).unwrap();

        // BATCH_MEDIANn
        w.write_all(
            format!(
                r#"
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "{feature}")]
#[inline(never)]
unsafe fn batch_median{i}(out: &mut [u16], sse_: &mut [u64; {i}], "#
            )
            .as_bytes(),
        )
        .unwrap();
        for j in 0..i {
            w.write_all(format!("a{j}: &[u16],").as_bytes()).unwrap();
        }
        w.write_all(
            format!(
                r#") {{
        let len = out.len();
        assert_eq!(len % 32, 0);
        sse_.fill(0u64);
        let pout: *mut {type_name} = std::mem::transmute(out.as_ptr());
        "#
            )
            .as_bytes(),
        )
        .unwrap();
        for j in 0..i {
            w.write_all(
                format!("let pa{j}: *const {type_name} = std::mem::transmute(a{j}.as_ptr());\n")
                    .as_bytes(),
            )
            .unwrap();
            w.write_all(format!("assert_eq!(len, a{j}.len());\n").as_bytes())
                .unwrap();
        }
        w.write_all(format!("for i in 0..len * 2 / size_of::<{type_name}>() {{\n").as_bytes())
            .unwrap();
        for j in 0..i {
            w.write_all(
                format!("let va{j} = {mm_loadu}(std::mem::transmute(pa{j}.add(i)));\n").as_bytes(),
            )
            .unwrap();
        }
        w.write_all(format!("let m = median{i}(\n").as_bytes())
            .unwrap();
        for j in 0..i {
            w.write_all(format!("va{j},\n").as_bytes()).unwrap();
        }
        w.write_all(");\n".as_bytes()).unwrap();
        for j in 0..i {
            w.write_all(format!("sse_[{j}] += sse(m, va{j});\n").as_bytes())
                .unwrap();
        }
        w.write_all(format!("{mm_storeu}(pout.add(i), m);\n").as_bytes())
            .unwrap();
        w.write_all("}\n}\n".as_bytes()).unwrap();
    }

    // BATCH_MEDIAN_N
    w.write_all(
        r#"
pub fn batch_median_n(out: &mut [u16], a: &[&[u16]], sse_: &mut [u64]) {{
    match a.len() {
"#
        .as_bytes(),
    )
    .unwrap();
    for i in 3..=15 {
        w.write_all(
            format!("{i} => unsafe {{ batch_median{i}(out, sse_.try_into().unwrap(),\n").as_bytes(),
        )
        .unwrap();
        for j in 0..i {
            w.write_all(format!("a[{j}],\n").as_bytes()).unwrap();
        }
        w.write_all(") },".as_bytes()).unwrap();
    }
    w.write_all(
        r#"
        _ => panic!(),
    }
}
}
"#
        .as_bytes(),
    )
    .unwrap();
}

fn main() {
    let i128 = SimdNames {
        feature: "sse4.1",
        type_name: "__m128i",
        mm_prefix: "_mm",
        mm_loadu: "_mm_loadu_si128",
        mm_storeu: "_mm_storeu_si128",
    };
    let i256 = SimdNames {
        feature: "avx2",
        type_name: "__m256i",
        mm_prefix: "_mm256",
        mm_loadu: "_mm256_loadu_si256",
        mm_storeu: "_mm256_storeu_si256",
    };
    let i512 = SimdNames {
        feature: "avx512bw",
        type_name: "__m512i",
        mm_prefix: "_mm512",
        mm_loadu: "_mm512_loadu_si512",
        mm_storeu: "_mm512_storeu_si512",
    };
    let out_dir = env::var_os("OUT_DIR").unwrap();
    write_simd(
        &mut std::fs::File::create(Path::new(&out_dir).join("simd_x86_128.rs")).unwrap(),
        &i128,
    );
    write_simd(
        &mut std::fs::File::create(Path::new(&out_dir).join("simd_x86_256.rs")).unwrap(),
        &i256,
    );
    write_simd(
        &mut std::fs::File::create(Path::new(&out_dir).join("simd_x86_512.rs")).unwrap(),
        &i512,
    );
    println!("cargo::rerun-if-changed=build.rs");
}
