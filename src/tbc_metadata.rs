//  This Source Code Form is subject to the terms of the Mozilla Public
//  License, v. 2.0. If a copy of the MPL was not distributed with this
//  file, You can obtain one at http://mozilla.org/MPL/2.0/.

use serde_derive::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum System {
    #[serde(rename = "PAL")]
    Pal,
    #[serde(rename = "NTSC")]
    Ntsc,
    #[serde(rename = "PAL-M")]
    PalM,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct VideoParameters {
    #[serde(rename = "numberOfSequentialFields")]
    pub number_of_sequential_fields: usize,

    #[serde(rename = "system")]
    pub system: System,

    #[serde(rename = "fieldWidth")]
    pub field_width: usize,

    #[serde(rename = "fieldHeight")]
    pub field_height: usize,

    #[serde(flatten)]
    pub other: HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct VitsMetrics {
    #[serde(rename = "bPSNR")]
    pub bpsnr: f64,

    #[serde(flatten)]
    pub other: HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct DropOuts {
    #[serde(rename = "fieldLine")]
    pub field_line: Vec<usize>,

    #[serde(rename = "startx")]
    pub startx: Vec<usize>,

    #[serde(rename = "endx")]
    pub endx: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct Field {
    #[serde(rename = "isFirstField")]
    pub is_first_field: bool,

    #[serde(rename = "seqNo")]
    pub seq_no: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "vitsMetrics")]
    pub vits_metrics: Option<VitsMetrics>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "dropOuts")]
    pub drop_outs: Option<DropOuts>, // Optional field

    #[serde(flatten)]
    pub other: HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct TbcMetadata {
    #[serde(rename = "videoParameters")]
    pub video_parameters: VideoParameters,

    #[serde(rename = "fields")]
    pub fields: Vec<Field>,

    #[serde(flatten)]
    pub other: HashMap<String, serde_json::Value>,
}
