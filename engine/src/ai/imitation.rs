use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImitationRecord {
    pub pid: usize,
    pub era: usize,
    pub value: [f32; 4],
    pub abs_vp: [f32; 4],
    pub winner: [f32; 4],
    pub econ: [f32; 2],
    pub snapshot: Vec<u8>,
    pub teacher_canonical: String,
}
