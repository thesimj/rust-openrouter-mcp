//! DTOs for `POST /api/v1/audio/speech` (synchronous text-to-speech).

use serde::Serialize;

/// Request body for `POST /api/v1/audio/speech`. `response_format`/`speed` are
/// omitted when unset.
#[derive(Debug, Serialize)]
pub struct SpeechBody {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
}

/// Raw audio bytes from `/audio/speech`, constructed from the response (not
/// deserialized): the MIME type, bytes, and optional generation id.
pub struct SpeechResult {
    pub mime: String,
    pub bytes: Vec<u8>,
    pub generation_id: Option<String>,
}
