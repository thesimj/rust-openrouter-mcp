//! DTOs for the synchronous audio endpoints: `POST /api/v1/audio/speech`
//! (text-to-speech) and `POST /api/v1/audio/transcriptions` (speech-to-text).

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

/// Request body for `POST /api/v1/audio/transcriptions` in its JSON form.
/// (The endpoint also accepts OpenAI-style multipart; JSON keeps one code path.)
#[derive(Debug, Serialize)]
pub struct TranscriptionBody {
    pub model: String,
    pub input_audio: InputAudio,
    /// ISO-639-1 hint (e.g. "en", "ja"); improves accuracy when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// "json" (default) or "verbose_json".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// "segment"/"word"; only honored with response_format=verbose_json on an
    /// OpenAI-compatible provider (others reject it).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub timestamp_granularities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// Inline audio payload. `data` is **raw** base64 - a `data:` URL prefix is
/// rejected upstream - and `format` is required so the model can decode it.
#[derive(Debug, Serialize)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}
