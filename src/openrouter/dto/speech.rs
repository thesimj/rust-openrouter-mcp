//! DTOs for the synchronous audio endpoints: `POST /api/v1/audio/speech`
//! (text-to-speech) and `POST /api/v1/audio/transcriptions` (speech-to-text).

use serde::Serialize;

use super::provider::ProviderOptions;

/// Request body for `POST /api/v1/audio/speech`. Every optional field is
/// omitted when unset: `voice` is provider-dependent (voice-cloning models
/// take none), `input_references` carries the stateless cloning sample, and
/// `provider` the per-provider passthrough.
#[derive(Debug, Serialize)]
pub struct SpeechBody {
    pub model: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// Stateless voice cloning: one `input_audio` sample, optionally followed
    /// by a `text` transcript of it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub input_references: Vec<SpeechInputReference>,
    /// Per-provider passthrough (`options.<slug>`); routing is ignored here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderOptions>,
}

/// One `input_references[]` part of the speech request, tagged by `type`:
/// `{"type":"input_audio","input_audio":{...}}` or `{"type":"text","text":...}`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SpeechInputReference {
    InputAudio { input_audio: SpeechReferenceAudio },
    Text { text: String },
}

/// The audio sample of a voice reference: raw base64 `data` plus the container
/// `format` when known (omitted otherwise - unlike transcription, the speech
/// endpoint does not require it).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeechReferenceAudio {
    pub data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
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
    /// Per-provider passthrough (`options.<slug>`); routing is ignored here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderOptions>,
}

/// Inline audio payload. `data` is **raw** base64 - a `data:` URL prefix is
/// rejected upstream - and `format` is required so the model can decode it.
/// Also the body of a chat `input_audio` content part.
#[derive(Debug, Clone, Serialize)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body(provider: Option<ProviderOptions>) -> TranscriptionBody {
        TranscriptionBody {
            model: "openai/whisper-1".into(),
            input_audio: InputAudio {
                data: "QUJD".into(),
                format: "mp3".into(),
            },
            language: None,
            response_format: None,
            timestamp_granularities: vec![],
            temperature: None,
            provider,
        }
    }

    fn speech_body(
        voice: Option<&str>,
        input_references: Vec<SpeechInputReference>,
        provider: Option<ProviderOptions>,
    ) -> SpeechBody {
        SpeechBody {
            model: "openai/gpt-4o-mini-tts".into(),
            input: "hello".into(),
            voice: voice.map(str::to_string),
            response_format: Some("mp3".into()),
            speed: None,
            input_references,
            provider,
        }
    }

    /// Serde lock for the speech body: `voice`, `input_references` and
    /// `provider` are all omitted when unset (no `null`, no `[]`), and the
    /// references serialize as the documented tagged parts - the audio part
    /// drops `format` when it is unknown.
    #[test]
    fn speech_body_omits_unset_optionals_and_shapes_input_references() {
        let bare = serde_json::to_value(speech_body(None, vec![], None)).unwrap();
        assert!(bare.get("voice").is_none(), "sent: {bare}");
        assert!(bare.get("input_references").is_none(), "sent: {bare}");
        assert!(bare.get("provider").is_none(), "sent: {bare}");

        let mut options = std::collections::BTreeMap::new();
        options.insert("openai".to_string(), json!({"instructions": "cheerful"}));
        let full = serde_json::to_value(speech_body(
            Some("alloy"),
            vec![
                SpeechInputReference::InputAudio {
                    input_audio: SpeechReferenceAudio {
                        data: "QUJD".into(),
                        format: Some("mp3".into()),
                    },
                },
                SpeechInputReference::Text {
                    text: "the words spoken in the sample".into(),
                },
            ],
            Some(ProviderOptions { options }),
        ))
        .unwrap();
        assert_eq!(full["voice"], "alloy");
        assert_eq!(
            full["input_references"],
            json!([
                {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "mp3"}},
                {"type": "text", "text": "the words spoken in the sample"}
            ])
        );
        assert_eq!(
            full["provider"],
            json!({"options": {"openai": {"instructions": "cheerful"}}})
        );

        let no_format = serde_json::to_value(SpeechInputReference::InputAudio {
            input_audio: SpeechReferenceAudio {
                data: "QUJD".into(),
                format: None,
            },
        })
        .unwrap();
        assert_eq!(
            no_format,
            json!({"type": "input_audio", "input_audio": {"data": "QUJD"}})
        );
    }

    /// Serde lock: `provider` is omitted entirely when unset (a bare
    /// `"provider": null` or `{}` is not what the endpoint documents), and
    /// serializes as the options-only block when set.
    #[test]
    fn transcription_body_omits_provider_when_none_and_nests_options_when_set() {
        let none = serde_json::to_value(body(None)).unwrap();
        assert!(none.get("provider").is_none(), "sent: {none}");

        let mut options = std::collections::BTreeMap::new();
        options.insert("deepgram".to_string(), json!({"diarize": true}));
        let some = serde_json::to_value(body(Some(ProviderOptions { options }))).unwrap();
        assert_eq!(
            some["provider"],
            json!({"options": {"deepgram": {"diarize": true}}})
        );
    }
}
