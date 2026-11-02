//! Pure views over a `/audio/transcriptions` response body: the compact
//! summary `transcribe_audio` returns in place of a saved response, and the
//! speaker turns its `normalize` option adds.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// How many characters of the transcript text the summary previews.
const PREVIEW_CHARS: usize = 300;

/// A pause longer than this between two words starts a new turn.
const TURN_GAP_SECONDS: f64 = 4.0;

/// The `words[].type` of a non-speech tag such as "(laughter)".
const AUDIO_EVENT: &str = "audio_event";

/// The compact summary of a response body. A field is present only when its
/// source is in the body.
pub(crate) fn summarize(body: &Value) -> Map<String, Value> {
    let mut summary = Map::new();
    let usage = body.get("usage");
    insert_some(&mut summary, "cost", present(usage, "cost").cloned());
    insert_some(
        &mut summary,
        "usage_seconds",
        present(usage, "seconds").cloned(),
    );
    insert_some(
        &mut summary,
        "language",
        present(Some(body), "language").cloned(),
    );
    insert_some(
        &mut summary,
        "duration",
        present(Some(body), "duration").cloned(),
    );
    if let Some(text) = body.get("text").and_then(Value::as_str) {
        let preview = text
            .char_indices()
            .nth(PREVIEW_CHARS)
            .map_or(text, |(cut, _)| &text[..cut]);
        summary.insert("text_chars".into(), text.chars().count().into());
        summary.insert("preview".into(), preview.into());
    }
    if let Some(segments) = body.get("segments").and_then(Value::as_array) {
        summary.insert("segments".into(), segments.len().into());
    }
    let words: Vec<&Value> = spoken_words(body).collect();
    if body.get("words").is_some_and(Value::is_array) {
        summary.insert("words".into(), words.len().into());
        insert_word_timing(&mut summary, &words);
    }
    if let Some(totals) = speaker_totals(body, &words) {
        let mut speakers: Vec<&Value> = totals.values().map(|(speaker, _)| *speaker).collect();
        speakers.sort_by(|a, b| speaker_order(a, b));
        let seconds: Map<String, Value> = totals
            .iter()
            .map(|(key, (_, total))| (key.clone(), round_to_ms(*total).into()))
            .collect();
        summary.insert(
            "speakers".into(),
            speakers.into_iter().cloned().collect::<Vec<_>>().into(),
        );
        summary.insert("speaker_seconds".into(), seconds.into());
    }
    summary
}

/// Speaker turns built from the body's words, or an empty list when the body
/// has no words. A turn ends when the speaker changes or after a pause longer
/// than [`TURN_GAP_SECONDS`].
pub(crate) fn turns(body: &Value) -> Vec<Value> {
    let mut turns: Vec<Turn> = Vec::new();
    let mut previous_end = None;
    for word in spoken_words(body) {
        let text = word
            .get("word")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        let speaker = speaker(word);
        let start = seconds(word, "start");
        let end = seconds(word, "end");
        let paused = matches!(
            (previous_end, start),
            (Some(previous), Some(start)) if start - previous > TURN_GAP_SECONDS
        );
        match turns.last_mut() {
            Some(turn) if turn.speaker == speaker && !paused => turn.extend(text, start, end),
            _ => turns.push(Turn {
                speaker,
                start,
                end,
                text: text.to_string(),
            }),
        }
        previous_end = end;
    }
    turns.into_iter().map(Turn::into_json).collect()
}

/// One speaker's uninterrupted run of words.
struct Turn<'a> {
    speaker: Option<&'a Value>,
    start: Option<f64>,
    end: Option<f64>,
    text: String,
}

impl Turn<'_> {
    /// Providers send each word without its surrounding spaces, so words are
    /// joined with one space.
    fn extend(&mut self, text: &str, start: Option<f64>, end: Option<f64>) {
        if !text.is_empty() {
            if !self.text.is_empty() {
                self.text.push(' ');
            }
            self.text.push_str(text);
        }
        self.start = self.start.or(start);
        self.end = end.or(self.end);
    }

    fn into_json(self) -> Value {
        let mut turn = Map::new();
        insert_some(&mut turn, "speaker", self.speaker.cloned());
        insert_some(&mut turn, "start", self.start.map(Value::from));
        insert_some(&mut turn, "end", self.end.map(Value::from));
        turn.insert("text".into(), self.text.into());
        turn.into()
    }
}

/// The body's `words`, without audio-event tags.
fn spoken_words(body: &Value) -> impl Iterator<Item = &Value> {
    body.get("words")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|word| word.get("type").and_then(Value::as_str) != Some(AUDIO_EVENT))
}

/// `first_word_s`, `last_word_s` and `max_gap_s`, from the words that carry
/// those times.
fn insert_word_timing(summary: &mut Map<String, Value>, words: &[&Value]) {
    let first_start = words
        .iter()
        .filter_map(|w| seconds(w, "start"))
        .reduce(f64::min);
    let last_end = words
        .iter()
        .filter_map(|w| seconds(w, "end"))
        .reduce(f64::max);
    let max_gap = words
        .windows(2)
        .filter_map(|pair| Some(seconds(pair[1], "start")? - seconds(pair[0], "end")?))
        .reduce(f64::max);
    insert_some(summary, "first_word_s", first_start.map(Value::from));
    insert_some(summary, "last_word_s", last_end.map(Value::from));
    insert_some(
        summary,
        "max_gap_s",
        max_gap.map(|gap| round_to_ms(gap).into()),
    );
}

/// Seconds spoken per speaker, keyed by the speaker's string form, read from
/// the words when any word names a speaker, else from the segments; `None`
/// when neither does (no diarization).
fn speaker_totals<'a>(
    body: &'a Value,
    words: &[&'a Value],
) -> Option<BTreeMap<String, (&'a Value, f64)>> {
    let segments: Vec<&Value> = body
        .get("segments")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .collect();
    let diarized = |items: &[&Value]| items.iter().any(|item| speaker(item).is_some());
    let source = if diarized(words) {
        words
    } else if diarized(&segments) {
        &segments
    } else {
        return None;
    };
    let mut totals = BTreeMap::new();
    for item in source {
        let Some(speaker) = speaker(item) else {
            continue;
        };
        let spoken = match (seconds(item, "start"), seconds(item, "end")) {
            (Some(start), Some(end)) => end - start,
            _ => 0.0,
        };
        totals
            .entry(speaker_key(speaker))
            .or_insert((speaker, 0.0))
            .1 += spoken;
    }
    Some(totals)
}

/// The speaker of a word or segment: its `speaker` index, else its
/// `speaker_label`.
fn speaker(item: &Value) -> Option<&Value> {
    ["speaker", "speaker_label"]
        .into_iter()
        .find_map(|key| present(Some(item), key))
}

/// A speaker id as an object key: a string as it is, a number as written.
fn speaker_key(speaker: &Value) -> String {
    match speaker {
        Value::String(label) => label.clone(),
        other => other.to_string(),
    }
}

/// Numbers in numeric order, then labels in text order.
fn speaker_order(a: &Value, b: &Value) -> Ordering {
    match (a.as_f64(), b.as_f64()) {
        (Some(a), Some(b)) => a.total_cmp(&b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => speaker_key(a).cmp(&speaker_key(b)),
    }
}

/// `item[key]` when it is present and not null.
fn present<'a>(item: Option<&'a Value>, key: &str) -> Option<&'a Value> {
    item?.get(key).filter(|value| !value.is_null())
}

fn seconds(item: &Value, key: &str) -> Option<f64> {
    item.get(key).and_then(Value::as_f64)
}

/// Sums and differences of float seconds pick up noise such as
/// 0.30000000000000004; a millisecond is finer than any word timestamp.
fn round_to_ms(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}

fn insert_some(map: &mut Map<String, Value>, key: &str, value: Option<Value>) {
    if let Some(value) = value {
        map.insert(key.into(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn word(text: &str, start: f64, end: f64, speaker: Value) -> Value {
        json!({"word": text, "start": start, "end": end, "speaker": speaker, "type": "word"})
    }

    #[test]
    fn summary_reads_usage_text_and_word_timing() {
        let body = json!({
            "text": "hello there friend",
            "language": "en",
            "duration": 12.5,
            "segments": [{"id": 0, "text": "hello there friend"}],
            "words": [
                word("hello", 0.5, 1.0, json!(0)),
                word("there", 1.25, 2.0, json!(1)),
                word("friend", 9.5, 10.0, json!(0)),
            ],
            "usage": {"cost": 0.0717, "seconds": 12.5}
        });
        let summary = summarize(&body);
        assert_eq!(summary["cost"], 0.0717);
        assert_eq!(summary["usage_seconds"], 12.5);
        assert_eq!(summary["language"], "en");
        assert_eq!(summary["duration"], 12.5);
        assert_eq!(summary["text_chars"], 18);
        assert_eq!(summary["preview"], "hello there friend");
        assert_eq!(summary["segments"], 1);
        assert_eq!(summary["words"], 3);
        assert_eq!(summary["speakers"], json!([0, 1]));
        assert_eq!(summary["speaker_seconds"], json!({"0": 1.0, "1": 0.75}));
        assert_eq!(summary["first_word_s"], 0.5);
        assert_eq!(summary["last_word_s"], 10.0);
        assert_eq!(summary["max_gap_s"], 7.5);
    }

    #[test]
    fn summary_without_words_keeps_only_the_fields_it_has() {
        let body = json!({"text": "hi", "usage": {"cost": 0.001}});
        let summary = summarize(&body);
        assert_eq!(
            Value::Object(summary),
            json!({"cost": 0.001, "text_chars": 2, "preview": "hi"})
        );
    }

    #[test]
    fn summary_keys_string_speaker_labels_by_their_text() {
        let label = |text, start, end, label: &str| json!({"word": text, "start": start, "end": end, "speaker_label": label});
        let body = json!({
            "text": "a b c",
            "words": [
                label("a", 0.0, 1.0, "speaker_1"),
                label("b", 1.0, 1.5, "speaker_0"),
                label("c", 1.5, 3.0, "speaker_1"),
            ]
        });
        let summary = summarize(&body);
        assert_eq!(summary["speakers"], json!(["speaker_0", "speaker_1"]));
        assert_eq!(
            summary["speaker_seconds"],
            json!({"speaker_0": 0.5, "speaker_1": 2.5})
        );
    }

    #[test]
    fn summary_prefers_the_speaker_index_over_its_label() {
        let body = json!({
            "text": "a",
            "words": [{"word": "a", "start": 0.0, "end": 1.0, "speaker": 3, "speaker_label": "speaker_3"}]
        });
        assert_eq!(summarize(&body)["speakers"], json!([3]));
    }

    #[test]
    fn summary_sorts_numeric_speakers_numerically() {
        let body = json!({
            "text": "a b c",
            "words": [
                word("a", 0.0, 1.0, json!(10)),
                word("b", 1.0, 2.0, json!(2)),
                word("c", 2.0, 3.0, json!(10)),
            ]
        });
        assert_eq!(summarize(&body)["speakers"], json!([2, 10]));
    }

    #[test]
    fn summary_reads_speakers_from_segments_when_words_have_none() {
        let body = json!({
            "text": "a b",
            "segments": [
                {"start": 0.0, "end": 4.0, "text": "a", "speaker": 1},
                {"start": 4.0, "end": 5.0, "text": "b", "speaker": 0}
            ],
            "words": [{"word": "a", "start": 0.0, "end": 1.0}]
        });
        let summary = summarize(&body);
        assert_eq!(summary["speakers"], json!([0, 1]));
        assert_eq!(summary["speaker_seconds"], json!({"0": 1.0, "1": 4.0}));
    }

    #[test]
    fn summary_omits_speakers_without_diarization() {
        let body = json!({
            "text": "a",
            "segments": [{"start": 0.0, "end": 1.0, "text": "a"}],
            "words": [{"word": "a", "start": 0.0, "end": 1.0}]
        });
        let summary = summarize(&body);
        assert!(!summary.contains_key("speakers"), "{summary:?}");
        assert!(!summary.contains_key("speaker_seconds"), "{summary:?}");
    }

    #[test]
    fn summary_counts_chars_and_cuts_a_multibyte_preview_on_a_char_boundary() {
        let text = "Привіт ".repeat(60);
        let body = json!({ "text": text });
        let summary = summarize(&body);
        assert_eq!(summary["text_chars"], 420);
        let preview = summary["preview"].as_str().unwrap();
        assert_eq!(preview.chars().count(), 300);
        assert!(text.starts_with(preview));
    }

    #[test]
    fn summary_and_turns_skip_audio_events() {
        let body = json!({
            "text": "hi (laughter) there",
            "words": [
                word("hi", 0.0, 1.0, json!(0)),
                {"word": "(laughter)", "start": 1.0, "end": 30.0, "speaker": 7, "type": "audio_event"},
                word("there", 1.5, 2.0, json!(0)),
            ]
        });
        let summary = summarize(&body);
        assert_eq!(summary["words"], 2);
        assert_eq!(summary["speakers"], json!([0]));
        assert_eq!(summary["speaker_seconds"], json!({"0": 1.5}));
        assert_eq!(summary["last_word_s"], 2.0);
        assert_eq!(summary["max_gap_s"], 0.5);
        assert_eq!(
            turns(&body),
            vec![json!({"speaker": 0, "start": 0.0, "end": 2.0, "text": "hi there"})]
        );
    }

    #[test]
    fn turns_start_when_the_speaker_changes() {
        let body = json!({
            "words": [
                word("Чи", 0.0, 0.4, json!(0)),
                word("були", 0.5, 1.0, json!(0)),
                word("так", 1.1, 1.5, json!(1)),
                word("ні", 1.6, 2.0, json!(0)),
            ]
        });
        assert_eq!(
            turns(&body),
            vec![
                json!({"speaker": 0, "start": 0.0, "end": 1.0, "text": "Чи були"}),
                json!({"speaker": 1, "start": 1.1, "end": 1.5, "text": "так"}),
                json!({"speaker": 0, "start": 1.6, "end": 2.0, "text": "ні"}),
            ]
        );
    }

    #[test]
    fn turns_start_after_a_pause_longer_than_four_seconds() {
        let body = json!({
            "words": [
                word("one", 0.0, 1.0, json!(0)),
                word("two", 5.0, 5.5, json!(0)),
                word("three", 10.0, 10.5, json!(0)),
            ]
        });
        // 1.0 -> 5.0 is exactly 4 s and stays in the turn; 5.5 -> 10.0 is longer.
        assert_eq!(
            turns(&body),
            vec![
                json!({"speaker": 0, "start": 0.0, "end": 5.5, "text": "one two"}),
                json!({"speaker": 0, "start": 10.0, "end": 10.5, "text": "three"}),
            ]
        );
    }

    #[test]
    fn turns_trim_word_spacing_and_omit_a_missing_speaker() {
        let body = json!({
            "words": [
                {"word": " hello", "start": 0.0, "end": 0.5},
                {"word": "world ", "start": 0.6, "end": 1.0},
            ]
        });
        assert_eq!(
            turns(&body),
            vec![json!({"start": 0.0, "end": 1.0, "text": "hello world"})]
        );
    }

    #[test]
    fn turns_are_empty_without_words() {
        assert!(turns(&json!({"text": "hi"})).is_empty());
        assert!(turns(&json!({"text": "hi", "words": []})).is_empty());
    }
}
