//! Incremental parser for a `text/event-stream` body.
//!
//! OpenRouter streams chat completions as Server-Sent Events: `data: {json}`
//! lines separated by blank lines, interleaved with `: OPENROUTER PROCESSING`
//! comment lines while the provider works, and terminated by `data: [DONE]`.
//! The parser is fed raw byte chunks as they arrive (a chunk may end in the
//! middle of a line) and hands back the `data` payload of every completed event.

/// The sentinel payload that ends an OpenRouter stream.
pub(crate) const DONE: &str = "[DONE]";

#[derive(Default)]
pub(crate) struct SseParser {
    /// Bytes of the line currently being received (no terminator yet).
    line: Vec<u8>,
    /// `data` of the event currently being received; multi-line `data` fields
    /// are joined with `\n` per the spec.
    data: String,
    /// Whether the current event has seen a `data` line at all (an empty
    /// `data:` line still makes an event, with an empty payload).
    has_data: bool,
}

impl SseParser {
    /// Feed the next body chunk; returns the payloads of every event the chunk
    /// completed. Comment lines and non-`data` fields (`event`, `id`, `retry`)
    /// are ignored.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        let mut events = Vec::new();
        let mut rest = chunk;
        while let Some(newline) = rest.iter().position(|b| *b == b'\n') {
            self.line.extend_from_slice(&rest[..newline]);
            rest = &rest[newline + 1..];
            events.extend(self.end_line());
        }
        self.line.extend_from_slice(rest);
        events
    }

    /// Flush an event left unterminated when the body ended (no final blank
    /// line). A trailing partial `data` line is included.
    pub(crate) fn finish(&mut self) -> Option<String> {
        if !self.line.is_empty() {
            // A non-blank line never completes an event, so nothing is returned.
            self.end_line();
        }
        self.dispatch()
    }

    /// Consume the buffered line: a blank line completes the event, a `data`
    /// line extends it, anything else is skipped.
    fn end_line(&mut self) -> Option<String> {
        if self.line.last() == Some(&b'\r') {
            self.line.pop();
        }
        let event = if self.line.is_empty() {
            self.dispatch()
        } else {
            if let Some(value) = self.line.strip_prefix(b"data:") {
                let value = value.strip_prefix(b" ").unwrap_or(value);
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(&String::from_utf8_lossy(value));
                self.has_data = true;
            }
            None
        };
        self.line.clear();
        event
    }

    fn dispatch(&mut self) -> Option<String> {
        if !self.has_data {
            return None;
        }
        self.has_data = false;
        Some(std::mem::take(&mut self.data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_events_split_across_arbitrary_chunk_boundaries() {
        let body = b": OPENROUTER PROCESSING\n\ndata: {\"a\":1}\n\ndata: {\"b\":2}\r\n\r\ndata: [DONE]\n\n";
        for cut in 0..body.len() {
            let mut parser = SseParser::default();
            let mut events = parser.push(&body[..cut]);
            events.extend(parser.push(&body[cut..]));
            assert_eq!(
                events,
                vec!["{\"a\":1}", "{\"b\":2}", DONE],
                "chunk boundary at {cut}"
            );
            assert_eq!(parser.finish(), None);
        }
    }

    #[test]
    fn ignores_comments_and_other_fields_and_joins_multiline_data() {
        let mut parser = SseParser::default();
        let events = parser.push(b"event: chunk\nid: 7\n: keepalive\ndata: first\ndata:second\n\n");
        assert_eq!(events, vec!["first\nsecond"]);
    }

    #[test]
    fn an_empty_data_line_still_makes_an_event() {
        let mut parser = SseParser::default();
        assert_eq!(parser.push(b"data:\n\ndata: x\n\n"), vec!["", "x"]);
    }

    #[test]
    fn finish_flushes_a_stream_that_ends_without_a_blank_line() {
        let mut parser = SseParser::default();
        assert!(parser.push(b"data: tail").is_empty());
        assert_eq!(parser.finish().as_deref(), Some("tail"));
        assert_eq!(parser.finish(), None, "flushing twice yields nothing new");

        let mut parser = SseParser::default();
        assert!(parser.push(b"data: line\n").is_empty());
        assert_eq!(parser.finish().as_deref(), Some("line"));

        // A trailing comment line does not swallow the pending data.
        let mut parser = SseParser::default();
        assert!(parser.push(b"data: kept\n: trailing comment").is_empty());
        assert_eq!(parser.finish().as_deref(), Some("kept"));
    }
}
