//! Billing survives local decoding and file-delivery failures.

use serde::Serialize;

/// An upstream generation response was received; its cost may be unknown.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct Receipt {
    pub cost: Option<f64>,
    pub generation_id: Option<String>,
}

impl std::fmt::Display for Receipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the provider response was received and may be billed")?;
        if let Some(id) = &self.generation_id {
            write!(f, " (generation {id})")?;
        }
        Ok(())
    }
}

impl std::error::Error for Receipt {}

impl Receipt {
    /// Attach this receipt to a post-response failure so the real cause stays
    /// the headline message and the receipt is the root cause, where
    /// [`Receipt::from_error`] can still find it. (`anyhow::Context` would make
    /// the receipt the outermost message instead, prefixing every user-facing
    /// error with billing text.)
    pub fn attach(self, error: anyhow::Error) -> anyhow::Error {
        anyhow::Error::new(self).context(format!("{error:#}"))
    }

    /// Run a fallible post-response step, attaching this receipt to its error.
    pub fn wrap<T>(self, f: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
        f().map_err(|e| self.attach(e))
    }

    /// The receipt carried anywhere in an error chain, if any.
    pub fn from_error(error: &anyhow::Error) -> Option<&Receipt> {
        error.downcast_ref::<Receipt>()
    }
}

/// Aggregate request costs once, independently of the number of saved files.
#[derive(Debug, Default)]
pub(crate) struct Totals {
    pub cost: f64,
    pub unknown: u64,
}

impl Totals {
    pub fn add(&mut self, receipt: &Receipt) {
        match receipt.cost {
            Some(cost) => self.cost += cost,
            None => self.unknown += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attached_receipt_keeps_the_cause_as_headline_and_stays_recoverable() {
        let receipt = Receipt {
            cost: Some(0.5),
            generation_id: Some("gen-1".into()),
        };
        let error = receipt.wrap(|| -> anyhow::Result<()> {
            Err(anyhow::anyhow!("inner").context("model returned no text"))
        });
        let error = error.unwrap_err();
        let rendered = format!("{error:#}");
        assert!(
            rendered.starts_with("model returned no text: inner"),
            "{rendered}"
        );
        assert!(rendered.contains("gen-1"), "{rendered}");
        assert_eq!(error.to_string(), "model returned no text: inner");
        assert_eq!(Receipt::from_error(&error).unwrap().cost, Some(0.5));
    }
}
