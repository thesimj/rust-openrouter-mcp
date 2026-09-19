//! Structured decisions for the MCP tool: `POST /api/alpha/decisions`.
//!
//! Like [`crate::embed_gen`], this is the single path from validated inputs to
//! the result JSON: local checks, the request body, the receipt handling and
//! the result envelope all live here once.

use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::collections::BTreeMap;

use crate::openrouter::{
    DecisionAnswer, DecisionQuestion, DecisionsBody, OpenRouterClient, ProviderRouting,
};

/// Inputs for one `/api/alpha/decisions` request.
#[derive(Debug, Clone)]
pub struct DecideRequest {
    pub model: String,
    /// The content to evaluate: a non-blank string, a non-empty object or a
    /// non-empty array.
    pub state: Value,
    /// Already-typed questions keyed by the caller's names.
    pub questions: BTreeMap<String, DecisionQuestion>,
    /// Routing block, already validated (this endpoint has no `options`).
    pub provider: Option<ProviderRouting>,
}

/// The typed answers plus usage and the receipt id.
#[derive(Debug)]
pub struct DecideResult {
    pub model: String,
    pub id: Option<String>,
    pub provider: Option<String>,
    pub answers: BTreeMap<String, DecisionAnswer>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cost: Option<f64>,
    pub generation_id: Option<String>,
}

impl DecideResult {
    /// The result envelope returned by the MCP tool. Answers keep the wire
    /// shape (`type` plus `noul` / `choice` / `score` and their optionals).
    pub fn to_json(&self) -> Value {
        json!({
            "model": self.model,
            "id": self.id,
            "provider": self.provider,
            "answers": self.answers,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost": self.cost,
            },
            "generation_id": self.generation_id,
        })
    }
}

impl DecideRequest {
    /// The local checks run before any HTTP call. They enforce what the
    /// endpoint would reject anyway, plus the provider's own documented
    /// minimums (a score needs at least two levels), with errors that name
    /// the question.
    pub fn validate(&self) -> Result<()> {
        check_state(&self.state)?;
        if self.questions.is_empty() {
            bail!("questions must contain at least one question");
        }
        for (name, question) in &self.questions {
            if name.trim().is_empty() {
                bail!("every question needs a non-blank name (key)");
            }
            check_question(name, question)?;
        }
        Ok(())
    }
}

/// `state` is guidance that must also carry something to evaluate: an empty
/// object or array has nothing to judge.
fn check_state(state: &Value) -> Result<()> {
    check_guidance("state", state)?;
    match state {
        Value::Object(m) if m.is_empty() => bail!("state must not be an empty object"),
        Value::Array(a) if a.is_empty() => bail!("state must not be an empty array"),
        _ => Ok(()),
    }
}

/// Guidance may be a string, an object or an array - but a string must say
/// something (whitespace-only counts as absent, the repo-wide rule). `what`
/// names the field in the error.
fn check_guidance(what: &str, guidance: &Value) -> Result<()> {
    match guidance {
        Value::String(s) if s.trim().is_empty() => bail!("{what} must not be blank"),
        Value::String(_) | Value::Object(_) | Value::Array(_) => Ok(()),
        other => bail!("{what} must be a string, an object or an array, got {other}"),
    }
}

fn check_question(name: &str, question: &DecisionQuestion) -> Result<()> {
    let field = |f: &str| format!("questions[{name:?}].{f}");
    match question {
        DecisionQuestion::Noul {
            instructions,
            criteria,
        } => {
            check_guidance(&field("instructions"), instructions)?;
            if let Some(c) = criteria {
                check_guidance(&field("criteria.true"), &c.when_true)?;
                check_guidance(&field("criteria.false"), &c.when_false)?;
            }
        }
        DecisionQuestion::Choice {
            instructions,
            criteria,
        } => {
            check_guidance(&field("instructions"), instructions)?;
            if criteria.is_empty() {
                bail!(
                    "{} must map at least one label to its description",
                    field("criteria")
                );
            }
            for (label, guidance) in criteria {
                if label.trim().is_empty() {
                    bail!("{} has a blank label", field("criteria"));
                }
                // `null` guidance is allowed: the label speaks for itself.
                if !guidance.is_null() {
                    check_guidance(&field(&format!("criteria[{label:?}]")), guidance)?;
                }
            }
        }
        DecisionQuestion::Score {
            instructions,
            criteria,
        } => {
            check_guidance(&field("instructions"), instructions)?;
            if criteria.len() < 2 {
                bail!(
                    "{} must list at least two levels, lowest first",
                    field("criteria")
                );
            }
            for (i, guidance) in criteria.iter().enumerate() {
                check_guidance(&field(&format!("criteria[{i}]")), guidance)?;
            }
        }
    }
    Ok(())
}

/// Ask a decisions model. Validates locally, sends the documented body, and
/// flattens the reply. A 2xx with no answers is an error that still carries the
/// receipt (the provider may have billed).
pub async fn decide(client: &OpenRouterClient, req: &DecideRequest) -> Result<DecideResult> {
    req.validate()?;
    let body = DecisionsBody {
        model: req.model.clone(),
        state: req.state.clone(),
        questions: req.questions.clone(),
        provider: req.provider.clone(),
    };
    let reply = client.decisions(&body).await?;
    let usage = reply.body.usage.unwrap_or_default();
    let receipt = crate::billing::Receipt {
        cost: usage.cost,
        generation_id: reply.generation_id.clone(),
    };
    if reply.body.answers.is_empty() {
        return Err(receipt.attach(anyhow::anyhow!("model returned no answers")));
    }
    Ok(DecideResult {
        model: reply.body.model.unwrap_or_else(|| req.model.clone()),
        id: reply.body.id,
        provider: reply.body.provider,
        answers: reply.body.answers,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cost: usage.cost,
        generation_id: reply.generation_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter::{NoulCriteria, OpenRouterClient};
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn noul(instructions: &str) -> DecisionQuestion {
        DecisionQuestion::Noul {
            instructions: json!(instructions),
            criteria: None,
        }
    }

    fn choice(labels: &[(&str, Value)]) -> DecisionQuestion {
        DecisionQuestion::Choice {
            instructions: json!("Which one?"),
            criteria: labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        }
    }

    fn score(levels: &[&str]) -> DecisionQuestion {
        DecisionQuestion::Score {
            instructions: json!("How much?"),
            criteria: levels.iter().map(|l| json!(l)).collect(),
        }
    }

    fn request(questions: &[(&str, DecisionQuestion)]) -> DecideRequest {
        DecideRequest {
            model: "typesafe/jev-1.13".into(),
            state: json!({"ticket": "checkout is blank"}),
            questions: questions
                .iter()
                .map(|(k, q)| (k.to_string(), q.clone()))
                .collect(),
            provider: None,
        }
    }

    fn error_of(req: &DecideRequest) -> String {
        req.validate().unwrap_err().to_string()
    }

    /// `state` must be a non-blank string or a non-empty object/array.
    #[test]
    fn validate_rejects_empty_or_scalar_state() {
        let ok = request(&[("q", noul("Is it a bug?"))]);
        ok.validate().unwrap();
        for (state, needle) in [
            (json!("   "), "blank"),
            (json!({}), "empty object"),
            (json!([]), "empty array"),
            (json!(42), "string, an object or an array"),
            (json!(true), "string, an object or an array"),
            (Value::Null, "string, an object or an array"),
        ] {
            let mut req = ok.clone();
            req.state = state.clone();
            let err = error_of(&req);
            assert!(err.contains("state"), "{state}: {err}");
            assert!(err.contains(needle), "{state}: {err}");
        }
        let mut text = ok.clone();
        text.state = json!("plain text is fine");
        text.validate().unwrap();
        let mut list = ok;
        list.state = json!(["a", {"b": 1}]);
        list.validate().unwrap();
    }

    /// At least one question, every name non-blank, every instruction non-blank.
    #[test]
    fn validate_requires_named_questions_with_instructions() {
        assert!(error_of(&request(&[])).contains("at least one question"));
        assert!(error_of(&request(&[(" ", noul("x"))])).contains("non-blank name"));
        let err = error_of(&request(&[("q", noul("  "))]));
        assert!(err.contains("questions[\"q\"].instructions"), "{err}");
        assert!(err.contains("blank"), "{err}");
        // Structured instructions are guidance too.
        let mut structured = request(&[("q", noul("x"))]);
        structured.questions.insert(
            "q".into(),
            DecisionQuestion::Noul {
                instructions: json!({"what": "Is it spam?"}),
                criteria: Some(NoulCriteria {
                    when_true: json!("yes"),
                    when_false: json!("no"),
                }),
            },
        );
        structured.validate().unwrap();
        let mut blank_criteria = structured.clone();
        blank_criteria.questions.insert(
            "q".into(),
            DecisionQuestion::Noul {
                instructions: json!("x"),
                criteria: Some(NoulCriteria {
                    when_true: json!("yes"),
                    when_false: json!(""),
                }),
            },
        );
        let err = error_of(&blank_criteria);
        assert!(err.contains("criteria.false"), "{err}");
    }

    /// A choice needs at least one non-blank label; `null` guidance is fine.
    #[test]
    fn validate_checks_choice_labels() {
        assert!(error_of(&request(&[("q", choice(&[]))])).contains("at least one label"));
        assert!(error_of(&request(&[("q", choice(&[(" ", json!("a"))]))])).contains("blank label"));
        let err = error_of(&request(&[("q", choice(&[("a", json!(""))]))]));
        assert!(err.contains("criteria[\"a\"]"), "{err}");
        request(&[("q", choice(&[("a", Value::Null), ("b", json!("B"))]))])
            .validate()
            .unwrap();
    }

    /// A score needs two or more levels (the provider's documented minimum),
    /// each of them non-blank.
    #[test]
    fn validate_requires_two_score_levels() {
        let err = error_of(&request(&[("q", score(&["only"]))]));
        assert!(err.contains("at least two levels"), "{err}");
        let err = error_of(&request(&[("q", score(&["low", " "]))]));
        assert!(err.contains("criteria[1]"), "{err}");
        request(&[("q", score(&["low", "high"]))])
            .validate()
            .unwrap();
    }

    /// The body reaches the alpha path, answers flatten into the envelope with
    /// the usage counters and the generation id.
    #[tokio::test]
    async fn decide_sends_the_body_and_flattens_the_reply() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/alpha/decisions"))
            .and(body_partial_json(json!({
                "model": "typesafe/jev-1.13",
                "state": {"ticket": "checkout is blank"},
                "questions": {
                    "is_bug": {"type": "noul", "instructions": "Is it a bug?"},
                    "urgency": {"type": "score", "instructions": "How much?", "criteria": ["low", "high"]}
                }
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-d")
                    .set_body_json(json!({
                        "id": "gen-dec-1",
                        "model": "typesafe/jev-1.13-20260917",
                        "provider": "TypeSafe",
                        "answers": {
                            "is_bug": {"type": "noul", "noul": 0.96},
                            "urgency": {"type": "score", "score": 0.9, "confidence": 0.9,
                                        "legend": {"0": "low", "1": "high"},
                                        "probabilities": {"0": 0.1, "1": 0.9}}
                        },
                        "usage": {"input_tokens": 40, "output_tokens": 5, "cost": 0.0000017}
                    })),
            )
            .mount(&mock)
            .await;

        let client = OpenRouterClient::with_base_url(mock.uri(), "test-key");
        let req = request(&[
            ("is_bug", noul("Is it a bug?")),
            ("urgency", score(&["low", "high"])),
        ]);
        let result = decide(&client, &req).await.unwrap();
        assert_eq!(result.model, "typesafe/jev-1.13-20260917");
        assert_eq!(result.generation_id.as_deref(), Some("gen-d"));
        assert_eq!(result.cost, Some(0.0000017));

        let v = result.to_json();
        assert_eq!(v["id"], "gen-dec-1");
        assert_eq!(v["provider"], "TypeSafe");
        assert_eq!(
            v["answers"]["is_bug"],
            json!({"type": "noul", "noul": 0.96})
        );
        assert_eq!(v["answers"]["urgency"]["score"], 0.9);
        assert_eq!(v["answers"]["urgency"]["legend"]["1"], "high");
        assert_eq!(
            v["usage"],
            json!({"input_tokens": 40, "output_tokens": 5, "cost": 0.0000017})
        );
        assert_eq!(v["generation_id"], "gen-d");
    }

    /// Validation runs before any HTTP call; a 2xx with no answers is an error
    /// that still carries the receipt.
    #[tokio::test]
    async fn decide_rejects_bad_input_locally_and_empty_answers_with_receipt() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/alpha/decisions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-empty")
                    .set_body_json(json!({"model": "m", "answers": {}, "usage": {"cost": 0.0}})),
            )
            .mount(&mock)
            .await;
        let client = OpenRouterClient::with_base_url(mock.uri(), "test-key");

        let err = decide(&client, &request(&[])).await.unwrap_err();
        assert!(err.to_string().contains("questions"), "got: {err}");
        assert_eq!(mock.received_requests().await.unwrap().len(), 0);

        let err = decide(&client, &request(&[("q", noul("x"))]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no answers"), "got: {err}");
        let receipt = crate::billing::Receipt::from_error(&err).expect("receipt kept");
        assert_eq!(receipt.generation_id.as_deref(), Some("gen-empty"));
        assert_eq!(receipt.cost, Some(0.0));
    }
}
