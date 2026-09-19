//! DTOs for `POST /api/alpha/decisions` - the decisions endpoint that serves
//! TypeSafe's System One models (`typesafe/jev-1.13`, `~typesafe/jev-latest`).
//!
//! A decisions model does not generate text. It reads a `state` and answers a
//! map of named questions, each of one of three kinds: `noul` (a true/false
//! probability), `choice` (one label from a set) or `score` (a graded scale).
//! Guidance values (`instructions`, `criteria` entries) are a string, a JSON
//! object or an array - the OpenAPI spec calls the union "guidance", so they
//! travel as [`serde_json::Value`] here.
//!
//! `session_id`, `trace` and `user` are documented upstream but out of scope
//! (skipped on purpose).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::provider::ProviderRouting;

/// Request body for `POST /api/alpha/decisions`. Unset optionals are omitted.
#[derive(Debug, Serialize)]
pub struct DecisionsBody {
    pub model: String,
    /// The content to evaluate: a string, an object or an array.
    pub state: Value,
    /// Questions keyed by the caller's own names; answers come back under
    /// the same keys. The model never sees the keys.
    pub questions: BTreeMap<String, DecisionQuestion>,
    /// Routing-only block, the same subset chat, `/embeddings` and `/rerank` take.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderRouting>,
}

/// One question, discriminated by `type` on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionQuestion {
    /// A true/false question answered with a probability.
    Noul {
        instructions: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// Pick one label; `criteria` maps each label to its guidance (or null).
    Choice {
        instructions: Value,
        criteria: BTreeMap<String, Value>,
    },
    /// A graded scale; `criteria` lists the levels in order (index = score).
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
}

/// Optional `noul` criteria: what makes the answer true, and what makes it
/// false. OpenRouter requires both keys when the object is present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true")]
    pub when_true: Value,
    #[serde(rename = "false")]
    pub when_false: Value,
}

/// Response body: one typed answer per question name plus usage.
#[derive(Debug, Deserialize)]
pub struct DecisionsResponse {
    /// OpenRouter's request id (`gen-dec-...`); the header may carry it too.
    pub id: Option<String>,
    pub model: Option<String>,
    /// The serving provider's name (e.g. `TypeSafe`).
    pub provider: Option<String>,
    #[serde(default)]
    pub answers: BTreeMap<String, DecisionAnswer>,
    pub usage: Option<DecisionsUsage>,
}

/// One answer, discriminated by `type`. `confidence` (0..1) appears only on
/// `choice` and `score`; `probabilities` is keyed by label (choice) or by the
/// level index as a string (score).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionAnswer {
    Noul {
        /// Probability that the answer is true.
        noul: f64,
    },
    Choice {
        choice: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        probabilities: Option<BTreeMap<String, f64>>,
    },
    Score {
        /// Expected level, a float over the scale's index range.
        score: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
        /// The scale echoed back, keyed by level index (`"0"`, `"1"`, ...).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        legend: Option<BTreeMap<String, Value>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        probabilities: Option<BTreeMap<String, f64>>,
    },
}

#[derive(Debug, Default, Deserialize)]
pub struct DecisionsUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// USD charge for the request, when OpenRouter reports it inline.
    pub cost: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter::ProviderRouting;
    use serde_json::json;

    /// The request example from OpenRouter's OpenAPI spec, verbatim.
    pub(crate) fn spec_request() -> Value {
        json!({
            "model": "typesafe/jev-1.13",
            "questions": {
                "is_bug": {
                    "criteria": {
                        "false": "The customer is asking a question or requesting a feature.",
                        "true": "The customer describes broken or unexpected product behavior."
                    },
                    "instructions": "Is the customer reporting a software defect?",
                    "type": "noul"
                },
                "team": {
                    "criteria": {
                        "account": "Login, permissions, or profile issues.",
                        "frontend": "Rendering, layout, or browser compatibility issues.",
                        "payments": "Checkout, billing, or payment processing issues."
                    },
                    "instructions": "Which team should own this ticket?",
                    "type": "choice"
                },
                "urgency": {
                    "criteria": [
                        "Can wait for the next release",
                        "Should be fixed this week",
                        "Blocking revenue right now"
                    ],
                    "instructions": "How urgent is this ticket?",
                    "type": "score"
                }
            },
            "state": {
                "customer_tier": "enterprise",
                "ticket": "My checkout page shows a blank screen after I click Pay. I have tried two browsers."
            }
        })
    }

    /// The response example from OpenRouter's OpenAPI spec, verbatim.
    pub(crate) fn spec_response() -> Value {
        json!({
            "answers": {
                "is_bug": { "noul": 0.96, "type": "noul" },
                "team": {
                    "choice": "payments",
                    "confidence": 0.75,
                    "probabilities": { "account": 0, "frontend": 0.16, "payments": 0.84 },
                    "type": "choice"
                },
                "urgency": {
                    "confidence": 0.99,
                    "legend": {
                        "0": "Can wait for the next release",
                        "1": "Should be fixed this week",
                        "2": "Blocking revenue right now"
                    },
                    "probabilities": { "0": 0, "1": 0.01, "2": 0.99 },
                    "score": 1.99,
                    "type": "score"
                }
            },
            "id": "gen-dec-1789738314-X5e5eKGQdvR9rblyX250",
            "model": "typesafe/jev-1.13-20260917",
            "provider": "TypeSafe",
            "usage": { "cost": 0.000019992, "input_tokens": 476, "output_tokens": 70 }
        })
    }

    /// Serde lock: the three question kinds, built as Rust values, serialize
    /// to the spec's request example byte-for-byte as JSON values (noul
    /// criteria keys are `"true"`/`"false"`, score criteria stay ordered), and
    /// the same JSON deserializes back to equal values.
    #[test]
    fn request_matches_the_documented_wire_shape() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "is_bug".to_string(),
            DecisionQuestion::Noul {
                instructions: json!("Is the customer reporting a software defect?"),
                criteria: Some(NoulCriteria {
                    when_true: json!(
                        "The customer describes broken or unexpected product behavior."
                    ),
                    when_false: json!("The customer is asking a question or requesting a feature."),
                }),
            },
        );
        questions.insert(
            "team".to_string(),
            DecisionQuestion::Choice {
                instructions: json!("Which team should own this ticket?"),
                criteria: [
                    ("account", "Login, permissions, or profile issues."),
                    (
                        "frontend",
                        "Rendering, layout, or browser compatibility issues.",
                    ),
                    (
                        "payments",
                        "Checkout, billing, or payment processing issues.",
                    ),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_string(), json!(v)))
                .collect(),
            },
        );
        questions.insert(
            "urgency".to_string(),
            DecisionQuestion::Score {
                instructions: json!("How urgent is this ticket?"),
                criteria: vec![
                    json!("Can wait for the next release"),
                    json!("Should be fixed this week"),
                    json!("Blocking revenue right now"),
                ],
            },
        );
        let body = DecisionsBody {
            model: "typesafe/jev-1.13".into(),
            state: spec_request()["state"].clone(),
            questions: questions.clone(),
            provider: None,
        };
        assert_eq!(serde_json::to_value(&body).unwrap(), spec_request());

        let parsed: BTreeMap<String, DecisionQuestion> =
            serde_json::from_value(spec_request()["questions"].clone()).unwrap();
        assert_eq!(parsed, questions);

        let routed = DecisionsBody {
            provider: Some(ProviderRouting {
                order: vec!["typesafe".into()],
                ..Default::default()
            }),
            ..body
        };
        assert_eq!(
            serde_json::to_value(&routed).unwrap()["provider"],
            json!({"order": ["typesafe"]})
        );
    }

    /// A noul question without criteria omits the key; a choice label may
    /// carry `null` guidance; structured (object/array) guidance passes through.
    #[test]
    fn optional_and_structured_guidance_round_trip() {
        let q = DecisionQuestion::Noul {
            instructions: json!({"what": "Is it spam?", "examples": ["buy now"]}),
            criteria: None,
        };
        let v = serde_json::to_value(&q).unwrap();
        assert_eq!(
            v,
            json!({"type": "noul", "instructions": {"what": "Is it spam?", "examples": ["buy now"]}})
        );
        let c = DecisionQuestion::Choice {
            instructions: json!("Pick"),
            criteria: [
                ("a".to_string(), json!("A")),
                ("b".to_string(), Value::Null),
            ]
            .into_iter()
            .collect(),
        };
        assert_eq!(
            serde_json::to_value(&c).unwrap(),
            json!({"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": null}})
        );
    }

    /// The spec's response example decodes into the typed answers: integer
    /// zero probabilities become f64, the score legend keeps string keys.
    #[test]
    fn response_parses_the_documented_example() {
        let r: DecisionsResponse = serde_json::from_value(spec_response()).unwrap();
        assert_eq!(
            r.id.as_deref(),
            Some("gen-dec-1789738314-X5e5eKGQdvR9rblyX250")
        );
        assert_eq!(r.model.as_deref(), Some("typesafe/jev-1.13-20260917"));
        assert_eq!(r.provider.as_deref(), Some("TypeSafe"));
        assert_eq!(r.answers["is_bug"], DecisionAnswer::Noul { noul: 0.96 });
        match &r.answers["team"] {
            DecisionAnswer::Choice {
                choice,
                confidence,
                probabilities,
            } => {
                assert_eq!(choice, "payments");
                assert_eq!(*confidence, Some(0.75));
                assert_eq!(probabilities.as_ref().unwrap()["account"], 0.0);
                assert_eq!(probabilities.as_ref().unwrap()["payments"], 0.84);
            }
            other => panic!("expected choice, got {other:?}"),
        }
        match &r.answers["urgency"] {
            DecisionAnswer::Score {
                score,
                confidence,
                legend,
                probabilities,
            } => {
                assert_eq!(*score, 1.99);
                assert_eq!(*confidence, Some(0.99));
                assert_eq!(legend.as_ref().unwrap()["2"], "Blocking revenue right now");
                assert_eq!(probabilities.as_ref().unwrap()["2"], 0.99);
            }
            other => panic!("expected score, got {other:?}"),
        }
        let usage = r.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(476));
        assert_eq!(usage.output_tokens, Some(70));
        assert_eq!(usage.cost, Some(0.000019992));

        // Answers re-serialize in the wire shape, so the tool can echo them
        // (probabilities are f64, so the spec's integer `0` comes back as `0.0`).
        assert_eq!(
            serde_json::to_value(&r.answers["team"]).unwrap(),
            json!({
                "type": "choice",
                "choice": "payments",
                "confidence": 0.75,
                "probabilities": {"account": 0.0, "frontend": 0.16, "payments": 0.84}
            })
        );
    }

    /// Minimal answers (no optional fields) and an absent usage block decode.
    #[test]
    fn response_tolerates_missing_optionals() {
        let r: DecisionsResponse = serde_json::from_value(json!({
            "model": "typesafe/jev-1.13",
            "answers": {"ok": {"type": "choice", "choice": "yes"}}
        }))
        .unwrap();
        assert_eq!(
            r.answers["ok"],
            DecisionAnswer::Choice {
                choice: "yes".into(),
                confidence: None,
                probabilities: None
            }
        );
        assert!(r.usage.is_none());
        assert!(r.id.is_none());
    }
}
