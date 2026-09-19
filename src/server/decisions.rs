//! The `make_decisions` tool (`POST /api/alpha/decisions`) and its argument
//! structs. Decisions models (TypeSafe Jev) answer typed questions about a
//! state instead of generating text, and OpenRouter serves them on an alpha
//! endpoint outside `/api/v1` - `chat_completion` cannot reach them.
//!
//! Schema safety: `state` and every guidance value is a string, an object or
//! an array. That union is a nested `anyOf` of typed branches on a newtype
//! (never a `type` array, never a nullable union - see `schema.rs`). `criteria`
//! differs by question type (object or array), so it is a second newtype whose
//! `Default` means "absent"; an `Option<Criteria>` would render the nullable
//! `anyOf` the client-safety lint rejects.

use std::borrow::Cow;
use std::collections::BTreeMap;

use rmcp::{
    ErrorData, handler::server::wrapper::Parameters, model::CallToolResult, tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::decision_gen;
use crate::openrouter::{DecisionQuestion, DecisionsBody, NoulCriteria};
use crate::server::provider::ProviderRoutingArgs;
use crate::server::result::json_text_result;
use crate::server::schema::{de_lenient, scalarize_nullable};

use super::OpenRouterServer;

/// The one shape OpenRouter accepts for `noul` criteria.
const NOUL_CRITERIA_SHAPE: &str =
    "noul criteria, when given, must be an object with exactly the keys \"true\" and \"false\"";

/// A guidance value: a string, a JSON object, or an array. Used for `state`
/// and `instructions`. A string is sent as a string - it is never parsed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Guidance(pub Value);

impl JsonSchema for Guidance {
    fn schema_name() -> Cow<'static, str> {
        "Guidance".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "A plain string, or a JSON object or array of structured guidance.",
            "anyOf": [
                {"type": "string"},
                {"type": "object", "additionalProperties": true},
                {"type": "array"}
            ]
        })
    }
}

impl<'de> Deserialize<'de> for Guidance {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        match Value::deserialize(d)? {
            v @ (Value::String(_) | Value::Object(_) | Value::Array(_)) => Ok(Guidance(v)),
            other => Err(D::Error::custom(format!(
                "expected a string, an object or an array, got {other}"
            ))),
        }
    }
}

/// A question's `criteria`: an object (noul, choice) or an array (score).
/// `Value::Null` means absent. Like [`de_lenient`], a JSON *string* holding
/// the object or array is parsed - a bare string is never valid criteria, so
/// this is unambiguous - and `null` or a blank string means absent.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Criteria(pub Value);

impl Default for Criteria {
    fn default() -> Self {
        Criteria(Value::Null)
    }
}

impl JsonSchema for Criteria {
    fn schema_name() -> Cow<'static, str> {
        "Criteria".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "noul: optional {\"true\": guidance, \"false\": guidance}. choice: \
                an object mapping each label to its description (or null). score: an array of \
                levels, lowest first (2 to 10).",
            "anyOf": [
                {"type": "object", "additionalProperties": true},
                {"type": "array"}
            ]
        })
    }
}

impl<'de> Deserialize<'de> for Criteria {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let parsed = match Value::deserialize(d)? {
            Value::Null => Value::Null,
            Value::String(s) if s.trim().is_empty() => Value::Null,
            Value::String(s) => serde_json::from_str(&s)
                .map_err(|e| D::Error::custom(format!("invalid JSON in criteria string: {e}")))?,
            other => other,
        };
        match parsed {
            Value::Null | Value::Object(_) | Value::Array(_) => Ok(Criteria(parsed)),
            other => Err(D::Error::custom(format!(
                "criteria must be an object or an array, got {other}"
            ))),
        }
    }
}

/// One question in the `questions` map of `make_decisions`.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct QuestionArgs {
    /// "noul" (a true/false question answered with a probability), "choice"
    /// (pick one label from `criteria`), or "score" (place the state on the
    /// scale `criteria` defines).
    #[serde(rename = "type")]
    pub kind: String,
    /// What to decide about the state, e.g. "Is the customer reporting a
    /// defect?". A string, or a JSON object/array of structured guidance.
    pub instructions: Guidance,
    /// noul: optional {"true": "...", "false": "..."} describing each outcome.
    /// choice: REQUIRED object, label -> description (or null), up to 255
    /// labels. score: REQUIRED array of 2 to 10 levels, lowest first; the
    /// answer's `score` is a float over the index range.
    #[serde(default)]
    pub criteria: Criteria,
}

impl QuestionArgs {
    /// Check the vocabulary and the criteria shape for this question type,
    /// naming the question in every error. Content checks (blank text, level
    /// counts) run later in [`decision_gen::validate`].
    fn into_question(self, name: &str) -> Result<DecisionQuestion, ErrorData> {
        let invalid =
            |msg: String| ErrorData::invalid_params(format!("questions[{name:?}]: {msg}"), None);
        let kind = self.kind.trim().to_ascii_lowercase();
        let instructions = self.instructions.0;
        match kind.as_str() {
            "noul" => {
                let criteria = match self.criteria.0 {
                    Value::Null => None,
                    Value::Object(mut m) => {
                        // Take both keys; anything left over is a stray key.
                        let taken = (m.remove("true"), m.remove("false"));
                        match taken {
                            (Some(when_true), Some(when_false)) if m.is_empty() => {
                                Some(NoulCriteria {
                                    when_true,
                                    when_false,
                                })
                            }
                            _ => return Err(invalid(NOUL_CRITERIA_SHAPE.into())),
                        }
                    }
                    _ => return Err(invalid(NOUL_CRITERIA_SHAPE.into())),
                };
                Ok(DecisionQuestion::Noul {
                    instructions,
                    criteria,
                })
            }
            "choice" => match self.criteria.0 {
                Value::Object(m) => Ok(DecisionQuestion::Choice {
                    instructions,
                    criteria: m.into_iter().collect::<BTreeMap<_, _>>(),
                }),
                _ => Err(invalid(
                    "choice requires criteria: an object mapping each label to its description \
                     (or null)"
                        .into(),
                )),
            },
            "score" => match self.criteria.0 {
                Value::Array(levels) => Ok(DecisionQuestion::Score {
                    instructions,
                    criteria: levels,
                }),
                _ => Err(invalid(
                    "score requires criteria: an array of levels, lowest first".into(),
                )),
            },
            other => Err(invalid(format!(
                "type must be one of \"noul\", \"choice\", \"score\" (got {other:?})"
            ))),
        }
    }
}

/// Arguments for the `make_decisions` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct MakeDecisionsArgs {
    /// Decisions model id: "typesafe/jev-1.13", or the alias
    /// "~typesafe/jev-latest" (the leading tilde is required). Discover them
    /// with list_models using output_modalities="decisions".
    pub model: String,
    /// The content to evaluate: a string, a JSON object (e.g. {"ticket": "...",
    /// "customer_tier": "enterprise"}), or an array of strings/objects. Pass
    /// objects as JSON objects, not as JSON text. Text only - no images or audio.
    pub state: Guidance,
    /// Questions keyed by names you choose (the model never sees the names);
    /// each answer comes back under its name. Every question has a `type`
    /// ("noul" | "choice" | "score"), `instructions`, and - for choice and
    /// score - `criteria`.
    // No `serde(default)` on purpose: `questions` must appear in the schema's
    // `required` list. `de_lenient` still accepts a stringified object.
    #[serde(deserialize_with = "de_lenient")]
    pub questions: BTreeMap<String, QuestionArgs>,
    /// Provider block for this request: routing only, as {"order": [slugs],
    /// "only": [slugs], "ignore": [slugs], "allow_fallbacks": bool,
    /// "require_parameters": bool, "zdr": bool, "sort":
    /// "price"|"throughput"|"latency"|"exacto", "sort_partition": "model"|"none"}.
    /// This endpoint has no per-provider `options` passthrough.
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: ProviderRoutingArgs,
}

#[tool_router(router = decisions_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Ask an OpenRouter decisions model (TypeSafe Jev: typesafe/jev-1.13, alias \
        ~typesafe/jev-latest - the tilde is required) typed questions about a `state` and get \
        back probabilities, not text. This is a synchronous call to POST /api/alpha/decisions \
        (an alpha endpoint; chat_completion cannot use these models). Use it for routing, \
        classification, triage, verification of another model's answer, or any decision point \
        where a fast (~250 ms), calibrated, structured answer beats generated prose. `state` is \
        what to judge: a string, a JSON object, or an array of them (text only). `questions` is \
        an object keyed by names you choose; each value is {type, instructions, criteria}: \
        type \"noul\" - a true/false question, optional criteria {\"true\": what makes it true, \
        \"false\": what makes it false}, answered as {noul: probability 0..1}; type \"choice\" - \
        pick one label, criteria REQUIRED as {label: description or null} (up to 255 labels), \
        answered as {choice: label, confidence, probabilities: {label: p}}; type \"score\" - a \
        graded scale, criteria REQUIRED as an array of 2 to 10 levels lowest first, answered as \
        {score: float over the level indexes, confidence, legend: {index: level}, \
        probabilities: {index: p}}. Example: {\"state\": {\"ticket\": \"Checkout shows a blank \
        page after Pay\"}, \"questions\": {\"is_bug\": {\"type\": \"noul\", \"instructions\": \
        \"Is the customer reporting a software defect?\"}, \"team\": {\"type\": \"choice\", \
        \"instructions\": \"Which team should own this ticket?\", \"criteria\": {\"payments\": \
        \"Checkout or billing\", \"frontend\": \"Rendering or layout\"}}, \"urgency\": {\"type\": \
        \"score\", \"instructions\": \"How urgent is this ticket?\", \"criteria\": [\"Can wait\", \
        \"This week\", \"Blocking revenue now\"]}}}. `confidence` (0..1) appears only on choice \
        and score answers. `provider` carries routing only (order, only, ignore, \
        allow_fallbacks, require_parameters, zdr, sort) - no passthrough options. Returns JSON: \
        model, id, provider, answers {name: answer}, usage {input_tokens, output_tokens, cost in \
        USD}, and generation_id - pass that to get_generation for the full cost record. \
        Pricing is per input token only (Jev: $0.042/M). Discover decisions models with \
        list_models using output_modalities=\"decisions\" - they are not in the default model \
        list.",
        annotations(
            title = "Make Decisions",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    pub(crate) async fn make_decisions(
        &self,
        Parameters(args): Parameters<MakeDecisionsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let _work = self.admit_work()?;
        let model = args.model.clone();
        let questions = args
            .questions
            .into_iter()
            .map(|(name, q)| q.into_question(&name).map(|q| (name, q)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let body = DecisionsBody {
            model: args.model,
            state: args.state.0,
            questions,
            provider: args.provider.into_routing()?,
        };
        decision_gen::validate(&body)
            .map_err(|e| ErrorData::invalid_params(format!("{e:#}"), None))?;

        match decision_gen::decide(&self.client, &body).await {
            Ok(result) => {
                self.stats.record_text(&model, result.cost).await;
                json_text_result(&result.to_json())
            }
            Err(e) => {
                self.stats.record_text_failure(&model, &e).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::schema::schema_json;
    use crate::server::test_support::{server_for, tool_result_json};
    use rmcp::model::ErrorCode;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn args(v: Value) -> Parameters<MakeDecisionsArgs> {
        Parameters(serde_json::from_value(v).unwrap())
    }

    /// The spec's ticket-triage example as a tool call.
    fn triage() -> Value {
        json!({
            "model": "typesafe/jev-1.13",
            "state": {"customer_tier": "enterprise", "ticket": "Blank screen after Pay."},
            "questions": {
                "is_bug": {
                    "type": "noul",
                    "instructions": "Is the customer reporting a software defect?",
                    "criteria": {
                        "true": "The customer describes broken or unexpected product behavior.",
                        "false": "The customer is asking a question or requesting a feature."
                    }
                },
                "team": {
                    "type": "choice",
                    "instructions": "Which team should own this ticket?",
                    "criteria": {"payments": "Checkout or billing.", "frontend": null}
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this ticket?",
                    "criteria": ["Can wait", "This week", "Blocking revenue now"]
                }
            },
            "provider": {"order": ["typesafe"]}
        })
    }

    /// The documented body - object state, all three question types with the
    /// `"true"`/`"false"` noul keys, a null choice description, provider.order -
    /// reaches `/api/alpha/decisions`; answers come back in the wire shape and
    /// the cost lands in the usage stats as a text generation.
    #[tokio::test]
    async fn make_decisions_forwards_the_body_and_returns_typed_answers() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/alpha/decisions"))
            .and(body_partial_json(json!({
                "model": "typesafe/jev-1.13",
                "state": {"customer_tier": "enterprise", "ticket": "Blank screen after Pay."},
                "questions": {
                    "is_bug": {
                        "type": "noul",
                        "instructions": "Is the customer reporting a software defect?",
                        "criteria": {
                            "true": "The customer describes broken or unexpected product behavior.",
                            "false": "The customer is asking a question or requesting a feature."
                        }
                    },
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should own this ticket?",
                        "criteria": {"payments": "Checkout or billing.", "frontend": null}
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "How urgent is this ticket?",
                        "criteria": ["Can wait", "This week", "Blocking revenue now"]
                    }
                },
                "provider": {"order": ["typesafe"]}
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-dec")
                    .set_body_json(json!({
                        "id": "gen-dec",
                        "model": "typesafe/jev-1.13-20260917",
                        "provider": "TypeSafe",
                        "answers": {
                            "is_bug": {"type": "noul", "noul": 0.96},
                            "team": {"type": "choice", "choice": "payments", "confidence": 0.75,
                                     "probabilities": {"payments": 0.84, "frontend": 0.16}},
                            "urgency": {"type": "score", "score": 1.99, "confidence": 0.99,
                                        "legend": {"0": "Can wait", "1": "This week", "2": "Blocking revenue now"},
                                        "probabilities": {"0": 0.0, "1": 0.01, "2": 0.99}}
                        },
                        "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992}
                    })),
            )
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let res = server.make_decisions(args(triage())).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["model"], "typesafe/jev-1.13-20260917");
        assert_eq!(v["id"], "gen-dec");
        assert_eq!(v["provider"], "TypeSafe");
        assert_eq!(
            v["answers"]["is_bug"],
            json!({"type": "noul", "noul": 0.96})
        );
        assert_eq!(v["answers"]["team"]["choice"], "payments");
        assert_eq!(v["answers"]["team"]["probabilities"]["payments"], 0.84);
        assert_eq!(v["answers"]["urgency"]["score"], 1.99);
        assert_eq!(
            v["answers"]["urgency"]["legend"]["2"],
            "Blocking revenue now"
        );
        assert_eq!(
            v["usage"],
            json!({"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992})
        );
        assert_eq!(v["generation_id"], "gen-dec");

        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["text_generations"], 1);
        // Jev charges fractions of a cent; the snapshot keeps six decimals so
        // the charge is visible, and it was known, not unknown.
        assert_eq!(stats["actual_cost_usd"], 0.00002);
        assert_eq!(stats["unknown_cost_count"], 0);
        assert_eq!(stats["by_model"]["typesafe/jev-1.13"]["requests"], 1);
    }

    /// A stringified `questions` object and stringified `criteria` (clients
    /// that serialize nested objects) are accepted; a string `state` is sent
    /// as a string, never parsed; no provider block is sent when none was given.
    #[tokio::test]
    async fn make_decisions_accepts_stringified_nested_objects_and_keeps_string_state() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/alpha/decisions"))
            .and(body_partial_json(json!({
                "state": "{\"looks\": \"like json\"}",
                "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B"}}}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "m", "answers": {"q": {"type": "choice", "choice": "a"}},
                "usage": {"input_tokens": 3, "output_tokens": 1}
            })))
            .mount(&mock)
            .await;
        let server = server_for(mock.uri());
        let res = server
            .make_decisions(args(json!({
                "model": "m",
                "state": "{\"looks\": \"like json\"}",
                "questions": "{\"q\": {\"type\": \"Choice\", \"instructions\": \"Pick\", \
                              \"criteria\": \"{\\\"a\\\": \\\"A\\\", \\\"b\\\": \\\"B\\\"}\"}}"
            })))
            .await
            .unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["answers"]["q"]["choice"], "a");
        // Absent usage cost is reported as null, not invented.
        assert!(v["usage"]["cost"].is_null());
        let sent: Value =
            serde_json::from_slice(&mock.received_requests().await.unwrap()[0].body).unwrap();
        assert!(sent.get("provider").is_none(), "sent: {sent}");
    }

    /// Shape errors are invalid params caught before any HTTP call and name
    /// the question; nothing is counted as a request.
    #[tokio::test]
    async fn make_decisions_rejects_bad_shapes_before_any_call() {
        let mock = MockServer::start().await;
        let server = server_for(mock.uri());
        let call = |questions: Value| {
            server.make_decisions(args(json!({
                "model": "m", "state": "text", "questions": questions
            })))
        };

        let cases: [(Value, &str); 7] = [
            (json!({}), "at least one question"),
            (
                json!({"q": {"type": "maybe", "instructions": "x"}}),
                "\"noul\", \"choice\", \"score\"",
            ),
            (
                json!({"q": {"type": "noul", "instructions": "x", "criteria": {"yes": "y"}}}),
                "exactly the keys \"true\" and \"false\"",
            ),
            (
                json!({"q": {"type": "choice", "instructions": "x"}}),
                "choice requires criteria",
            ),
            (
                json!({"q": {"type": "score", "instructions": "x", "criteria": {"a": 1}}}),
                "score requires criteria",
            ),
            (
                json!({"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}),
                "at least two levels",
            ),
            (
                json!({"q": {"type": "noul", "instructions": "  "}}),
                "instructions must not be blank",
            ),
        ];
        for (questions, needle) in cases {
            let err = call(questions.clone()).await.unwrap_err();
            assert_eq!(err.code, ErrorCode::INVALID_PARAMS, "{questions}");
            assert!(err.message.contains(needle), "{questions}: {}", err.message);
            assert!(
                err.message.contains("questions[\"q\"]") || needle.contains("at least one"),
                "{}",
                err.message
            );
        }

        let err = server
            .make_decisions(args(json!({
                "model": "m", "state": "  ", "questions": {"q": {"type": "noul", "instructions": "x"}}
            })))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("state"), "{}", err.message);

        let err = server
            .make_decisions(args(json!({
                "model": "m", "state": "text",
                "questions": {"q": {"type": "noul", "instructions": "x"}},
                "provider": {"sort": "cheapest"}
            })))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("sort"), "{}", err.message);

        assert_eq!(mock.received_requests().await.unwrap().len(), 0);
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["requests_total"], 0);
    }

    /// Argument-level type errors (a number for `state`, a bare string for
    /// `criteria` that is not JSON) fail deserialization with a clear message.
    #[test]
    fn guidance_and_criteria_reject_scalars() {
        let err = serde_json::from_value::<MakeDecisionsArgs>(json!({
            "model": "m", "state": 42, "questions": {}
        }))
        .unwrap_err();
        assert!(
            err.to_string().contains("string, an object or an array"),
            "{err}"
        );

        let err = serde_json::from_value::<QuestionArgs>(json!({
            "type": "score", "instructions": "x", "criteria": "not json"
        }))
        .unwrap_err();
        assert!(
            err.to_string().contains("invalid JSON in criteria"),
            "{err}"
        );

        let err = serde_json::from_value::<QuestionArgs>(json!({
            "type": "score", "instructions": "x", "criteria": 7
        }))
        .unwrap_err();
        assert!(err.to_string().contains("object or an array"), "{err}");

        // null / blank / absent criteria all mean "absent".
        for criteria in [json!(null), json!("  ")] {
            let q: QuestionArgs = serde_json::from_value(json!({
                "type": "noul", "instructions": "x", "criteria": criteria
            }))
            .unwrap();
            assert!(q.criteria.0.is_null());
        }
        let q: QuestionArgs =
            serde_json::from_value(json!({"type": "noul", "instructions": "x"})).unwrap();
        assert!(q.criteria.0.is_null());
    }

    /// An upstream failure is an internal error that counts as a failed request.
    #[tokio::test]
    async fn make_decisions_surfaces_upstream_errors_and_counts_the_failure() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/alpha/decisions"))
            .respond_with(ResponseTemplate::new(402).set_body_string("insufficient credits"))
            .mount(&mock)
            .await;
        let server = server_for(mock.uri());
        let err = server.make_decisions(args(triage())).await.unwrap_err();
        assert_eq!(err.code, ErrorCode::INTERNAL_ERROR);
        assert!(err.message.contains("402"), "got: {}", err.message);
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["requests_total"], 1);
        assert_eq!(stats["requests_failed"], 1);
    }

    /// The tools/list schema: `model`, `state` and `questions` are required,
    /// `state` and `instructions` are the guidance `anyOf` (string | object |
    /// array, no null branch), `criteria` is an optional `$ref` (no nullable
    /// union), and `questions` is an open object of `QuestionArgs`.
    #[test]
    fn schema_requires_state_and_questions_and_carries_typed_unions() {
        let schema = schema_json::<MakeDecisionsArgs>();
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for name in ["model", "state", "questions"] {
            assert!(required.contains(&name), "{required:?}");
        }
        assert!(!required.contains(&"provider"), "{required:?}");
        assert_eq!(
            schema["properties"]["state"]["$ref"],
            json!("#/$defs/Guidance")
        );
        assert_eq!(
            schema["$defs"]["Guidance"]["anyOf"],
            json!([
                {"type": "string"},
                {"type": "object", "additionalProperties": true},
                {"type": "array"}
            ])
        );
        assert_eq!(
            schema["properties"]["questions"]["additionalProperties"]["$ref"],
            json!("#/$defs/QuestionArgs")
        );
        let question = &schema["$defs"]["QuestionArgs"];
        let q_required = question["required"].as_array().unwrap();
        assert!(q_required.contains(&json!("type")), "{q_required:?}");
        assert!(
            q_required.contains(&json!("instructions")),
            "{q_required:?}"
        );
        assert!(!q_required.contains(&json!("criteria")), "{q_required:?}");
        assert_eq!(
            question["properties"]["criteria"]["$ref"],
            json!("#/$defs/Criteria")
        );
        assert!(question["properties"]["criteria"].get("default").is_none());
        assert_eq!(
            schema["$defs"]["Criteria"]["anyOf"],
            json!([{"type": "object", "additionalProperties": true}, {"type": "array"}])
        );
    }
}
