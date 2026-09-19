//! `POST /api/alpha/decisions`.

use anyhow::Result;

use crate::openrouter::{DecisionsBody, DecisionsReply, DecisionsResponse, OpenRouterClient};

/// Path of the decisions endpoint under [`OpenRouterClient::api_root`]. It is
/// outside `/api/v1`, so it is the one endpoint not built from `base_url`.
const DECISIONS_PATH: &str = "/api/alpha/decisions";

impl OpenRouterClient {
    /// `POST /api/alpha/decisions` - synchronous structured decisions. Returns
    /// the decoded body plus a generation id: the `X-Generation-Id` header when
    /// present (undocumented for this endpoint), else the body's `id`. A 2xx
    /// whose body cannot be decoded keeps a billing receipt on the error; an
    /// HTTP failure surfaces the upstream error body verbatim.
    pub async fn decisions(&self, req: &DecisionsBody) -> Result<DecisionsReply> {
        let rb = self
            .http
            .post(format!("{}{DECISIONS_PATH}", self.api_root()))
            .bearer_auth(&self.api_key)
            .json(req);
        let (body, header_id): (DecisionsResponse, _) =
            self.send_json_receipted(rb, DECISIONS_PATH).await?;
        let generation_id = header_id.or_else(|| body.id.clone());
        Ok(DecisionsReply {
            body,
            generation_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::{DecisionAnswer, DecisionQuestion, DecisionsBody, ProviderRouting};

    fn request() -> DecisionsBody {
        DecisionsBody {
            model: "typesafe/jev-1.13".into(),
            state: json!({"ticket": "blank screen after Pay"}),
            questions: [(
                "is_bug".to_string(),
                DecisionQuestion::Noul {
                    instructions: json!("Is this a defect?"),
                    criteria: None,
                },
            )]
            .into_iter()
            .collect(),
            provider: Some(ProviderRouting {
                order: vec!["typesafe".into()],
                ..Default::default()
            }),
        }
    }

    /// The body reaches `/api/alpha/decisions` (not under `/api/v1`), answers
    /// come back typed, and the header id wins over the body id.
    #[tokio::test]
    async fn decisions_posts_to_the_alpha_path_and_prefers_the_header_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/alpha/decisions"))
            .and(body_partial_json(json!({
                "model": "typesafe/jev-1.13",
                "state": {"ticket": "blank screen after Pay"},
                "questions": {"is_bug": {"type": "noul", "instructions": "Is this a defect?"}},
                "provider": {"order": ["typesafe"]}
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-header")
                    .set_body_json(json!({
                        "id": "gen-dec-body",
                        "model": "typesafe/jev-1.13-20260917",
                        "provider": "TypeSafe",
                        "answers": {"is_bug": {"type": "noul", "noul": 0.9}},
                        "usage": {"input_tokens": 12, "output_tokens": 3, "cost": 0.0000005}
                    })),
            )
            .mount(&server)
            .await;

        let client = crate::openrouter::OpenRouterClient::with_base_url(server.uri(), "test-key");
        let reply = client.decisions(&request()).await.unwrap();
        assert_eq!(reply.generation_id.as_deref(), Some("gen-header"));
        assert_eq!(
            reply.body.answers["is_bug"],
            DecisionAnswer::Noul { noul: 0.9 }
        );
        assert_eq!(reply.body.usage.unwrap().cost, Some(0.0000005));
        assert_eq!(reply.body.provider.as_deref(), Some("TypeSafe"));
    }

    /// Without the header the body's `id` is the generation id.
    #[tokio::test]
    async fn decisions_falls_back_to_the_body_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/alpha/decisions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "gen-dec-body",
                "model": "typesafe/jev-1.13",
                "answers": {},
                "usage": {"input_tokens": 1, "output_tokens": 0}
            })))
            .mount(&server)
            .await;
        let client = crate::openrouter::OpenRouterClient::with_base_url(server.uri(), "test-key");
        let reply = client.decisions(&request()).await.unwrap();
        assert_eq!(reply.generation_id.as_deref(), Some("gen-dec-body"));
    }

    #[tokio::test]
    async fn malformed_success_retains_receipt_but_http_failure_does_not() {
        for status in [200, 402] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/api/alpha/decisions"))
                .respond_with(
                    ResponseTemplate::new(status)
                        .insert_header("x-generation-id", "gen-paid-dec")
                        .set_body_string("{"),
                )
                .mount(&server)
                .await;
            let client =
                crate::openrouter::OpenRouterClient::with_base_url(server.uri(), "test-key");
            let error = client
                .decisions(&request())
                .await
                .expect_err("invalid response");
            let receipt = crate::billing::Receipt::from_error(&error);
            if status == 200 {
                let receipt = receipt.expect("unknown charge survives");
                assert_eq!(receipt.cost, None);
                assert_eq!(receipt.generation_id.as_deref(), Some("gen-paid-dec"));
            } else {
                assert!(receipt.is_none());
                assert!(error.to_string().contains("402"), "got: {error}");
                assert!(
                    error.to_string().contains("/api/alpha/decisions"),
                    "got: {error}"
                );
            }
        }
    }
}
