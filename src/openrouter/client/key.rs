//! `GET /api/v1/key` endpoint.

use anyhow::{Context, Result};

use crate::openrouter::{KeyInfo, KeyInfoResponse, OpenRouterClient};

impl OpenRouterClient {
    /// `GET /api/v1/key` - basic information about the API key in use: its label,
    /// the owning `creator_user_id`, credit usage (total and per period), spending
    /// limit / remaining balance, tier/management flags, and the (deprecated)
    /// rate limit. This is key/account-level info, not the owner's name or email.
    pub async fn get_key_info(&self) -> Result<KeyInfo> {
        let resp = self
            .http
            .get(format!("{}/key", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("request to OpenRouter /key failed")?
            .error_for_status()
            .context("OpenRouter /key returned an error status")?;

        let parsed: KeyInfoResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /key response")?;
        Ok(parsed.data)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::OpenRouterClient;

    #[tokio::test]
    async fn get_key_info_parses_live_shaped_response() {
        let server = MockServer::start().await;
        // Body mirrors the real GET /api/v1/key payload (extra fields ignored,
        // rate_limit.requests is -1 = unlimited, limit is null = unlimited).
        Mock::given(method("GET"))
            .and(path("/key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "label": "sk-or-v1-813...ca1",
                    "creator_user_id": "user_39wcop8",
                    "is_free_tier": false,
                    "is_provisioning_key": false,
                    "is_management_key": false,
                    "limit": null,
                    "limit_reset": null,
                    "limit_remaining": null,
                    "expires_at": null,
                    "usage": 63.17,
                    "usage_daily": 3.25,
                    "usage_weekly": 10.28,
                    "usage_monthly": 10.46,
                    "byok_usage": 0,
                    "byok_usage_daily": 0,
                    "rate_limit": { "requests": -1, "interval": "10s", "note": "deprecated" }
                }
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let info = client.get_key_info().await.unwrap();
        assert_eq!(info.label.as_deref(), Some("sk-or-v1-813...ca1"));
        assert_eq!(info.creator_user_id.as_deref(), Some("user_39wcop8"));
        assert_eq!(info.is_free_tier, Some(false));
        assert_eq!(info.is_management_key, Some(false));
        assert!(info.limit.is_none(), "null limit -> unlimited");
        assert!(info.limit_remaining.is_none());
        assert_eq!(info.usage, Some(63.17));
        assert_eq!(info.usage_monthly, Some(10.46));
        assert_eq!(info.byok_usage, Some(0.0));
        let rl = info.rate_limit.unwrap();
        assert_eq!(rl.requests, Some(-1));
        assert_eq!(rl.interval.as_deref(), Some("10s"));
    }

    #[tokio::test]
    async fn get_key_info_errors_on_non_success_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/key"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "bad-key");
        let err = client.get_key_info().await.unwrap_err();
        assert!(err.to_string().contains("error status"));
    }
}
