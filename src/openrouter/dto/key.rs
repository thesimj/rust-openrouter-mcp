//! DTOs for `GET /api/v1/key` (API-key/account-level info).

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct KeyInfoResponse {
    pub data: KeyInfo,
}

/// Basic information about the API key in use (`GET /api/v1/key`). Every field is
/// optional/defaulted: the upstream schema evolves, and `limit`/`limit_remaining`
/// are `null` for unlimited keys. Fields OpenRouter returns but we don't surface
/// (e.g. `limit_reset`, `expires_at`, BYOK period breakdowns) are ignored.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct KeyInfo {
    /// Human-readable label, usually a masked key (e.g. "sk-or-v1-813...ca1").
    #[serde(default)]
    pub label: Option<String>,
    /// Opaque id of the user who owns the key (closest available "owner" identity).
    #[serde(default)]
    pub creator_user_id: Option<String>,
    /// Whether this is a free-tier key.
    #[serde(default)]
    pub is_free_tier: Option<bool>,
    /// Whether this key can provision (create/manage) other keys.
    #[serde(default)]
    pub is_provisioning_key: Option<bool>,
    /// Whether this is an account management key.
    #[serde(default)]
    pub is_management_key: Option<bool>,
    /// Spending cap in USD; `None` (null upstream) means unlimited.
    #[serde(default)]
    pub limit: Option<f64>,
    /// Remaining balance in USD; `None` means unlimited.
    #[serde(default)]
    pub limit_remaining: Option<f64>,
    /// Total credits consumed (USD).
    #[serde(default)]
    pub usage: Option<f64>,
    /// Credits consumed today (USD).
    #[serde(default)]
    pub usage_daily: Option<f64>,
    /// Credits consumed this week (USD).
    #[serde(default)]
    pub usage_weekly: Option<f64>,
    /// Credits consumed this month (USD).
    #[serde(default)]
    pub usage_monthly: Option<f64>,
    /// Spend on bring-your-own-key providers (USD), not billed as credits.
    #[serde(default)]
    pub byok_usage: Option<f64>,
    /// Legacy rate-limit descriptor (deprecated upstream; kept for completeness).
    #[serde(default)]
    pub rate_limit: Option<RateLimit>,
}

/// Legacy per-key rate limit. `requests` is signed because OpenRouter returns
/// `-1` to mean "no limit"; the field is deprecated and safe to ignore.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RateLimit {
    #[serde(default)]
    pub requests: Option<i64>,
    #[serde(default)]
    pub interval: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreditsResponse {
    pub data: Credits,
}

/// Account-wide credit totals (`GET /api/v1/credits`). Unlike [`KeyInfo`], which
/// is scoped to the single API key in use, these aggregate across all of the
/// account's keys: `total_credits` is everything purchased/granted and
/// `total_usage` is everything spent. `remaining` is derived (`total_credits -
/// total_usage`) on the way out so callers don't have to do the subtraction.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Credits {
    /// Total credits ever purchased or granted to the account (USD).
    #[serde(default)]
    pub total_credits: Option<f64>,
    /// Total credits ever consumed across all of the account's keys (USD).
    #[serde(default)]
    pub total_usage: Option<f64>,
    /// Derived remaining balance (`total_credits - total_usage`, USD); `None`
    /// when either input is missing. Serialized for the caller, ignored on input.
    #[serde(default)]
    pub remaining: Option<f64>,
}

impl Credits {
    /// Populate the derived [`remaining`](Self::remaining) balance from the two
    /// totals OpenRouter returns. No-op if either total is absent.
    pub fn with_remaining(mut self) -> Self {
        self.remaining = match (self.total_credits, self.total_usage) {
            (Some(c), Some(u)) => Some(c - u),
            _ => None,
        };
        self
    }
}
