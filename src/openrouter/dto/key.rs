//! DTOs for `GET /api/v1/key` (API-key/account-level info).

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct KeyInfoResponse {
    pub data: KeyInfo,
}

/// Basic information about the API key in use (`GET /api/v1/key`). Every field is
/// optional/defaulted: the upstream schema evolves. `limit`/`limit_remaining`
/// cover only the key's own limit (`null` when it has none); guardrail and
/// workspace budgets are in `budgets`, and `effective_budget` is the one that
/// blocks first. Fields OpenRouter returns but we don't surface (e.g.
/// `limit_reset`, `expires_at`, BYOK period breakdowns) are ignored.
#[derive(Debug, Deserialize, Serialize)]
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
    /// Deprecated upstream: the same meaning as `is_management_key`.
    #[serde(default)]
    pub is_provisioning_key: Option<bool>,
    /// Whether this is an account management key.
    #[serde(default)]
    pub is_management_key: Option<bool>,
    /// The key's own spending cap in USD; `None` (null upstream) means the key
    /// has no cap of its own. A guardrail or workspace budget can still apply.
    #[serde(default)]
    pub limit: Option<f64>,
    /// Remaining USD under the key's own cap; `None` when it has none.
    #[serde(default)]
    pub limit_remaining: Option<f64>,
    /// Every budget layer checked for this key, in enforcement order
    /// (guardrails, workspace, then the key's own limit). Empty when nothing
    /// is budgeted.
    #[serde(default)]
    pub budgets: Option<Vec<KeyBudget>>,
    /// The budget that blocks requests first: a spent one, else the one with
    /// the least remaining. `None` when nothing is budgeted.
    #[serde(default)]
    pub effective_budget: Option<KeyBudget>,
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

/// One budget layer from `GET /api/v1/key` (`KeyBudget` in openapi.json). The
/// guardrail and workspace budget ids are opaque and not surfaced.
#[derive(Debug, Deserialize, Serialize)]
pub struct KeyBudget {
    /// Which layer: `workspace_default_member`, `workspace_default_key`,
    /// `member`, `api_key`, `workspace`, or `key_limit`.
    #[serde(default)]
    pub scope: Option<String>,
    /// Limit for the current window (USD).
    #[serde(default)]
    pub limit_usd: Option<f64>,
    /// Spend counted against the limit in the current window (USD).
    #[serde(default)]
    pub spend_usd: Option<f64>,
    /// `limit_usd - spend_usd`; zero or below means requests get a 403 until
    /// the window resets.
    #[serde(default)]
    pub remaining_usd: Option<f64>,
    /// `daily`, `weekly` or `monthly`; `None` for a lifetime budget.
    #[serde(default)]
    pub reset_interval: Option<String>,
    /// ISO 8601 UTC instant the window rolls over; `None` for a lifetime budget.
    #[serde(default)]
    pub resets_at: Option<String>,
    /// Whether BYOK inference spend counts toward this budget.
    #[serde(default)]
    pub include_byok: Option<bool>,
}

/// Legacy per-key rate limit. `requests` is signed because OpenRouter returns
/// `-1` to mean "no limit"; the field is deprecated and safe to ignore.
#[derive(Debug, Deserialize, Serialize)]
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
#[derive(Debug, Deserialize, Serialize)]
pub struct Credits {
    /// Total credits ever purchased or granted to the account (USD).
    #[serde(default)]
    pub total_credits: Option<f64>,
    /// Total credits ever consumed across all of the account's keys (USD).
    #[serde(default)]
    pub total_usage: Option<f64>,
    /// Derived remaining balance (`total_credits - total_usage`, USD); `None`
    /// when either input is missing. Serialized for the caller, ignored on input.
    #[serde(skip_deserializing)]
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
