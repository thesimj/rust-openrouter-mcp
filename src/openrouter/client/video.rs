//! Asynchronous video-generation endpoints: list, submit, poll, download.

use anyhow::{Context, Result};

use crate::openrouter::{
    OpenRouterClient, VideoModel, VideoModelsResponse, VideoPollResponse, VideoSubmitBody,
    VideoSubmitResponse, content_type,
};

impl OpenRouterClient {
    /// `GET /api/v1/videos/models` - video-generation models with `pricing_skus`
    /// (per video-second / per video-token), resolutions, durations, etc.
    pub async fn list_video_models(&self) -> Result<Vec<VideoModel>> {
        let resp = self
            .http
            .get(format!("{}/videos/models", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("request to OpenRouter /videos/models failed")?
            .error_for_status()
            .context("OpenRouter /videos/models returned an error status")?;

        let parsed: VideoModelsResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /videos/models response")?;
        Ok(parsed.data)
    }

    /// `POST /api/v1/videos` - submit an asynchronous video-generation job. This
    /// is **not** the chat endpoint: it returns `202` with a job id to poll. On a
    /// non-2xx status the upstream error body is surfaced verbatim.
    pub async fn submit_video(&self, req: &VideoSubmitBody) -> Result<VideoSubmitResponse> {
        let rb = self
            .http
            .post(format!("{}/videos", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req);
        let resp = self.send_checked(rb, "/videos", "/videos").await?;
        let parsed: VideoSubmitResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /videos submit response")?;
        Ok(parsed)
    }

    /// `GET /api/v1/videos/{id}` - poll a submitted video job for its status and,
    /// once complete, the (unsigned) download URLs and usage.
    pub async fn poll_video(&self, job_id: &str) -> Result<VideoPollResponse> {
        let rb = self
            .http
            .get(format!("{}/videos/{job_id}", self.base_url))
            .bearer_auth(&self.api_key);
        let resp = self
            .send_checked(rb, "/videos/{id}", &format!("/videos/{job_id}"))
            .await?;
        let parsed: VideoPollResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /videos/{id} response")?;
        Ok(parsed)
    }

    /// `GET /api/v1/videos/{id}/content?index=N` - download one generated clip.
    /// Returns `(content_type, bytes)`; the content type (e.g. `video/mp4`) is
    /// used to choose the file extension.
    pub async fn download_video(&self, job_id: &str, index: usize) -> Result<(String, Vec<u8>)> {
        let rb = self
            .http
            .get(format!("{}/videos/{job_id}/content", self.base_url))
            .query(&[("index", index.to_string())])
            .bearer_auth(&self.api_key);
        let resp = self
            .send_checked(
                rb,
                "/videos/{id}/content",
                &format!("/videos/{job_id}/content"),
            )
            .await?;
        let content_type = content_type(&resp, "video/mp4");
        let bytes = resp
            .bytes()
            .await
            .context("failed to read video content bytes")?
            .to_vec();
        Ok((content_type, bytes))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::{OpenRouterClient, VideoSubmitBody};

    #[tokio::test]
    async fn list_video_models_parses_pricing_skus() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/videos/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    {"id": "google/veo", "pricing_skus": {"duration_seconds": "0.1"}}
                ]
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let vms = client.list_video_models().await.unwrap();
        assert_eq!(vms.len(), 1);
        assert_eq!(vms[0].id, "google/veo");
        assert_eq!(
            vms[0]
                .pricing_skus
                .get("duration_seconds")
                .map(String::as_str),
            Some("0.1")
        );
    }

    #[tokio::test]
    async fn submit_video_posts_body_and_parses_job_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/videos"))
            .and(body_partial_json(
                json!({ "model": "google/veo-3.1", "prompt": "a dog" }),
            ))
            // The async job API responds 202 with a job id and polling url.
            .respond_with(ResponseTemplate::new(202).set_body_json(json!({
                "id": "vid-1",
                "polling_url": "https://openrouter.ai/api/v1/videos/vid-1",
                "status": "pending"
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let body = VideoSubmitBody {
            model: "google/veo-3.1".to_string(),
            prompt: "a dog".to_string(),
            duration: Some(4),
            resolution: None,
            aspect_ratio: Some("16:9".to_string()),
            size: None,
            frame_images: vec![],
            input_references: vec![],
            generate_audio: Some(false),
            seed: None,
        };
        let resp = client.submit_video(&body).await.unwrap();
        assert_eq!(resp.id, "vid-1");
    }

    #[tokio::test]
    async fn submit_video_surfaces_error_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/videos"))
            .respond_with(ResponseTemplate::new(400).set_body_string("{\"error\":\"unsupported\"}"))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let body = VideoSubmitBody {
            model: "m".to_string(),
            prompt: "p".to_string(),
            duration: None,
            resolution: None,
            aspect_ratio: None,
            size: None,
            frame_images: vec![],
            input_references: vec![],
            generate_audio: None,
            seed: None,
        };
        let err = client.submit_video(&body).await.unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }

    #[tokio::test]
    async fn poll_video_parses_status_urls_and_usage() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/videos/vid-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "vid-1",
                "generation_id": "gen-7",
                "status": "completed",
                "unsigned_urls": ["https://cdn/0.mp4", "https://cdn/1.mp4"],
                "usage": { "cost": 0.9, "is_byok": false }
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let poll = client.poll_video("vid-1").await.unwrap();
        assert_eq!(poll.status, "completed");
        assert_eq!(poll.generation_id.as_deref(), Some("gen-7"));
        assert_eq!(poll.unsigned_urls.len(), 2);
        assert_eq!(poll.usage.unwrap().cost, Some(0.9));
    }

    #[tokio::test]
    async fn download_video_returns_content_type_and_bytes() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/videos/vid-1/content"))
            .and(query_param("index", "2"))
            .respond_with(
                ResponseTemplate::new(200)
                    // A charset suffix must be stripped to the bare MIME type.
                    .insert_header("content-type", "video/webm; charset=binary")
                    .set_body_bytes(b"WEBM".to_vec()),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let (mime, bytes) = client.download_video("vid-1", 2).await.unwrap();
        assert_eq!(mime, "video/webm");
        assert_eq!(bytes, b"WEBM");
    }
}
