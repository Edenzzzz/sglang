//! Unified Worker Management Module
//!
//! Handles all aspects of worker lifecycle including discovery, initialization,
//! runtime management, and health monitoring.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures::future;
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, error, info, warn};

use crate::{
    core::{ConnectionMode, WorkerRegistry, WorkerType},
    policies::PolicyRegistry,
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

/// Unified worker management
pub struct WorkerManager;

#[derive(Debug, Default)]
struct ServerLoadSnapshot {
    load: isize,
    num_decode_tokens: Option<isize>,
    num_prefill_tokens: Option<isize>,
    num_waiting_reqs: Option<isize>,
    decode_batch_size: Option<isize>,
    decode_kv_tokens: Option<isize>,
    decode_remaining_tokens: Option<isize>,
    prefill_prefix_tokens: Option<isize>,
    predicted_forward_ms: Option<f64>,
    snapshot_timestamp: Option<f64>,
    snapshot_age_ms: Option<f64>,
}

impl WorkerManager {
    fn parse_worker_load(json: &Value, now_secs: f64) -> Option<ServerLoadSnapshot> {
        let array = json.as_array()?;

        fn sum_field(array: &[Value], field: &str) -> Option<isize> {
            let values: Vec<i64> = array
                .iter()
                .filter_map(|entry| entry.get(field).and_then(Value::as_i64))
                .collect();
            (!values.is_empty()).then(|| values.into_iter().sum::<i64>() as isize)
        }

        let snapshot_timestamp = array
            .iter()
            .filter_map(|entry| entry.get("snapshot_timestamp").and_then(Value::as_f64))
            .reduce(f64::min);
        let snapshot_age_ms =
            snapshot_timestamp.map(|timestamp| ((now_secs - timestamp).max(0.0)) * 1000.0);

        Some(ServerLoadSnapshot {
            load: sum_field(array, "num_tokens").unwrap_or(0),
            num_decode_tokens: sum_field(array, "num_decode_tokens"),
            num_prefill_tokens: sum_field(array, "num_prefill_tokens"),
            num_waiting_reqs: sum_field(array, "num_waiting_reqs"),
            decode_batch_size: sum_field(array, "decode_batch_size"),
            decode_kv_tokens: sum_field(array, "decode_kv_tokens"),
            decode_remaining_tokens: sum_field(array, "decode_remaining_tokens"),
            prefill_prefix_tokens: sum_field(array, "prefill_prefix_tokens"),
            predicted_forward_ms: array
                .iter()
                .filter_map(|entry| entry.get("predicted_forward_ms").and_then(Value::as_f64))
                .reduce(f64::max),
            snapshot_timestamp,
            snapshot_age_ms,
        })
    }

    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    /// Flush cache on all workers
    ///
    /// Sends a POST request to /flush_cache endpoint on all HTTP workers.
    /// Returns detailed results showing which workers succeeded and which failed.
    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> Result<FlushCacheResult, String> {
        warn!("Flushing cache for ALL workers - this may impact performance temporarily");

        let workers = worker_registry.get_all();

        let http_workers: Vec<_> = workers
            .iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return Ok(FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers: workers.len(),
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            });
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total workers)",
            http_workers.len(),
            workers.len()
        );

        let mut tasks = Vec::new();
        for worker in &http_workers {
            let url = worker.url().to_string();
            let flush_url = format!("{}/flush_cache", url);
            let mut request = client.post(&flush_url);

            if let Some(api_key) = worker.api_key() {
                request = request.header("Authorization", format!("Bearer {}", api_key));
            }

            let worker_url = url.clone();
            tasks.push(async move {
                let result = request.send().await;
                (worker_url, result)
            });
        }

        let results = future::join_all(tasks).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for (url, result) in results {
            match result {
                Ok(response) if response.status().is_success() => {
                    debug!("Successfully flushed cache on worker: {}", url);
                    successful.push(url);
                }
                Ok(response) => {
                    let error = format!("HTTP {}", response.status());
                    warn!("Failed to flush cache on worker {}: {}", url, error);
                    failed.push((url, error));
                }
                Err(e) => {
                    let error = e.to_string();
                    error!("Failed to connect to worker {}: {}", url, error);
                    failed.push((url, error));
                }
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush completed: {} succeeded, {} failed (out of {} HTTP workers)",
                successful.len(),
                failed.len(),
                http_workers.len()
            )
        };

        info!("{}", message);

        Ok(FlushCacheResult {
            successful,
            failed,
            total_workers: workers.len(),
            http_workers: http_workers.len(),
            message,
        })
    }
    async fn get_worker_load(
        url: &str,
        api_key: Option<&str>,
        client: &reqwest::Client,
    ) -> Option<ServerLoadSnapshot> {
        let load_url = format!("{}/get_load", url);
        let mut request = client.get(&load_url);

        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }

        match request.send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<Value>().await {
                    Ok(json) => {
                        let now_secs = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs_f64();
                        if let Some(snapshot) = Self::parse_worker_load(&json, now_secs) {
                            debug!(
                                "Worker {} load: total={}, decode={:?}, prefill={:?}, predicted_ms={:?}, age_ms={:?}",
                                url,
                                snapshot.load,
                                snapshot.num_decode_tokens,
                                snapshot.num_prefill_tokens,
                                snapshot.predicted_forward_ms,
                                snapshot.snapshot_age_ms,
                            );
                            Some(snapshot)
                        } else {
                            warn!(
                                "Invalid load response from {}: expected array, got {:?}",
                                url, json
                            );
                            None
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse load response from {}: {}", url, e);
                        None
                    }
                }
            }
            Ok(response) => {
                warn!(
                    "Failed to get load from {}: HTTP {}",
                    url,
                    response.status()
                );
                None
            }
            Err(e) => {
                warn!("Failed to connect to {} for load check: {}", url, e);
                None
            }
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        // Prepare tasks for parallel execution
        let mut tasks = Vec::new();
        for worker in &workers {
            let url = worker.url().to_string();
            let api_key = worker.api_key().clone();
            let worker_type = match worker.worker_type() {
                WorkerType::Regular => None,
                WorkerType::Prefill { .. } => Some("prefill".to_string()),
                WorkerType::Decode => Some("decode".to_string()),
            };
            let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
            let client = client.clone();

            tasks.push(async move {
                let snapshot = if is_http {
                    Self::get_worker_load(&url, api_key.as_deref(), &client)
                        .await
                        .unwrap_or_else(|| ServerLoadSnapshot {
                            load: -1,
                            ..Default::default()
                        })
                } else {
                    ServerLoadSnapshot {
                        load: -1,
                        ..Default::default()
                    }
                };

                WorkerLoadInfo {
                    worker: url,
                    worker_type,
                    load: snapshot.load,
                    num_decode_tokens: snapshot.num_decode_tokens,
                    num_prefill_tokens: snapshot.num_prefill_tokens,
                    num_waiting_reqs: snapshot.num_waiting_reqs,
                    decode_batch_size: snapshot.decode_batch_size,
                    decode_kv_tokens: snapshot.decode_kv_tokens,
                    decode_remaining_tokens: snapshot.decode_remaining_tokens,
                    prefill_prefix_tokens: snapshot.prefill_prefix_tokens,
                    predicted_forward_ms: snapshot.predicted_forward_ms,
                    snapshot_timestamp: snapshot.snapshot_timestamp,
                    snapshot_age_ms: snapshot.snapshot_age_ms,
                }
            });
        }

        let loads = future::join_all(tasks).await;

        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }
}

/// Load monitoring service that periodically fetches worker loads
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: reqwest::Client,
    interval: Duration,
    tx: watch::Sender<HashMap<String, isize>>,
    rx: watch::Receiver<HashMap<String, isize>>,
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    /// Create a new load monitor
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Start monitoring worker loads
    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        info!(
            "Starting load monitoring with interval: {:?}",
            self.interval
        );

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let interval = self.interval;
        let tx = self.tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(worker_registry, policy_registry, client, interval, tx).await;
        });

        *handle_guard = Some(handle);
    }

    /// Stop monitoring worker loads
    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    /// Get a receiver for load updates
    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

    /// The main monitoring loop
    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, isize>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();

            if power_of_two_policies.is_empty() {
                debug!("No PowerOfTwo policies found, skipping load fetch");
                continue;
            }

            let result = WorkerManager::get_all_worker_loads(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            for load_info in result.loads {
                loads.insert(load_info.worker, load_info.load);
            }

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                    loads.len(),
                    power_of_two_policies.len()
                );
                for policy in &power_of_two_policies {
                    policy.update_loads(&loads);
                }
                let _ = tx.send(loads);
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

    /// Check if monitoring is currently active
    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::WorkerManager;

    #[test]
    fn parses_structured_load_and_uses_oldest_dp_timestamp() {
        let response = json!([
            {
                "num_tokens": 100,
                "num_decode_tokens": 4,
                "num_prefill_tokens": 20,
                "num_waiting_reqs": 2,
                "decode_batch_size": 4,
                "decode_kv_tokens": 8000,
                "decode_remaining_tokens": 1000,
                "prefill_prefix_tokens": 3000,
                "predicted_forward_ms": 40.0,
                "snapshot_timestamp": 99.9
            },
            {
                "num_tokens": 200,
                "num_decode_tokens": 8,
                "num_prefill_tokens": 30,
                "num_waiting_reqs": 3,
                "decode_batch_size": 8,
                "decode_kv_tokens": 16000,
                "decode_remaining_tokens": 2000,
                "prefill_prefix_tokens": 4000,
                "predicted_forward_ms": 55.0,
                "snapshot_timestamp": 99.8
            }
        ]);

        let load = WorkerManager::parse_worker_load(&response, 100.0).unwrap();

        assert_eq!(load.load, 300);
        assert_eq!(load.num_decode_tokens, Some(12));
        assert_eq!(load.num_prefill_tokens, Some(50));
        assert_eq!(load.decode_batch_size, Some(12));
        assert_eq!(load.decode_kv_tokens, Some(24000));
        assert_eq!(load.predicted_forward_ms, Some(55.0));
        assert_eq!(load.snapshot_timestamp, Some(99.8));
        assert!((load.snapshot_age_ms.unwrap() - 200.0).abs() < 1e-6);
    }

    #[test]
    fn remains_compatible_with_legacy_token_only_response() {
        let response = json!([{"num_tokens": 123}]);

        let load = WorkerManager::parse_worker_load(&response, 100.0).unwrap();

        assert_eq!(load.load, 123);
        assert_eq!(load.predicted_forward_ms, None);
        assert_eq!(load.snapshot_age_ms, None);
    }

    #[test]
    fn rejects_non_array_response() {
        assert!(WorkerManager::parse_worker_load(&json!({"num_tokens": 1}), 1.0).is_none());
    }
}
