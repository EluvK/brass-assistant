//! Strategy adapter backed by a Python worker subprocess.
//!
//! One worker per `python:` seat owns a network checkpoint and answers
//! line-delimited JSON requests over stdin/stdout; the wire snapshot format is
//! the same `GameState.snapshot()` bytes the pyo3 bridge uses. See
//! `python/brass_ai/replay_worker.py` for the worker side of the protocol.

use crate::move_codec;
use crate::pymod::{SNAPSHOT_MAGIC, SNAPSHOT_VERSION};
use crate::replay::{ActionDto, DecisionTrace, StrategyAdapter};
use crate::rules::ResolvedMove;
use crate::state::GameState;
use base64::Engine as _;
use base64::prelude::BASE64_STANDARD;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

/// Split a `python:` seat's worker config into argv tokens. Whitespace
/// separates tokens unless inside single or double quotes, so arguments like
/// checkpoint paths may contain spaces.
fn split_worker_config(config: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut quote: Option<char> = None;
    for c in config.chars() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                } else {
                    token.push(c);
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                c if c.is_whitespace() => {
                    if !token.is_empty() {
                        tokens.push(std::mem::take(&mut token));
                    }
                }
                c => token.push(c),
            },
        }
    }
    if !token.is_empty() {
        tokens.push(token);
    }
    tokens
}

pub struct PythonWorkerStrategy {
    name: String,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    responses: Receiver<String>,
    request_id: u64,
    timeout: Duration,
}

impl PythonWorkerStrategy {
    /// Spawn `python -u -m brass_ai.replay_worker <worker_config>` and wait for
    /// the startup handshake, so a bad checkpoint or interpreter fails before
    /// the HTTP server starts serving.
    pub fn spawn(python_bin: &str, worker_config: &str, timeout: Duration) -> Result<Self, String> {
        let python_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("python");
        let mut child = Command::new(python_bin)
            .args(["-u", "-m", "brass_ai.replay_worker"])
            .args(split_worker_config(worker_config))
            .env("PYTHONPATH", &python_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                format!("cannot spawn {python_bin} -m brass_ai.replay_worker: {error}")
            })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "python worker stdout unavailable".to_string())?;
        let (tx, rx) = mpsc::channel::<String>();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if tx.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "python worker stdin unavailable".to_string())?;
        let mut worker = Self {
            name: String::new(),
            child: Some(child),
            stdin: Some(stdin),
            responses: rx,
            request_id: 0,
            timeout,
        };
        let ready = worker.recv_line("startup (checkpoint load)")?;
        let ready: Value = serde_json::from_str(&ready)
            .map_err(|error| format!("python worker handshake is not JSON: {error}"))?;
        if ready.get("type").and_then(Value::as_str) != Some("ready") {
            return Err(format!("python worker did not report ready: {ready}"));
        }
        worker.name = ready
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("python-worker")
            .to_string();
        Ok(worker)
    }

    fn kill(&mut self) {
        self.stdin.take();
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn recv_line(&mut self, what: &str) -> Result<String, String> {
        let deadline = Instant::now() + self.timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match self.responses.recv_timeout(remaining) {
                Ok(line) => return Ok(line),
                Err(RecvTimeoutError::Timeout) => {
                    self.kill();
                    return Err(format!(
                        "python worker timed out during {what} after {}s",
                        self.timeout.as_secs()
                    ));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(
                        "python worker exited unexpectedly; check its stderr output on the console"
                            .into(),
                    );
                }
            }
        }
    }
}

impl Drop for PythonWorkerStrategy {
    fn drop(&mut self) {
        self.kill();
    }
}

fn score_map(evidence: &Value, key: &str) -> Map<String, Value> {
    evidence
        .get(key)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

/// Convert the worker's evidence into a full-legal-set trace: one row per
/// structural operation, `score` = root visits (mcts) or policy probability
/// (policy), with the coalesced policy probability always shown in `note`.
fn trace_from_evidence(
    state: &GameState,
    legal: &[ResolvedMove],
    selected: &ResolvedMove,
    strategy: String,
    evidence: &Value,
) -> DecisionTrace {
    let mode = evidence
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("policy")
        .to_string();
    let policy = score_map(evidence, "policy");
    let visits = score_map(evidence, "visits");
    let root_value = evidence.get("root_value").and_then(Value::as_f64);
    let selected_key = move_codec::encode(selected);

    let mut grouped: HashMap<String, Vec<&ResolvedMove>> = HashMap::new();
    for mv in legal {
        grouped
            .entry(crate::heuristic_ai::operation_key(mv))
            .or_default()
            .push(mv);
    }
    let mut candidates: Vec<ActionDto> = grouped
        .into_iter()
        .map(|(_op_key, group)| {
            let representative = group
                .iter()
                .find(|mv| move_codec::encode(mv) == selected_key)
                .copied()
                .unwrap_or(group[0]);
            let canonical = move_codec::encode(representative);
            let probability: f64 = group
                .iter()
                .map(|mv| {
                    policy
                        .get(&move_codec::encode(mv))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                })
                .sum();
            let group_visits: u64 = group
                .iter()
                .map(|mv| {
                    visits
                        .get(&move_codec::encode(mv))
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                })
                .sum();
            let is_mcts = mode == "mcts";
            let (score, evaluated) = if is_mcts {
                (
                    (group_visits > 0).then_some(group_visits as f64),
                    group_visits > 0,
                )
            } else {
                (Some(probability), true)
            };
            let mut note = format!("策略 {:.1}%", probability * 100.0);
            if is_mcts && group_visits == 0 {
                note.push_str(" · 未被搜索展开");
            }
            ActionDto {
                canonical,
                description: representative.describe(state),
                action: representative.action().to_string(),
                selected: crate::heuristic_ai::operation_key(representative)
                    == crate::heuristic_ai::operation_key(selected),
                evaluated,
                score,
                card_score: None,
                card: crate::replay::move_card_name(state, representative),
                note: Some(note),
            }
        })
        .collect();
    candidates.sort_by(|a, b| {
        b.score
            .unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&a.score.unwrap_or(f64::NEG_INFINITY))
    });
    DecisionTrace {
        strategy,
        selected: selected_key,
        evidence_kind: format!("net-{mode}"),
        candidates,
        card_keep_scores: Vec::new(),
        root_value,
    }
}

impl StrategyAdapter for PythonWorkerStrategy {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn choose(
        &mut self,
        state: &mut GameState,
        legal: &[ResolvedMove],
    ) -> Result<(ResolvedMove, DecisionTrace), String> {
        if legal.is_empty() {
            return Err("current player has no legal move".into());
        }
        let mut wire = Vec::new();
        wire.extend_from_slice(SNAPSHOT_MAGIC);
        wire.push(SNAPSHOT_VERSION);
        wire.extend_from_slice(
            &state
                .snapshot_bytes()
                .map_err(|error| format!("cannot snapshot state for worker: {error}"))?,
        );
        let legal_canonicals: Vec<String> = legal.iter().map(move_codec::encode).collect();
        let request = json!({
            "type": "choose",
            "request_id": self.request_id,
            "snapshot": BASE64_STANDARD.encode(&wire),
            "legal": legal_canonicals,
        });
        {
            let stdin = self
                .stdin
                .as_mut()
                .ok_or_else(|| "python worker already closed".to_string())?;
            if let Err(error) = writeln!(stdin, "{request}").and_then(|_| stdin.flush()) {
                self.kill();
                return Err(format!("cannot write to python worker: {error}"));
            }
        }
        let request_id = self.request_id;
        self.request_id += 1;

        let line = self.recv_line("a choose request")?;
        let response: Value = serde_json::from_str(&line)
            .map_err(|error| format!("python worker response is not JSON: {error}: {line}"))?;
        let response_id = response.get("request_id").and_then(Value::as_u64);
        match response.get("type").and_then(Value::as_str) {
            Some("choice") if response_id == Some(request_id) => {}
            Some("error") => {
                return Err(format!(
                    "python worker failed: {}",
                    response
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                ));
            }
            _ => {
                return Err(format!(
                    "python worker protocol violation (request {request_id}): {line}"
                ));
            }
        }
        let chosen = response
            .get("canonical")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("python worker response lacks canonical: {line}"))?
            .to_string();
        let mv = legal
            .iter()
            .find(|mv| move_codec::encode(mv) == chosen)
            .ok_or_else(|| {
                format!("python worker chose {chosen:?} which is not in the legal set")
            })?;
        let evidence = response.get("evidence").cloned().unwrap_or(Value::Null);
        let trace = trace_from_evidence(state, legal, mv, self.name(), &evidence);
        Ok((mv.clone(), trace))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    #[test]
    fn evidence_maps_policy_probabilities_into_notes() {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(7), 2);
        let legal = crate::rules::legal_resolved_moves(&mut state);
        assert!(!legal.is_empty());
        let selected = &legal[0];
        let policy: Map<String, Value> = legal
            .iter()
            .enumerate()
            .map(|(i, mv)| (move_codec::encode(mv), Value::from(0.5 / (i as f64 + 1.0))))
            .collect();
        let evidence = json!({
            "mode": "policy",
            "policy": policy,
            "visits": {},
            "root_value": 0.25,
        });
        let trace = trace_from_evidence(&state, &legal, selected, "net".into(), &evidence);
        assert_eq!(trace.evidence_kind, "net-policy");
        assert_eq!(trace.root_value, Some(0.25));
        assert!(trace.candidates.iter().any(|c| c.selected));
        assert!(
            trace
                .candidates
                .iter()
                .all(|c| c.evaluated && c.score.is_some())
        );
    }

    #[test]
    fn worker_config_tokenizer_honors_quotes() {
        assert_eq!(
            split_worker_config("--ckpt checkpoints/a.pt --sims 200"),
            vec!["--ckpt", "checkpoints/a.pt", "--sims", "200"]
        );
        assert_eq!(
            split_worker_config("--ckpt \"checkpoints/my run.pt\" --device 'cpu'"),
            vec!["--ckpt", "checkpoints/my run.pt", "--device", "cpu"]
        );
        assert!(split_worker_config("   ").is_empty());
    }
}
