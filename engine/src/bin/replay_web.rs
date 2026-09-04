//! Local HTTP host for the replay frontend.
//!
//! The HTTP accept loop never owns `ReplaySession`: a single worker thread
//! serializes game advancement and publishes immutable status snapshots.
use _engine::replay::{
    ReplaySession, ReplayStep, ReplayStepKind, StateDto, StrategySpec, WorkerSettings,
};
use serde::Serialize;
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, RwLock, mpsc};
use std::thread;

#[derive(Clone, Serialize)]
struct ReplayStatus {
    seed: u64,
    strategies: Vec<String>,
    current: StateDto,
    steps: Vec<ReplayStepSummary>,
    failure: Option<String>,
    busy: bool,
    complete: bool,
}

#[derive(Clone, Serialize)]
struct ReplayStepSummary {
    index: usize,
    kind: ReplayStepKind,
    round: usize,
    player: usize,
    chosen: String,
    result: String,
    era_event: Option<String>,
    strategy: String,
}

impl ReplayStatus {
    fn from_session(session: &ReplaySession, busy: bool) -> Self {
        Self {
            seed: session.seed,
            strategies: session.strategies.clone(),
            current: session.current_state(),
            steps: session
                .steps
                .iter()
                .map(|step| ReplayStepSummary {
                    index: step.index,
                    kind: step.kind,
                    round: step.round,
                    player: step.player,
                    chosen: step.chosen.clone(),
                    result: step.result.clone(),
                    era_event: step.era_event.clone(),
                    strategy: step.trace.strategy.clone(),
                })
                .collect(),
            failure: session.failure.clone(),
            busy,
            complete: session.is_complete(),
        }
    }
}

enum Command {
    Step,
}

fn publish(status: &Arc<RwLock<ReplayStatus>>, session: &ReplaySession, busy: bool) {
    *status.write().expect("status lock") = ReplayStatus::from_session(session, busy);
}

fn start_session_worker(
    mut session: ReplaySession,
) -> (
    mpsc::SyncSender<Command>,
    Arc<RwLock<ReplayStatus>>,
    Arc<RwLock<Vec<ReplayStep>>>,
) {
    // A rendezvous channel prevents hidden queued advances while a policy runs.
    let (tx, rx) = mpsc::sync_channel(0);
    let status = Arc::new(RwLock::new(ReplayStatus::from_session(&session, false)));
    let published = Arc::clone(&status);
    let details = Arc::new(RwLock::new(Vec::new()));
    let published_details = Arc::clone(&details);
    thread::spawn(move || {
        while let Ok(Command::Step) = rx.recv() {
            publish(&published, &session, true);
            let _ = session.step();
            *published_details.write().expect("details lock") = session.steps.clone();
            publish(&published, &session, false);
        }
    });
    (tx, status, details)
}

fn respond(stream: &mut TcpStream, status: &str, body: &[u8], content_type: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(body);
}

fn status_json(status: &Arc<RwLock<ReplayStatus>>) -> Vec<u8> {
    serde_json::to_vec(&*status.read().expect("status lock")).expect("status JSON")
}

fn serve_static(stream: &mut TcpStream, path: &str) {
    let (file, content_type) = match path {
        "/" | "/index.html" => ("replay.html", "text/html; charset=utf-8"),
        _ => {
            respond(
                stream,
                "404 Not Found",
                b"not found",
                "text/plain; charset=utf-8",
            );
            return;
        }
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("web");
    match fs::read(root.join(file)) {
        Ok(body) => respond(stream, "200 OK", &body, content_type),
        Err(error) => respond(
            stream,
            "500 Internal Server Error",
            format!("cannot read frontend asset: {error}").as_bytes(),
            "text/plain; charset=utf-8",
        ),
    }
}

fn serve(
    mut stream: TcpStream,
    tx: &mpsc::SyncSender<Command>,
    status: &Arc<RwLock<ReplayStatus>>,
    details: &Arc<RwLock<Vec<ReplayStep>>>,
) {
    let mut buf = [0u8; 4096];
    let Ok(n) = stream.read(&mut buf) else { return };
    let request = String::from_utf8_lossy(&buf[..n]);
    let first = request.lines().next().unwrap_or("");
    if first.starts_with("GET /api/status ") {
        let body = status_json(status);
        respond(
            &mut stream,
            "200 OK",
            &body,
            "application/json; charset=utf-8",
        );
    } else if let Some(index) = first
        .strip_prefix("GET /api/steps/")
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<usize>().ok())
    {
        let step = details.read().expect("details lock").get(index).cloned();
        match step {
            Some(step) => respond(
                &mut stream,
                "200 OK",
                &serde_json::to_vec(&step).expect("step JSON"),
                "application/json; charset=utf-8",
            ),
            None => respond(
                &mut stream,
                "404 Not Found",
                b"step not found",
                "text/plain; charset=utf-8",
            ),
        }
    } else if first.starts_with("POST /api/step ") {
        let current = status.read().expect("status lock").clone();
        if current.busy || current.complete || current.failure.is_some() {
            let body = serde_json::to_vec(&current).expect("status JSON");
            respond(
                &mut stream,
                "409 Conflict",
                &body,
                "application/json; charset=utf-8",
            );
        } else if tx.try_send(Command::Step).is_ok() {
            let body = status_json(status);
            respond(
                &mut stream,
                "202 Accepted",
                &body,
                "application/json; charset=utf-8",
            );
        } else {
            let body = status_json(status);
            respond(
                &mut stream,
                "409 Conflict",
                &body,
                "application/json; charset=utf-8",
            );
        }
    } else if first.starts_with("GET ") {
        let path = first.split_whitespace().nth(1).unwrap_or("");
        serve_static(&mut stream, path);
    } else {
        respond(
            &mut stream,
            "405 Method Not Allowed",
            b"method not allowed",
            "text/plain; charset=utf-8",
        );
    }
}

fn main() -> Result<(), String> {
    let mut seed = 7u64;
    let mut players = 4usize;
    let mut port = 8787u16;
    let mut python_bin = "python".to_string();
    let mut worker_timeout = 300u64;
    let mut policy_args = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            args.next().ok_or_else(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "--seed" => {
                seed = value(&mut args, "--seed")?
                    .parse()
                    .map_err(|_| "bad --seed")?
            }
            "--players" => {
                players = value(&mut args, "--players")?
                    .parse()
                    .map_err(|_| "bad --players")?
            }
            "--player" => policy_args.push(value(&mut args, "--player")?),
            "--port" => {
                port = value(&mut args, "--port")?
                    .parse()
                    .map_err(|_| "bad --port")?
            }
            "--python-bin" => python_bin = value(&mut args, "--python-bin")?,
            "--worker-timeout" => {
                worker_timeout = value(&mut args, "--worker-timeout")?
                    .parse()
                    .map_err(|_| "bad --worker-timeout")?
            }
            "--help" | "-h" => {
                println!(
                    "replay-web --seed 7 --players 4 --player heuristic [--player \"python:--ckpt net.pt --sims 200\" ...] [--port 8787] [--python-bin python] [--worker-timeout 300]"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown option {arg}")),
        }
    }
    if policy_args.is_empty() {
        policy_args = vec!["heuristic".into(); players];
    }
    if policy_args.len() < players {
        policy_args.extend(vec!["heuristic".into(); players - policy_args.len()]);
    }
    let specs = policy_args
        .iter()
        .map(|p| StrategySpec::parse(p))
        .collect::<Result<Vec<_>, _>>()?;
    let worker = WorkerSettings {
        python_bin,
        timeout_secs: worker_timeout,
    };
    let (tx, status, details) = start_session_worker(ReplaySession::new_with_worker(
        seed, players, specs, worker,
    )?);
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("cannot bind 127.0.0.1:{port}: {e}"))?;
    println!("Replay web session: http://127.0.0.1:{port}/ (Ctrl+C exits; session is memory-only)");
    for stream in listener.incoming().flatten() {
        let tx = tx.clone();
        let status = Arc::clone(&status);
        let details = Arc::clone(&details);
        thread::spawn(move || serve(stream, &tx, &status, &details));
    }
    Ok(())
}
