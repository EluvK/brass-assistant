//! In-memory, developer-facing game replay and decision diagnostics.
//!
//! This deliberately owns no persistence format. Snapshots are only used to
//! make an already generated step independently inspectable during one CLI
//! process lifetime.

use crate::bridge::move_codec;
use crate::data::Era;
use crate::engine::{TurnResult, advance_turn, handle_turn_result};
use crate::heuristic_ai;
use crate::map::{ALL_LOCATIONS, CITY_COUNT, connections};
use crate::mcts_ai::{self, MctsConfig};
use crate::random_ai;
use crate::rules::{Move, apply_move, legal_moves};
use crate::state::{Card, GameState, loc_from_key};
use rand_chacha::ChaCha12Rng;
use rand_chacha::rand_core::SeedableRng;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum StrategySpec {
    Heuristic,
    Random,
    Mcts { simulations: usize },
    Python { worker_config: String },
}

impl StrategySpec {
    pub fn parse(value: &str, simulations: usize) -> Result<Self, String> {
        match value {
            "heuristic" => Ok(Self::Heuristic),
            "random" => Ok(Self::Random),
            "mcts" => Ok(Self::Mcts { simulations }),
            value if value.starts_with("python:") => Ok(Self::Python {
                worker_config: value[7..].to_string(),
            }),
            _ => Err(format!(
                "unknown strategy {value:?}; use heuristic, random, mcts, or python:<worker-config>"
            )),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Heuristic => "heuristic".into(),
            Self::Random => "random".into(),
            Self::Mcts { simulations } => format!("mcts({simulations})"),
            Self::Python { worker_config } => format!("python:{worker_config}"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ActionDto {
    pub canonical: String,
    pub description: String,
    pub action: String,
    pub selected: bool,
    pub evaluated: bool,
    pub score: Option<f64>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionTrace {
    pub strategy: String,
    pub selected: String,
    pub evidence_kind: String,
    pub candidates: Vec<ActionDto>,
    pub card_keep_scores: Vec<CardKeepDto>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CardKeepDto {
    pub index: usize,
    pub card: String,
    pub score: f64,
}

/// Policy seam for native policies and the future line-oriented Python worker.
pub trait StrategyAdapter: Send {
    fn name(&self) -> String;
    fn choose(
        &mut self,
        state: &mut GameState,
        legal: &[Move],
    ) -> Result<(Move, DecisionTrace), String>;
}

pub struct NativeStrategy {
    spec: StrategySpec,
    random: ChaCha12Rng,
}
impl NativeStrategy {
    pub fn new(spec: StrategySpec, seed: u64) -> Self {
        Self {
            spec,
            random: ChaCha12Rng::seed_from_u64(seed),
        }
    }
}

fn card_scores(state: &GameState) -> Vec<CardKeepDto> {
    let pid = state.current_player_id();
    heuristic_ai::ranked_card_choices(state, pid)
        .into_iter()
        .map(|(index, score)| CardKeepDto {
            index,
            card: state.players[pid]
                .hand
                .get(index)
                .map(Card::name)
                .unwrap_or_else(|| "?".into()),
            score,
        })
        .collect()
}

fn trace_for(
    state: &GameState,
    legal: &[Move],
    selected: Move,
    strategy: String,
    kind: &str,
    scored: Vec<(Move, f64)>,
) -> DecisionTrace {
    let selected_key = move_codec::encode(&selected);
    let scores: HashMap<String, f64> = scored
        .into_iter()
        .map(|(mv, score)| (move_codec::encode(&mv), score))
        .collect();
    let candidates = legal
        .iter()
        .map(|mv| {
            let canonical = move_codec::encode(mv);
            let score = scores.get(&canonical).copied();
            ActionDto {
                description: mv.describe(state),
                action: mv.action().to_string(),
                selected: canonical == selected_key,
                evaluated: score.is_some(),
                note: if score.is_none() {
                    Some("未被此策略评分".into())
                } else {
                    None
                },
                canonical,
                score,
            }
        })
        .collect();
    DecisionTrace {
        strategy,
        selected: selected_key,
        evidence_kind: kind.into(),
        candidates,
        card_keep_scores: card_scores(state),
    }
}

impl StrategyAdapter for NativeStrategy {
    fn name(&self) -> String {
        self.spec.label()
    }
    fn choose(
        &mut self,
        state: &mut GameState,
        legal: &[Move],
    ) -> Result<(Move, DecisionTrace), String> {
        if legal.is_empty() {
            return Err("current player has no legal move".into());
        }
        match self.spec.clone() {
            StrategySpec::Heuristic => {
                let scored = heuristic_ai::candidate_actions_k(state, 30);
                let decision = heuristic_ai::choose_action(state);
                let trace = trace_for(
                    state,
                    legal,
                    decision.mv.clone(),
                    self.name(),
                    "heuristic",
                    scored.into_iter().map(|d| (d.mv, d.score)).collect(),
                );
                Ok((decision.mv, trace))
            }
            StrategySpec::Random => {
                let mv = random_ai::choose_random_move(state, &mut self.random)
                    .ok_or("random policy found no move")?;
                let trace = trace_for(state, legal, mv.clone(), self.name(), "random", Vec::new());
                Ok((mv, trace))
            }
            StrategySpec::Mcts { simulations } => {
                let cfg = MctsConfig {
                    simulations,
                    ..Default::default()
                };
                let scored = heuristic_ai::candidate_actions_k(state, cfg.k_candidates);
                let decision = mcts_ai::choose_action_mcts(state, &cfg);
                let trace = trace_for(
                    state,
                    legal,
                    decision.mv.clone(),
                    self.name(),
                    "mcts-root-prior",
                    scored.into_iter().map(|d| (d.mv, d.score)).collect(),
                );
                Ok((decision.mv, trace))
            }
            StrategySpec::Python { worker_config } => Err(format!(
                "python worker {worker_config:?} is not configured; this replay session stopped before executing a move"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PlayerDto {
    pub id: usize,
    pub money: i32,
    pub income: i8,
    pub vp: u16,
    pub hand: Vec<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct TileDto {
    pub place: String,
    pub owner: usize,
    pub industry: String,
    pub level: u8,
    pub flipped: bool,
    pub resources: u8,
}
#[derive(Debug, Clone, Serialize)]
pub struct LinkDto {
    pub id: usize,
    pub owner: usize,
    pub kind: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct CityDto {
    pub id: usize,
    pub name: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct RouteDto {
    pub id: usize,
    pub a: usize,
    pub b: usize,
    pub via_farm: Option<usize>,
    pub canal: bool,
    pub rail: bool,
    pub owner: Option<usize>,
    pub kind: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct StateDto {
    pub era: String,
    pub round: u32,
    pub current_player: usize,
    pub turn_order: Vec<usize>,
    pub coal_market: usize,
    pub iron_market: usize,
    pub deck: usize,
    pub discard: usize,
    pub players: Vec<PlayerDto>,
    pub tiles: Vec<TileDto>,
    pub links: Vec<LinkDto>,
    pub cities: Vec<CityDto>,
    pub routes: Vec<RouteDto>,
}

pub fn state_dto(state: &GameState) -> StateDto {
    let players = state
        .players
        .iter()
        .enumerate()
        .map(|(id, p)| PlayerDto {
            id,
            money: p.money,
            income: p.income_level(),
            vp: p.vp,
            hand: p.hand.iter().map(Card::name).collect(),
        })
        .collect();
    let mut tiles: Vec<TileDto> = state
        .city_tiles
        .iter()
        .enumerate()
        .filter_map(|(key, t)| {
            t.as_ref().map(|tile| {
                let (loc, slot) = loc_from_key(key).expect("city tile key");
                TileDto {
                    place: format!("{} #{}", loc.name(), slot + 1),
                    owner: tile.player,
                    industry: tile.ind.name().into(),
                    level: tile.def.level,
                    flipped: tile.flipped,
                    resources: tile.resource_cubes,
                }
            })
        })
        .collect();
    for (index, tile) in state.farm_tiles.iter().enumerate() {
        if let Some(tile) = tile {
            tiles.push(TileDto {
                place: format!("Farm #{}", index + 1),
                owner: tile.player,
                industry: tile.ind.name().into(),
                level: tile.def.level,
                flipped: tile.flipped,
                resources: tile.resource_cubes,
            });
        }
    }
    let links = state
        .links
        .iter()
        .enumerate()
        .filter_map(|(id, link)| {
            link.as_ref().map(|link| LinkDto {
                id,
                owner: link.player,
                kind: if link.is_canal {
                    "canal".into()
                } else {
                    "rail".into()
                },
            })
        })
        .collect();
    let cities = ALL_LOCATIONS
        .iter()
        .take(CITY_COUNT)
        .enumerate()
        .map(|(id, loc)| CityDto {
            id,
            name: loc.name().into(),
        })
        .collect();
    let routes = connections()
        .iter()
        .map(|conn| {
            let link = state.links.get(conn.id).and_then(|x| x.as_ref());
            RouteDto {
                id: conn.id,
                a: conn.a as usize,
                b: conn.b as usize,
                via_farm: conn.via_farm.map(|x| x as usize),
                canal: conn.canal,
                rail: conn.rail,
                owner: link.map(|x| x.player),
                kind: link.map(|x| {
                    if x.is_canal {
                        "canal".into()
                    } else {
                        "rail".into()
                    }
                }),
            }
        })
        .collect();
    StateDto {
        era: match state.era {
            Era::Canal => "canal",
            Era::Rail => "rail",
        }
        .into(),
        round: state.round,
        current_player: state.current_player_id(),
        turn_order: state.turn_order.clone(),
        coal_market: state.coal_market,
        iron_market: state.iron_market,
        deck: state.deck.len(),
        discard: state.discard_pile.len(),
        players,
        tiles,
        links,
        cities,
        routes,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayStep {
    pub index: usize,
    pub player: usize,
    pub before: StateDto,
    pub after: StateDto,
    pub legal_actions: Vec<ActionDto>,
    pub chosen: String,
    pub result: String,
    pub era_event: Option<String>,
    pub trace: DecisionTrace,
}

pub struct ReplaySession {
    pub seed: u64,
    pub strategies: Vec<String>,
    state: GameState,
    adapters: Vec<Box<dyn StrategyAdapter>>,
    pub steps: Vec<ReplayStep>,
    snapshots: Vec<Vec<u8>>,
    pub failure: Option<String>,
}
impl ReplaySession {
    pub fn new(seed: u64, players: usize, specs: Vec<StrategySpec>) -> Result<Self, String> {
        if !(2..=4).contains(&players) {
            return Err("players must be between 2 and 4".into());
        }
        if specs.len() != players {
            return Err(format!(
                "expected {players} --player values, got {}",
                specs.len()
            ));
        }
        let strategies = specs.iter().map(StrategySpec::label).collect();
        let adapters = specs
            .into_iter()
            .enumerate()
            .map(|(seat, spec)| {
                Box::new(NativeStrategy::new(spec, seed ^ (seat as u64 + 1)))
                    as Box<dyn StrategyAdapter>
            })
            .collect();
        let state = GameState::new(ChaCha12Rng::seed_from_u64(seed), players);
        let snapshots = vec![state.snapshot_bytes()?];
        Ok(Self {
            seed,
            strategies,
            state,
            adapters,
            steps: Vec::new(),
            snapshots,
            failure: None,
        })
    }
    pub fn current_state(&self) -> StateDto {
        state_dto(&self.state)
    }
    pub fn is_complete(&self) -> bool {
        self.state.game_over
    }
    pub fn step(&mut self) -> Result<bool, String> {
        if self.failure.is_some() || self.state.game_over {
            return Ok(false);
        }
        let before = state_dto(&self.state);
        let player = self.state.current_player_id();
        let legal = legal_moves(&mut self.state);
        let (mv, trace) = match self.adapters[player].choose(&mut self.state, &legal) {
            Ok(v) => v,
            Err(e) => {
                self.failure = Some(e.clone());
                return Err(e);
            }
        };
        let selected = move_codec::encode(&mv);
        if !legal
            .iter()
            .any(|candidate| move_codec::encode(candidate) == selected)
        {
            let e = format!(
                "{} returned a non-legal action",
                self.adapters[player].name()
            );
            self.failure = Some(e.clone());
            return Err(e);
        }
        let legal_actions = trace.candidates.clone();
        let result = apply_move(&mut self.state, &mv).map_err(|e| {
            self.failure = Some(e.clone());
            e
        })?;
        let tr = advance_turn(&mut self.state);
        let era_event = match tr {
            TurnResult::Continue => None,
            TurnResult::EndCanalEra => {
                handle_turn_result(&mut self.state, tr);
                Some("canal-settlement-and-cleanup".into())
            }
            TurnResult::EndGame => {
                handle_turn_result(&mut self.state, tr);
                Some("final-scoring".into())
            }
        };
        let after = state_dto(&self.state);
        self.steps.push(ReplayStep {
            index: self.steps.len(),
            player,
            before,
            after,
            legal_actions,
            chosen: selected,
            result,
            era_event,
            trace,
        });
        self.snapshots.push(self.state.snapshot_bytes()?);
        Ok(true)
    }
    pub fn state_at(&self, index: usize) -> Result<StateDto, String> {
        let bytes = self
            .snapshots
            .get(index)
            .ok_or("step snapshot does not exist")?;
        Ok(state_dto(&GameState::from_snapshot_bytes(bytes)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saved_legal_actions_execute_on_each_before_snapshot() {
        let mut s =
            ReplaySession::new(7, 2, vec![StrategySpec::Heuristic, StrategySpec::Random]).unwrap();
        for _ in 0..8 {
            s.step().unwrap();
        }
        for step in &s.steps {
            for action in &step.legal_actions {
                let mut state = GameState::from_snapshot_bytes(&s.snapshots[step.index]).unwrap();
                // `legal_moves` is the engine's cache-rebuild boundary and is
                // also the exact source of the recorded action collection.
                legal_moves(&mut state);
                let mv = move_codec::decode(&action.canonical).unwrap();
                assert!(apply_move(&mut state, &mv).is_ok(), "{}", action.canonical);
            }
            assert!(
                step.legal_actions
                    .iter()
                    .any(|x| x.canonical == step.chosen)
            );
        }
    }
}
