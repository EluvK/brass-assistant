use _engine::data::{Era, IndustryType, industry_tiles};
use _engine::engine::{TurnResult, advance_turn, handle_turn_result, step};
use _engine::map::*;
use _engine::rules::{
    ResolvedMove, apply_move, execute_build, execute_network, execute_network_double,
    get_valid_build_targets, get_valid_network_targets, get_valid_second_rail_links,
    iron_source_options, legal_resolved_moves,
};
use _engine::scoring;
use _engine::state::{BoardTile, Card, GameState};
use rand_chacha::ChaCha12Rng;
use rand_chacha::rand_core::SeedableRng;

fn setup(players: usize) -> GameState {
    let rng = ChaCha12Rng::seed_from_u64(42);
    GameState::new(rng, players)
}

#[test]
fn default_heuristic_entry_point_is_the_search_policy() {
    let mut heuristic_state = setup(4);
    let heuristic = _engine::heuristic_ai::choose_action(&mut heuristic_state);
    assert!(heuristic.score.is_finite());
    assert!(
        legal_resolved_moves(&mut heuristic_state)
            .iter()
            .any(|mv| format!("{:?}", mv) == format!("{:?}", heuristic.mv))
    );
}

#[test]
fn heuristic_candidates_are_legal_and_never_dead_end() {
    for seed in 0..20u64 {
        let mut state = GameState::new(ChaCha12Rng::seed_from_u64(seed), 4);
        for _ in 0..120 {
            if state.game_over {
                break;
            }
            let legal = legal_resolved_moves(&mut state);
            let candidates = _engine::heuristic_ai::candidate_actions_k(&mut state, 3);
            if !legal.is_empty() {
                assert!(!candidates.is_empty(), "seed={seed} produced no candidate");
                for candidate in candidates {
                    let mut candidate_state = state.clone();
                    if apply_move(&mut candidate_state, &candidate.mv).is_err() {
                        assert!(
                            false,
                            "seed={seed} heuristic emitted an illegal candidate: {:?}, {}, {}, {:?}, {}, {:?}",
                            candidate.mv,
                            state.era,
                            state.round,
                            state.turn_order,
                            state.current_index,
                            apply_move(&mut candidate_state, &candidate.mv)
                        );
                    }
                }
            }
            let decision = _engine::heuristic_ai::choose_action(&mut state);
            let mut decision_state = state.clone();
            assert!(apply_move(&mut decision_state, &decision.mv).is_ok());
            apply_move(&mut state, &decision.mv).expect("heuristic decision must apply");
            let tr = advance_turn(&mut state);
            handle_turn_result(&mut state, tr);
        }
    }
}

fn setup_clean_rail_state(players: usize) -> GameState {
    let mut state = setup(players);
    state.era = Era::Rail;
    state.round = 1;
    state.turn_order = (0..players).collect();
    state.current_index = 0;
    state.actions_this_turn = 0;
    state.actions_per_turn = 2;
    state.is_first_round = false;
    state.game_over = false;
    state.city_tiles.iter_mut().for_each(|slot| *slot = None);
    state.farm_tiles = [None, None];
    state.links.iter_mut().for_each(|link| *link = None);
    state.merchants.clear();
    state.discard_pile.clear();
    for player in &mut state.players {
        player.money = 100;
        player.hand = vec![Card::WildLocation];
        player.canal_links = LINKS_PER_PLAYER;
        player.rail_links = LINKS_PER_PLAYER;
    }
    // Direct board clears above bypass the cache hooks; resync.
    state.rebuild_free_sources();
    state
}

fn place_test_coal_mine(state: &mut GameState, owner: usize) {
    state.place_tile(
        Loc::Cannock,
        0,
        BoardTile {
            player: owner,
            ind: IndustryType::CoalMine,
            def: industry_tiles(IndustryType::CoalMine)[0],
            flipped: false,
            resource_cubes: 2,
        },
    );
}

fn place_test_brewery(state: &mut GameState, loc: Loc, owner: usize, cubes: u8) {
    state.place_tile(
        loc,
        0,
        BoardTile {
            player: owner,
            ind: IndustryType::Brewery,
            def: industry_tiles(IndustryType::Brewery)[1],
            flipped: false,
            resource_cubes: cubes,
        },
    );
}

fn test_coal_from_cannock(state: &GameState) -> _engine::graph::CoalSource {
    let key = state
        .city_slot_key(Loc::Cannock, 0)
        .expect("cannock coal slot");
    _engine::graph::CoalSource {
        kind: _engine::graph::CoalSourceKind::Mine,
        key,
        price: 0,
        free: true,
    }
}

fn place_test_presence_tile(state: &mut GameState, loc: Loc, owner: usize) {
    state.place_tile(
        loc,
        0,
        BoardTile {
            player: owner,
            ind: IndustryType::Manufacturer,
            def: industry_tiles(IndustryType::Manufacturer)[0],
            flipped: false,
            resource_cubes: 0,
        },
    );
}

#[test]
fn overbuilding_opponent_tile_updates_old_owner_network_cache() {
    let mut state = setup_clean_rail_state(2);
    place_test_coal_mine(&mut state, 1);

    state.place_tile(
        Loc::Cannock,
        0,
        BoardTile {
            player: 0,
            ind: IndustryType::CoalMine,
            def: industry_tiles(IndustryType::CoalMine)[1],
            flipped: false,
            resource_cubes: 2,
        },
    );

    // This full rescan would catch Cannock remaining in P2's cached network.
    state.assert_caches_consistent();
}

#[test]
fn initial_setup_invariants() {
    let state = setup(4);
    assert_eq!(state.players.len(), 4);
    for p in &state.players {
        assert_eq!(p.hand.len(), HAND_SIZE);
        assert_eq!(p.money, INITIAL_MONEY);
        assert_eq!(p.income_space, INITIAL_INCOME_SPACE);
        assert_eq!(p.income_level(), 0);
        assert_eq!(p.canal_links, LINKS_PER_PLAYER);
        assert_eq!(p.rail_links, LINKS_PER_PLAYER);
        assert_eq!(p.vp, 0);
    }
    assert_eq!(state.coal_market, COAL_MARKET_INITIAL);
    assert_eq!(state.iron_market, IRON_MARKET_INITIAL);
    // Deck had 4 players * 8 = 32 cards dealt + 4 seeds burned.
    // Full 4p deck: locations 43? + industries 15 + dual 8. Sanity: deck non-empty.
    assert!(state.deck.len() > 0);
    // Merchant slots: 2p+3p+4p mix = 1+2+2+2+2=9 slots active at 4 players
    assert_eq!(state.merchants.len(), 9);
}

#[test]
fn merchant_slot_counts() {
    let s2 = setup(2);
    // 2p active merchants: shrewsbury(1)+gloucester(2)+oxford(2) = 5 slots
    assert_eq!(s2.merchants.len(), 5);
    let s3 = setup(3);
    // adds warrington(2) -> 7
    assert_eq!(s3.merchants.len(), 7);
    let s4 = setup(4);
    // adds nottingham(2) -> 9
    assert_eq!(s4.merchants.len(), 9);
}

#[test]
fn merchant_tile_composition_matches_rulebook() {
    // User-verified composition (see map.rs):
    //   2p: Blank x2, Any x1, Cotton x1, Manufacturer x1
    //   3p: Blank x3, Any x1, Cotton x1, Manufacturer x1, Pottery x1
    //   4p: Blank x3, Any x1, Cotton x2, Manufacturer x2, Pottery x1
    fn count_mix(p: usize) -> (usize, usize, usize, usize, usize) {
        let mut blanks = 0;
        let mut any = 0;
        let mut cotton = 0;
        let mut mfr = 0;
        let mut pottery = 0;
        for e in _engine::map::merchant_tile_mix(p) {
            match e {
                _engine::map::MerchantMixEntry::Blank => blanks += 1,
                _engine::map::MerchantMixEntry::Any => any += 1,
                _engine::map::MerchantMixEntry::Buys(t) => match t {
                    IndustryType::CottonMill => cotton += 1,
                    IndustryType::Manufacturer => mfr += 1,
                    IndustryType::Pottery => pottery += 1,
                    _ => {}
                },
            }
        }
        (blanks, any, cotton, mfr, pottery)
    }
    assert_eq!(count_mix(2), (2, 1, 1, 1, 0));
    assert_eq!(count_mix(3), (3, 1, 1, 1, 1));
    assert_eq!(count_mix(4), (3, 1, 2, 2, 1));
}

#[test]
fn every_4p_merchant_slot_gets_a_tile() {
    let s4 = setup(4);
    // All 9 active slots should be filled (no leftover 'Blank' from short mix).
    assert_eq!(s4.merchants.len(), 9);
    // Exactly one 'Any' tile across the 9 slots.
    let any_count = s4
        .merchants
        .iter()
        .filter(|m| m.buys == _engine::state::BuyType::Any)
        .count();
    assert_eq!(any_count, 1);
}

#[test]
fn build_targets_exist_early() {
    let state = setup(4);
    // With 8 random location cards, at least some build should be possible.
    let pid = state.current_player_id();
    let targets = get_valid_build_targets(&state, pid);
    // In canal era every location card is a legal build (empty board, money=17)
    // though some tiles cost more than 17. Assert at least one exists.
    assert!(
        targets.len() > 0,
        "expected some build targets, got {}",
        targets.len()
    );
}

#[test]
fn build_target_respects_money() {
    let state = setup(4);
    let pid = state.current_player_id();
    for t in get_valid_build_targets(&state, pid) {
        assert!(t.cost_total <= state.players[pid].money);
    }
}

#[test]
fn build_targets_include_multi_icon_slots_when_single_icon_slot_exists_elsewhere() {
    let mut state = setup(4);
    let pid = state.current_player_id();
    state.players[pid].money = 100;
    state.players[pid].hand = vec![Card::WildIndustry];

    let targets = get_valid_build_targets(&state, pid);

    // Worcester has an empty CottonMill-only slot, but that must not suppress
    // the legal CottonMill build in Derby's CottonMill/Brewery slot.
    assert!(targets.iter().any(|target| {
        target.loc == Loc::Derby && target.slot_index == 0 && target.ind == IndustryType::CottonMill
    }));
}

#[test]
fn network_targets_are_canal_or_rail_only() {
    let state = setup(4);
    let pid = state.current_player_id();
    for conn_id in get_valid_network_targets(&state, pid) {
        let conn = &connections()[conn_id];
        match state.era {
            Era::Canal => assert!(conn.canal, "canal era target must be canal-enabled"),
            Era::Rail => assert!(conn.rail, "rail era target must be rail-enabled"),
        }
        assert!(state.links[conn_id].is_none());
    }
}

#[test]
fn build_changes_board_and_discards_card() {
    let mut state = setup(4);
    let pid = state.current_player_id();
    let targets = get_valid_build_targets(&state, pid);
    assert!(targets.len() > 0);
    let t = targets[0].clone();
    let hand_before = state.players[pid].hand.len();

    // Find a valid card for this target
    let card_idx = state.players[pid]
        .hand
        .iter()
        .position(|c| match c {
            _engine::state::Card::Location(l) => *l == t.loc && !t.loc.is_farm(),
            _engine::state::Card::Industry { .. } => {
                c.is_industry(t.ind)
                    && (_engine::graph::is_in_network(&state, pid, t.loc)
                        || !_engine::graph::player_has_presence(&state, pid))
            }
            _engine::state::Card::WildLocation => !t.loc.is_farm(),
            _engine::state::Card::WildIndustry => true,
        })
        .expect("valid build target must have a matching card");

    let coal = _engine::rules::coal_source_options(&state, t.loc, t.cost_coal as usize)
        .into_iter()
        .next()
        .unwrap_or_default();
    let iron = _engine::rules::iron_source_options(&state, t.cost_iron as usize)
        .into_iter()
        .next()
        .unwrap_or_default();
    let res = execute_build(
        &mut state,
        pid,
        t.loc,
        t.slot_index,
        t.ind,
        &coal,
        &iron,
        card_idx,
    );
    assert!(res.is_ok(), "build failed: {:?}", res);
    assert_eq!(state.players[pid].hand.len(), hand_before - 1);
}

#[test]
fn execute_network_places_link() {
    let mut state = setup(4);
    let pid = state.current_player_id();
    let targets = get_valid_network_targets(&state, pid);
    assert!(targets.len() > 0, "no network targets at setup");
    let conn_id = targets[0];
    let res = execute_network(&mut state, pid, conn_id, None, 0);
    assert!(res.is_ok(), "network failed: {:?}", res);
    assert!(state.links[conn_id].is_some());
}

#[test]
fn raw_invalid_actions_are_rejected_without_state_changes() {
    let mut state = setup(4);
    let pid = state.current_player_id();
    let hand_before = state.players[pid].hand.len();
    let money_before = state.players[pid].money;

    assert!(execute_network(&mut state, pid, 0, None, hand_before).is_err());
    assert_eq!(state.players[pid].hand.len(), hand_before);
    assert_eq!(state.players[pid].money, money_before);
    assert!(state.links.iter().all(Option::is_none));

    // Slot 99 must not consume a tile, resources, money, or a card.
    let target = get_valid_build_targets(&state, pid)
        .into_iter()
        .next()
        .expect("expected an initial build target");
    let card =
        _engine::rules::valid_build_cards(&state, &state.players[pid], pid, target.loc, target.ind)
            [0];
    let coal = _engine::rules::coal_source_options(&state, target.loc, target.cost_coal as usize)
        .into_iter()
        .next()
        .unwrap_or_default();
    let iron = _engine::rules::iron_source_options(&state, target.cost_iron as usize)
        .into_iter()
        .next()
        .unwrap_or_default();
    assert!(
        execute_build(
            &mut state, pid, target.loc, 99, target.ind, &coal, &iron, card,
        )
        .is_err()
    );
    assert_eq!(state.players[pid].hand.len(), hand_before);
    assert_eq!(state.players[pid].money, money_before);
    assert!(state.links.iter().all(Option::is_none));
}

#[test]
fn raw_scout_and_double_rail_reject_malformed_indices() {
    let mut state = setup(4);
    let pid = state.current_player_id();
    let hand_before = state.players[pid].hand.clone();
    assert!(_engine::rules::execute_scout(&mut state, pid, [0, 0, 1]).is_err());
    assert_eq!(state.players[pid].hand, hand_before);

    let mut rail = setup_clean_rail_state(2);
    let rail_pid = 0;
    let invalid_coal = test_coal_from_cannock(&rail);
    assert!(
        execute_network_double(
            &mut rail,
            rail_pid,
            usize::MAX,
            0,
            invalid_coal,
            invalid_coal,
            _engine::graph::BeerSource {
                kind: _engine::graph::BeerSourceKind::Merchant,
                key: usize::MAX,
                farm_idx: None,
                merchant_idx: Some(usize::MAX),
            },
            0,
        )
        .is_err()
    );
    assert!(rail.links.iter().all(Option::is_none));
}

#[test]
fn failed_step_does_not_advance_turn() {
    let mut state = setup(4);
    let before = state.current_index;
    let invalid_card_index = state.players[state.current_player_id()].hand.len();
    let (result, turn) = step(
        &mut state,
        &ResolvedMove::Pass {
            card_index: invalid_card_index,
        },
    );
    assert!(result.is_err());
    assert!(matches!(turn, _engine::engine::TurnResult::Continue));
    assert_eq!(state.current_index, before);
}

#[test]
fn scoring_produces_nonzero_total() {
    // Manually place a flipped tile and verify score_era picks it up.
    let mut state = setup(4);
    let pid = state.current_player_id();

    // Give the player an affordable cotton mill and place+flip it directly.
    let def = industry_tiles(IndustryType::CottonMill)[0]; // Lv1, cost 12
    state.place_tile(
        Loc::Birmingham,
        0,
        _engine::state::BoardTile {
            player: pid,
            ind: IndustryType::CottonMill,
            def,
            flipped: true,
            resource_cubes: 0,
        },
    );
    let scores = scoring::score_era(&mut state);
    let mine = scores.iter().find(|s| s.player_id == pid).unwrap();
    assert!(mine.industry_vp >= 5, "cotton mill lv1 should score 5 VP");
}

#[test]
fn link_scoring_ignores_unflipped_tiles_until_they_flip() {
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    state.players[pid].vp = 0;

    state.place_tile(
        Loc::Belper,
        0,
        BoardTile {
            player: pid,
            ind: IndustryType::Manufacturer,
            def: industry_tiles(IndustryType::Manufacturer)[0],
            flipped: false,
            resource_cubes: 0,
        },
    );
    state.place_tile(
        Loc::Derby,
        0,
        BoardTile {
            player: pid,
            ind: IndustryType::Brewery,
            def: industry_tiles(IndustryType::Brewery)[0],
            flipped: true,
            resource_cubes: 0,
        },
    );
    state.links[0] = Some(_engine::state::Link {
        player: pid,
        is_canal: true,
    });

    let scores = scoring::score_era(&mut state);
    let mine = scores.iter().find(|s| s.player_id == pid).unwrap();
    assert_eq!(
        mine.link_vp, 2,
        "only the flipped Derby brewery should contribute link icons"
    );
    assert_eq!(
        mine.industry_vp, 4,
        "only flipped industries should score industry VP"
    );

    state.players[pid].vp = 0;
    let belper_key = state.city_slot_key(Loc::Belper, 0).unwrap();
    state.city_tiles[belper_key].as_mut().unwrap().flipped = true;

    let scores = scoring::score_era(&mut state);
    let mine = scores.iter().find(|s| s.player_id == pid).unwrap();
    assert_eq!(
        mine.link_vp, 4,
        "after Belper flips, both endpoint tiles should contribute link icons"
    );
}

#[test]
fn canal_era_end_removes_all_level1_tiles_including_pottery() {
    let mut state = setup(2);

    // Pottery I is the one level-1 tile that is also buildable in the rail era
    // (rail_era = true), but it must STILL be removed at the canal-era end:
    // cleanup is ruled by `level == 1`, not by the build-eligibility flags.
    let pottery_def = industry_tiles(IndustryType::Pottery)[0];
    assert_eq!(pottery_def.level, 1);
    assert!(pottery_def.rail_era, "Pottery I is a two-era build tile");

    state.place_tile(
        Loc::Birmingham,
        0,
        BoardTile {
            player: 0,
            ind: IndustryType::Pottery,
            def: pottery_def,
            flipped: false,
            resource_cubes: 0,
        },
    );
    state.place_tile(
        Loc::Derby,
        0,
        BoardTile {
            player: 0,
            ind: IndustryType::CottonMill,
            def: industry_tiles(IndustryType::CottonMill)[0],
            flipped: false,
            resource_cubes: 0,
        },
    );
    // Level-2+ tiles survive into the rail era.
    state.place_tile(
        Loc::Belper,
        0,
        BoardTile {
            player: 0,
            ind: IndustryType::Brewery,
            def: industry_tiles(IndustryType::Brewery)[1],
            flipped: false,
            resource_cubes: 1,
        },
    );

    handle_turn_result(&mut state, TurnResult::EndCanalEra);

    let bham = state.city_slot_key(Loc::Birmingham, 0).unwrap();
    assert!(
        state.city_tiles[bham].is_none(),
        "Pottery I (level 1, rail_era=true) must be removed at canal-era end"
    );
    let derby = state.city_slot_key(Loc::Derby, 0).unwrap();
    assert!(
        state.city_tiles[derby].is_none(),
        "Cotton Mill I must be removed at canal-era end"
    );
    let belper = state.city_slot_key(Loc::Belper, 0).unwrap();
    assert!(
        state.city_tiles[belper].is_some(),
        "Brewery II (level 2) must survive into the rail era"
    );
    assert_eq!(state.city_tiles[belper].as_ref().unwrap().def.level, 2);
}

#[test]
fn canal_era_rejects_brewery4_and_pottery5() {
    let mut state = setup(2);
    let pid = 0;

    // Advance both stacks to their top (rail-only) tile.
    let b_stack = _engine::state::player_industry_stack(IndustryType::Brewery);
    let p_stack = _engine::state::player_industry_stack(IndustryType::Pottery);
    state.players[pid].industry_next[IndustryType::Brewery as usize] = (b_stack.len() - 1) as u8;
    state.players[pid].industry_next[IndustryType::Pottery as usize] = (p_stack.len() - 1) as u8;

    let b4 = industry_tiles(IndustryType::Brewery)[3];
    let p5 = industry_tiles(IndustryType::Pottery)[4];
    assert_eq!(b4.level, 4);
    assert_eq!(p5.level, 5);
    assert!(
        !b4.canal_era && b4.rail_era,
        "Brewery IV is a rail-era-only build"
    );
    assert!(
        !p5.canal_era && p5.rail_era,
        "Pottery V is a rail-era-only build"
    );

    // 1) ResolvedMove generation must exclude them in the canal era.
    let targets = get_valid_build_targets(&state, pid);
    assert!(
        !targets.iter().any(|t| t.ind == IndustryType::Brewery),
        "Brewery IV must not be a canal-era build target"
    );
    assert!(
        !targets.iter().any(|t| t.ind == IndustryType::Pottery),
        "Pottery V must not be a canal-era build target"
    );

    // 2) Raw execution must also reject them (defense in depth).
    let iron = _engine::rules::iron_source_options(&state, 1)
        .into_iter()
        .next()
        .unwrap_or_default();
    let res = execute_build(
        &mut state,
        pid,
        Loc::BreweryNorth,
        0,
        IndustryType::Brewery,
        &[],
        &iron,
        0,
    );
    assert!(
        res.is_err(),
        "raw Brewery IV build must fail in the canal era: {res:?}"
    );

    // 3) In the rail era the same tiles ARE valid build targets.
    state.era = Era::Rail;
    state.players[pid].money = 100;
    state.players[pid].hand = vec![Card::WildIndustry];
    // Give the player a network so industry-card builds are legal: a coal mine
    // at Cannock plus links to the northern brewery farm (conn 16) and to
    // Stafford (conn 15), whose second slot is a pure Pottery slot. The
    // Cannock mine also supplies the 2 coal Pottery V needs.
    state.place_tile(
        Loc::Cannock,
        0,
        BoardTile {
            player: pid,
            ind: IndustryType::CoalMine,
            def: industry_tiles(IndustryType::CoalMine)[0],
            flipped: false,
            resource_cubes: 2,
        },
    );
    state.links[16] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });
    state.links[15] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });

    let targets = get_valid_build_targets(&state, pid);
    assert!(
        targets.iter().any(|t| t.ind == IndustryType::Brewery),
        "Brewery IV must be a rail-era build target"
    );
    assert!(
        targets.iter().any(|t| t.ind == IndustryType::Pottery),
        "Pottery V must be a rail-era build target"
    );
}

#[test]
fn canal_era_one_build_per_city_per_player() {
    let mut state = setup(2);
    let pid = 0;
    state.players[pid].money = 100;
    state.players[pid].hand = vec![Card::WildLocation, Card::WildIndustry];

    // First canal-era build: Cotton Mill I at Derby slot 0 (no resources).
    let res = execute_build(
        &mut state,
        pid,
        Loc::Derby,
        0,
        IndustryType::CottonMill,
        &[],
        &[],
        0,
    );
    assert!(res.is_ok(), "first Derby build failed: {res:?}");
    let derby0 = state.city_slot_key(Loc::Derby, 0).unwrap();
    assert!(state.city_tiles[derby0].is_some());

    // Canal era: no further build target may exist in Derby for this player.
    let targets = get_valid_build_targets(&state, pid);
    assert!(
        !targets.iter().any(|t| t.loc == Loc::Derby),
        "canal era: a second build in the same city must not be a valid target"
    );

    // Raw execution must also reject the second Derby build.
    let res = execute_build(
        &mut state,
        pid,
        Loc::Derby,
        1,
        IndustryType::Manufacturer,
        &[],
        &[],
        0,
    );
    assert!(
        res.is_err(),
        "canal era: raw second Derby build must be rejected: {res:?}"
    );

    // Rail era: the restriction is lifted — a second tile in Derby is legal.
    state.era = Era::Rail;
    state.players[pid].industry_next[IndustryType::Manufacturer as usize] = 1; // Mfg II
    let iron = _engine::rules::iron_source_options(&state, 1)
        .into_iter()
        .next()
        .unwrap_or_default();
    let res = execute_build(
        &mut state,
        pid,
        Loc::Derby,
        1,
        IndustryType::Manufacturer,
        &[],
        &iron,
        0,
    );
    assert!(res.is_ok(), "rail era: second Derby build failed: {res:?}");
    let derby1 = state.city_slot_key(Loc::Derby, 1).unwrap();
    let tile = state.city_tiles[derby1]
        .as_ref()
        .expect("second Derby tile placed");
    assert_eq!(
        tile.def.level, 2,
        "Mfg II built at Derby slot 1 in the rail era"
    );
}

#[test]
fn industry_card_build_requires_network() {
    let mut state = setup(2);
    let pid = 0;
    state.players[pid].money = 100;
    // Presence + network = only Derby. Hand: a CottonMill industry card and a
    // wild location card.
    state.place_tile(
        Loc::Derby,
        0,
        BoardTile {
            player: pid,
            ind: IndustryType::CottonMill,
            def: industry_tiles(IndustryType::CottonMill)[0],
            flipped: false,
            resource_cubes: 0,
        },
    );
    state.players[pid].hand = vec![
        Card::Industry {
            industries: [IndustryType::CottonMill, IndustryType::CottonMill],
            n: 1,
        },
        Card::WildLocation,
    ];

    // The industry card must NOT permit a build in Birmingham (not in network).
    let valid = _engine::rules::valid_build_cards(
        &state,
        &state.players[pid],
        pid,
        Loc::Birmingham,
        IndustryType::CottonMill,
    );
    assert!(
        !valid.contains(&0),
        "industry card must be rejected for a city outside the network"
    );
    assert!(
        valid.contains(&1),
        "wild location card must be allowed anywhere"
    );
    let res = execute_build(
        &mut state,
        pid,
        Loc::Birmingham,
        0,
        IndustryType::CottonMill,
        &[],
        &[],
        0,
    );
    assert!(
        res.is_err(),
        "industry-card build outside the network must be rejected: {res:?}"
    );

    // The same build is legal when discarding the wild LOCATION card.
    let res = execute_build(
        &mut state,
        pid,
        Loc::Birmingham,
        0,
        IndustryType::CottonMill,
        &[],
        &[],
        1,
    );
    assert!(
        res.is_ok(),
        "location-card build outside the network failed: {res:?}"
    );
    let bham0 = state.city_slot_key(Loc::Birmingham, 0).unwrap();
    assert!(state.city_tiles[bham0].is_some(), "Birmingham tile placed");
}

#[test]
fn kidderminster_worcester_link_connects_southern_brewery_farm() {
    let mut state = setup(2);
    let pid = 0;
    state.players[pid].money = 100;
    state.players[pid].hand = vec![Card::Industry {
        industries: [IndustryType::Brewery, IndustryType::Brewery],
        n: 1,
    }];
    state.place_tile(
        Loc::Kidderminster,
        0,
        BoardTile {
            player: pid,
            ind: IndustryType::CoalMine,
            def: industry_tiles(IndustryType::CoalMine)[0],
            flipped: false,
            resource_cubes: 1,
        },
    );

    state.set_link(connection_via_farm(), pid);

    assert!(
        _engine::graph::is_in_network(&state, pid, Loc::BrewerySouth),
        "the Kidderminster-Worcester link must put the southern farm in the network"
    );
    let iron_sources = iron_source_options(&state, 1)
        .into_iter()
        .next()
        .expect("Brewery I needs an iron source");
    let result = execute_build(
        &mut state,
        pid,
        Loc::BrewerySouth,
        0,
        IndustryType::Brewery,
        &[],
        &iron_sources,
        0,
    );
    assert!(
        result.is_ok(),
        "the southern farm must be buildable through connection 29: {result:?}"
    );
}

#[test]
fn loan_lowers_income_by_three_levels() {
    let mut state = setup(4);
    let pid = state.current_player_id();
    assert!(state.can_take_loan(pid));
    let before = state.players[pid].income_level();
    let res = _engine::rules::execute_loan(&mut state, pid, 0);
    assert!(res.is_ok());
    let after = state.players[pid].income_level();
    assert_eq!(before - after, 3, "loan should drop income 3 levels");
    assert_eq!(state.players[pid].money, INITIAL_MONEY + LOAN_AMOUNT);
}

#[test]
fn discard_tracks_face_down_pile() {
    let mut state = setup(4);
    let pid = state.current_player_id();
    let before = state.discard_pile.len();
    // Discard a non-wild card: should enter the discard pile.
    let idx = state.players[pid]
        .hand
        .iter()
        .position(|c| {
            !matches!(
                c,
                _engine::state::Card::WildLocation | _engine::state::Card::WildIndustry
            )
        })
        .unwrap();
    _engine::rules::discard_card(&mut state, pid, idx);
    assert_eq!(
        state.discard_pile.len(),
        before + 1,
        "non-wild discard should be tracked"
    );
}

#[test]
fn deck_composition_matches_era_card_count() {
    // 4p composition: location + industry + dual cotton/mfr cards.
    let comp = _engine::state::deck_composition(4);
    let loc_cards: usize = _engine::map::location_cards(4)
        .iter()
        .map(|(_, c)| *c as usize)
        .sum();
    let ind_cards: usize = _engine::map::industry_cards(4)
        .iter()
        .map(|(_, c)| *c as usize)
        .sum();
    let dual = _engine::map::dual_cotton_manufacturer_cards(4) as usize;
    assert_eq!(comp.len(), loc_cards + ind_cards + dual);
}

#[test]
fn mcts_returns_a_legal_pass_fallback_on_empty_hand() {
    use _engine::mcts_ai::{self, MctsConfig};
    let mut state = setup(2);
    // Empty the current player's hand: only Pass (and no legal moves) remains.
    let pid = state.current_player_id();
    let hand = state.players[pid].hand.clone();
    for (i, c) in hand.iter().enumerate() {
        state.players[pid].hand[i] = c.clone();
    }
    let cfg = MctsConfig {
        simulations: 50,
        ..Default::default()
    };
    // Should not panic; returns a Decision.
    let d = mcts_ai::choose_action_mcts(&mut state, &cfg);
    assert!(d.score.is_finite());
}

#[test]
fn mcts_determinize_keeps_own_hand_and_hand_size() {
    use _engine::mcts_ai::MctsConfig;
    let state = setup(4);
    let pid = state.current_player_id();
    let own = state.players[pid].hand.clone();
    let mut rng = ChaCha12Rng::seed_from_u64(123);
    let det = _engine::mcts_ai::determinize_for_test(&state, &mut rng, &MctsConfig::default());
    // Our own hand is preserved.
    assert_eq!(det.players[pid].hand, own);
    // Opponent hands keep their size.
    for i in 0..4 {
        if i != pid {
            assert_eq!(det.players[i].hand.len(), state.players[i].hand.len());
        }
    }
}

/// The determinized world must be a consistent multiset: every non-wild card
/// of the era composition appears exactly once across hands + deck + discard
/// pile, and nothing else. This guards the discard-pile subtraction in
/// `mcts_ai::determinize` (a played card must never reappear in a hand/deck).
#[test]
fn mcts_determinize_pool_is_multiset_consistent() {
    use _engine::mcts_ai::MctsConfig;
    let mut state = setup(4);
    // Discard one card per player so the pool actually has out-of-circulation
    // cards to subtract.
    let idxs: Vec<usize> = (0..4)
        .map(|pid| {
            state.players[pid]
                .hand
                .iter()
                .position(|c| !matches!(c, Card::WildLocation | Card::WildIndustry))
                .unwrap()
        })
        .collect();
    for pid in 0..4 {
        _engine::rules::discard_card(&mut state, pid, idxs[pid]);
    }

    let mut rng = ChaCha12Rng::seed_from_u64(7);
    let det = _engine::mcts_ai::determinize_for_test(&state, &mut rng, &MctsConfig::default());
    let comp = _engine::state::deck_composition(state.player_count());
    let mut all: Vec<Card> = Vec::new();
    for p in &det.players {
        for c in &p.hand {
            if !matches!(*c, Card::WildLocation | Card::WildIndustry) {
                all.push(c.clone());
            }
        }
    }
    all.extend(det.deck.clone());
    all.extend(det.discard_pile.clone());
    assert_eq!(
        all.len(),
        comp.len(),
        "non-wild cards in play must match composition"
    );
    for c in &comp {
        assert_eq!(
            all.iter().filter(|x| *x == c).count(),
            comp.iter().filter(|x| *x == c).count(),
            "multiset mismatch for {c:?}"
        );
    }
}

/// A card the current player has already discarded is out of circulation: it
/// must not reappear in any opponent hand or the deck after determinization.
#[test]
fn determinize_excludes_discarded_cards_from_opponent_hands() {
    use _engine::mcts_ai::MctsConfig;
    let mut state = setup(4);
    let pid = state.current_player_id();
    let idx = state.players[pid]
        .hand
        .iter()
        .position(|c| !matches!(c, Card::WildLocation | Card::WildIndustry))
        .unwrap();
    let discarded = state.players[pid].hand[idx].clone();
    _engine::rules::discard_card(&mut state, pid, idx);
    assert!(
        state.discard_pile.contains(&discarded),
        "setup: card must enter the discard pile"
    );

    let mut rng = ChaCha12Rng::seed_from_u64(99);
    let det = _engine::mcts_ai::determinize_for_test(&state, &mut rng, &MctsConfig::default());
    for (i, p) in det.players.iter().enumerate() {
        if i != pid {
            assert!(
                !p.hand.contains(&discarded),
                "discarded card {discarded:?} leaked into P{i}'s hand"
            );
        }
    }
    assert!(
        !det.deck.contains(&discarded),
        "discarded card {discarded:?} leaked into the deck"
    );
    assert!(
        det.discard_pile.contains(&discarded),
        "discard pile must keep the card"
    );
}

/// Era transition re-enters every canal-era card into the rail deck, so both
/// the anonymous discard pile and the per-player played history must reset.
#[test]
fn end_canal_era_resets_discard_pile_and_played() {
    let mut state = setup(4);
    let idxs: Vec<usize> = (0..4)
        .map(|pid| {
            state.players[pid]
                .hand
                .iter()
                .position(|c| !matches!(c, Card::WildLocation | Card::WildIndustry))
                .unwrap()
        })
        .collect();
    for pid in 0..4 {
        _engine::rules::discard_card(&mut state, pid, idxs[pid]);
    }
    assert!(state.discard_pile.len() >= 4);
    assert!(state.players.iter().any(|p| !p.played.is_empty()));

    handle_turn_result(&mut state, TurnResult::EndCanalEra);
    assert!(
        state.discard_pile.is_empty(),
        "discard pile must reset at era end"
    );
    for p in &state.players {
        assert!(p.played.is_empty(), "played history must reset at era end");
    }
}

/// Every hand-card consumption records the card in the player's `played`
/// history (only non-wild discards; the wilds *gained* by scouting go into
/// the hand, not the played history).
#[test]
fn discard_and_scout_record_per_player_played() {
    let mut state = setup(4);
    let pid = state.current_player_id();

    let idx = state.players[pid]
        .hand
        .iter()
        .position(|c| !matches!(c, Card::WildLocation | Card::WildIndustry))
        .unwrap();
    let discarded = state.players[pid].hand[idx].clone();
    _engine::rules::discard_card(&mut state, pid, idx);
    assert_eq!(state.players[pid].played.len(), 1);
    assert_eq!(state.players[pid].played[0], discarded);

    let three: [usize; 3] = [0, 1, 2];
    let res = _engine::rules::execute_scout(&mut state, pid, three);
    assert!(res.is_ok(), "scout failed: {res:?}");
    let p = &state.players[pid];
    assert_eq!(p.played.len(), 1 + 3, "1 discard + 3 scout cards recorded");
    // The two wilds gained by scouting enter the hand, not the played history.
    assert!(p.hand.contains(&Card::WildLocation));
    assert!(p.hand.contains(&Card::WildIndustry));
}

/// Wild cards return to the supply on use: they enter NEITHER the discard pile
/// NOR the player's played history, and the public holding flags flip to false.
#[test]
fn discarding_a_wild_returns_it_to_supply_and_skips_played() {
    let mut state = setup(4);
    let pid = state.current_player_id();

    // Scout to gain both wilds.
    let res = _engine::rules::execute_scout(&mut state, pid, [0, 1, 2]);
    assert!(res.is_ok(), "scout failed: {res:?}");
    assert!(state.players[pid].has_wild_location);
    assert!(state.players[pid].has_wild_industry);
    let before_played = state.players[pid].played.len(); // the 3 scout cards
    let loc_pile_before = state.wild_location_pile;
    let discard_before = state.discard_pile.len();

    let wl_idx = state.players[pid]
        .hand
        .iter()
        .position(|c| matches!(c, Card::WildLocation))
        .unwrap();
    _engine::rules::discard_card(&mut state, pid, wl_idx);

    assert_eq!(
        state.players[pid].played.len(),
        before_played,
        "wild must not enter played"
    );
    assert_eq!(
        state.discard_pile.len(),
        discard_before,
        "wild must not enter the discard pile"
    );
    assert_eq!(
        state.wild_location_pile,
        loc_pile_before + 1,
        "wild returns to the supply"
    );
    assert!(
        !state.players[pid].has_wild_location,
        "wild-location holding flag clears"
    );
    assert!(
        state.players[pid].has_wild_industry,
        "wild-industry holding flag unaffected"
    );
}

/// Holding a wild card is public information: the engine exposes it per player
/// and the training tensor encodes both flags separately in the global vector.
#[test]
fn wild_holding_is_public_and_encoded_in_training_tensor() {
    let mut state = setup(4);
    let pid = state.current_player_id();
    let base = 4 + pid * 17;
    let t = _engine::encode::state_to_tensor(&state, pid);
    assert_eq!(t.global[base + 8], 0.0, "no wilds held before scout");
    assert_eq!(t.global[base + 9], 0.0, "no wilds held before scout");

    let res = _engine::rules::execute_scout(&mut state, pid, [0, 1, 2]);
    assert!(res.is_ok(), "scout failed: {res:?}");
    let t = _engine::encode::state_to_tensor(&state, pid);
    assert_eq!(
        t.global[base + 8],
        1.0,
        "wild-location holding encoded separately"
    );
    assert_eq!(
        t.global[base + 9],
        1.0,
        "wild-industry holding encoded separately"
    );

    let wl_idx = state.players[pid]
        .hand
        .iter()
        .position(|c| matches!(c, Card::WildLocation))
        .unwrap();
    _engine::rules::discard_card(&mut state, pid, wl_idx);
    let t = _engine::encode::state_to_tensor(&state, pid);
    assert_eq!(
        t.global[base + 8],
        0.0,
        "wild-location flag reflects the discard"
    );
    assert_eq!(t.global[base + 9], 1.0, "wild-industry flag unaffected");
}

#[test]
fn legal_moves_do_not_offer_unaffordable_double_develop() {
    let mut state = setup(4);
    let pid = state.current_player_id();

    state.players[pid].money = 0;
    state.place_tile(
        Loc::Birmingham,
        2,
        BoardTile {
            player: (pid + 1) % 4,
            ind: IndustryType::IronWorks,
            def: industry_tiles(IndustryType::IronWorks)[0],
            flipped: false,
            resource_cubes: 1,
        },
    );

    let moves = legal_resolved_moves(&mut state);
    let single_develops = moves
        .iter()
        .filter(|mv| matches!(mv, _engine::rules::ResolvedMove::Develop { ind2: None, .. }))
        .count();
    let double_develops = moves
        .iter()
        .filter(|mv| {
            matches!(
                mv,
                _engine::rules::ResolvedMove::Develop { ind2: Some(_), .. }
            )
        })
        .count();

    assert!(
        single_develops > 0,
        "expected at least one single develop with one free iron"
    );
    assert_eq!(
        double_develops, 0,
        "double develop should not be generated when only one iron is affordable"
    );
}

#[test]
fn second_rail_links_allow_own_unconnected_beer() {
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    place_test_presence_tile(&mut state, Loc::Stafford, pid);
    place_test_coal_mine(&mut state, 1);
    place_test_brewery(&mut state, Loc::BrewerySouth, pid, 1);

    let second_links = get_valid_second_rail_links(&mut state, pid, 15);

    assert!(
        second_links.contains(&33),
        "own beer anywhere should allow the second rail link even without network connection"
    );
}

#[test]
fn second_rail_links_require_opponent_beer_to_be_connected() {
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    place_test_coal_mine(&mut state, 1);
    place_test_brewery(&mut state, Loc::BrewerySouth, 1, 1);

    let second_links = get_valid_second_rail_links(&mut state, pid, 15);

    assert!(
        !second_links.contains(&33),
        "opponent beer must be connected to the new rail network to count"
    );
}

#[test]
fn execute_network_double_consumes_two_coal_and_one_own_beer() {
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    place_test_presence_tile(&mut state, Loc::Stafford, pid);
    place_test_coal_mine(&mut state, 1);
    place_test_brewery(&mut state, Loc::BrewerySouth, pid, 1);

    let own_beer = _engine::graph::BeerSource {
        kind: _engine::graph::BeerSourceKind::Own,
        key: usize::MAX,
        farm_idx: Some(1),
        merchant_idx: None,
    };
    let coal1 = test_coal_from_cannock(&state);
    let coal2 = test_coal_from_cannock(&state);
    let res = execute_network_double(&mut state, pid, 15, 33, coal1, coal2, own_beer, 0);

    assert!(
        res.is_ok(),
        "double rail should succeed with 2 coal and 1 own beer available: {res:?}"
    );
    let coal_tile = state
        .tile_at(Loc::Cannock, 0)
        .expect("coal tile should remain on board");
    assert_eq!(
        coal_tile.resource_cubes, 0,
        "double rail should consume exactly 2 coal"
    );
    assert!(coal_tile.flipped, "depleted coal mine should flip");
    let own_brewery = state
        .farm_tile(Loc::BrewerySouth)
        .expect("own brewery should remain on board");
    assert_eq!(
        own_brewery.resource_cubes, 0,
        "double rail should consume exactly 1 beer"
    );
    assert!(own_brewery.flipped, "depleted brewery should flip");
    assert!(state.links[15].is_some());
    assert!(state.links[33].is_some());
    assert_eq!(state.players[pid].rail_links, LINKS_PER_PLAYER - 2);
    assert_eq!(
        state.players[pid].money, 85,
        "double rail with free coal should cost only £15"
    );
}

#[test]
fn execute_network_double_can_use_connected_opponent_beer() {
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    place_test_coal_mine(&mut state, 1);
    place_test_brewery(&mut state, Loc::Stone, 1, 1);

    let stone_key = state
        .city_slot_key(Loc::Stone, 0)
        .expect("stone slot should exist");
    let opp_beer = _engine::graph::BeerSource {
        kind: _engine::graph::BeerSourceKind::Opponent,
        key: stone_key,
        farm_idx: None,
        merchant_idx: None,
    };
    let coal1 = test_coal_from_cannock(&state);
    let coal2 = test_coal_from_cannock(&state);
    let res = execute_network_double(&mut state, pid, 15, 33, coal1, coal2, opp_beer, 0);

    assert!(
        res.is_ok(),
        "connected opponent beer should be usable for double rail: {res:?}"
    );
    let opp_brewery = state
        .tile_at(Loc::Stone, 0)
        .expect("opponent brewery should remain on board");
    assert_eq!(
        opp_brewery.resource_cubes, 0,
        "connected opponent beer should be consumed"
    );
    assert!(opp_brewery.flipped, "depleted opponent brewery should flip");
}

#[test]
fn execute_network_double_without_connected_beer_rolls_back_temp_changes() {
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    place_test_coal_mine(&mut state, 1);
    place_test_brewery(&mut state, Loc::BrewerySouth, 1, 1);

    let opp_beer = _engine::graph::BeerSource {
        kind: _engine::graph::BeerSourceKind::Opponent,
        key: usize::MAX,
        farm_idx: Some(1),
        merchant_idx: None,
    };
    let coal1 = test_coal_from_cannock(&state);
    let coal2 = test_coal_from_cannock(&state);
    let res = execute_network_double(&mut state, pid, 15, 33, coal1, coal2, opp_beer, 0);

    assert_eq!(
        res,
        Err("Chosen beer source is not legal for the second link".to_string())
    );
    let coal_tile = state
        .tile_at(Loc::Cannock, 0)
        .expect("coal tile should remain on board");
    assert_eq!(
        coal_tile.resource_cubes, 2,
        "failed double rail must not consume coal"
    );
    assert!(
        !coal_tile.flipped,
        "failed double rail must not flip the coal mine"
    );
    let opp_brewery = state
        .farm_tile(Loc::BrewerySouth)
        .expect("opponent brewery should remain on board");
    assert_eq!(
        opp_brewery.resource_cubes, 1,
        "failed double rail must not consume beer"
    );
    assert!(
        state.links[15].is_none() && state.links[33].is_none(),
        "failed double rail must not leave links on the board"
    );
    assert_eq!(
        state.players[pid].money, 100,
        "failed double rail must not spend money"
    );
    assert_eq!(
        state.players[pid].rail_links, LINKS_PER_PLAYER,
        "failed double rail must not consume rail links"
    );
}

#[test]
fn network_rejects_market_coal_while_free_source_available() {
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    // Player has presence + a connected free coal source (Cannock mine).
    place_test_presence_tile(&mut state, Loc::Stafford, pid);
    place_test_coal_mine(&mut state, 1);
    state.links[15] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });

    // Build a rail link using a PAID market coal source while a free connected
    // mine (Cannock) is available -> must fail under the free-first rule.
    let market_coal = _engine::graph::CoalSource {
        kind: _engine::graph::CoalSourceKind::Market,
        key: usize::MAX,
        price: state.coal_price(),
        free: false,
    };
    let res = execute_network(&mut state, pid, 33, Some(market_coal), 0);
    assert_eq!(
        res,
        Err("Free sources must be used before the market".to_string()),
        "market coal must be rejected while a free source is available"
    );
    assert!(
        state.links[33].is_none(),
        "failed network must not place the link"
    );
    let mine = state
        .tile_at(Loc::Cannock, 0)
        .expect("cannock mine should remain");
    assert_eq!(
        mine.resource_cubes, 2,
        "failed network must not consume free coal"
    );
    assert!(!mine.flipped, "failed network must not flip the mine");
}

#[test]
fn network_uses_the_explicitly_chosen_free_coal_source() {
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    place_test_presence_tile(&mut state, Loc::Stafford, pid);
    place_test_coal_mine(&mut state, 1);
    state.links[15] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });

    // Explicitly choose the Cannock free mine as the rail coal source.
    let coal_key = state.city_slot_key(Loc::Cannock, 0).expect("cannock slot");
    let coal = _engine::graph::CoalSource {
        kind: _engine::graph::CoalSourceKind::Mine,
        key: coal_key,
        price: 0,
        free: true,
    };
    let res = execute_network(&mut state, pid, 33, Some(coal), 0);
    assert!(res.is_ok(), "free coal network should succeed: {res:?}");
    assert!(state.links[33].is_some());
    let mine = state
        .tile_at(Loc::Cannock, 0)
        .expect("cannock mine should remain");
    assert_eq!(
        mine.resource_cubes, 1,
        "one free coal cube should be consumed"
    );
}

#[test]
fn sell_uses_the_explicitly_chosen_merchant_bonus() {
    let mut state = setup_clean_rail_state(3);
    let pid = 0;
    state.players[pid].hand = vec![Card::WildLocation];

    state.place_tile(
        Loc::Stone,
        0,
        BoardTile {
            player: pid,
            ind: IndustryType::CottonMill,
            def: industry_tiles(IndustryType::CottonMill)[0],
            flipped: false,
            resource_cubes: 0,
        },
    );
    place_test_brewery(&mut state, Loc::BrewerySouth, pid, 1);
    state.links[34] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });
    state.links[35] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });
    state.links[29] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });
    state.links[28] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });
    state.merchants = vec![
        _engine::state::MerchantTile {
            loc: Loc::Oxford,
            buys: _engine::state::BuyType::Industry(IndustryType::CottonMill),
            has_beer: true,
        },
        _engine::state::MerchantTile {
            loc: Loc::Warrington,
            buys: _engine::state::BuyType::Industry(IndustryType::CottonMill),
            has_beer: true,
        },
    ];
    let oxford_idx = 0usize;
    let warrington_idx = 1usize;

    let tile_key = state
        .city_slot_key(Loc::Stone, 0)
        .expect("stone slot should exist");
    let beer_sources = _engine::rules::plan_sell_beer_sources(
        &state,
        pid,
        &[tile_key],
        &[warrington_idx],
        &[true],
    )
    .expect("test sale should have a valid beer payment");
    let res = _engine::rules::execute_sell_with_free_develop(
        &mut state,
        pid,
        &[tile_key],
        &[warrington_idx],
        &[true],
        &beer_sources,
        None,
        0,
    );

    assert!(
        res.is_ok(),
        "sell should succeed when explicitly targeting warrington: {res:?}"
    );
    assert_eq!(state.players[pid].money, 105, "warrington should grant +£5");
    assert!(
        state
            .tile_at(Loc::Stone, 0)
            .expect("sold tile should remain on board")
            .flipped,
        "sold tile should flip"
    );
    assert!(
        !state.merchants[warrington_idx].has_beer,
        "chosen merchant beer should be consumed"
    );
    assert!(
        state.merchants[oxford_idx].has_beer,
        "unchosen merchant beer should remain"
    );
}

#[test]
fn gloucester_bonus_becomes_pending_resolve_move() {
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    state.players[pid].hand = vec![Card::WildLocation];
    state.players[pid].industry_next[IndustryType::Manufacturer as usize] = 1;

    state.place_tile(
        Loc::Kidderminster,
        1,
        BoardTile {
            player: pid,
            ind: IndustryType::CottonMill,
            def: industry_tiles(IndustryType::CottonMill)[0],
            flipped: false,
            resource_cubes: 0,
        },
    );
    place_test_brewery(&mut state, Loc::BrewerySouth, pid, 1);
    state.links[29] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });
    state.links[28] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });
    state.merchants = vec![_engine::state::MerchantTile {
        loc: Loc::Gloucester,
        buys: _engine::state::BuyType::Industry(IndustryType::CottonMill),
        has_beer: true,
    }];

    let tile_key = state
        .city_slot_key(Loc::Kidderminster, 1)
        .expect("kidderminster slot should exist");
    let beer_sources =
        _engine::rules::plan_sell_beer_sources(&state, pid, &[tile_key], &[0], &[true])
            .expect("test sale should have a valid beer payment");
    let res = _engine::rules::execute_sell_with_free_develop(
        &mut state,
        pid,
        &[tile_key],
        &[0],
        &[true],
        &beer_sources,
        Some(IndustryType::CottonMill),
        0,
    );
    assert!(res.is_ok(), "sell to gloucester should succeed: {res:?}");
    let turn_result = advance_turn(&mut state);
    assert!(matches!(turn_result, _engine::engine::TurnResult::Continue));
    assert!(
        state.players[pid].remaining_count(IndustryType::CottonMill)
            < _engine::state::player_industry_stack(IndustryType::CottonMill).len(),
        "free develop resolves within the sell action"
    );
}

#[test]
fn heuristic_sell_plan_does_not_overbook_a_single_merchant_beer() {
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    state.players[pid].hand = vec![Card::WildLocation];

    state.place_tile(
        Loc::Worcester,
        0,
        BoardTile {
            player: pid,
            ind: IndustryType::CottonMill,
            def: industry_tiles(IndustryType::CottonMill)[0],
            flipped: false,
            resource_cubes: 0,
        },
    );
    state.place_tile(
        Loc::Worcester,
        1,
        BoardTile {
            player: pid,
            ind: IndustryType::CottonMill,
            def: industry_tiles(IndustryType::CottonMill)[0],
            flipped: false,
            resource_cubes: 0,
        },
    );
    state.links[28] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });
    state.merchants = vec![_engine::state::MerchantTile {
        loc: Loc::Gloucester,
        buys: _engine::state::BuyType::Industry(IndustryType::CottonMill),
        has_beer: true,
    }];

    let sell = _engine::heuristic_ai::candidate_actions_k(&mut state, 1)
        .into_iter()
        .find_map(|d| match d.mv {
            _engine::rules::ResolvedMove::Sell {
                keys,
                merchant_indices,
                use_merchant_beer,
                beer_sources,
                card_index,
                ..
            } => Some((
                keys,
                merchant_indices,
                use_merchant_beer,
                beer_sources,
                card_index,
            )),
            _ => None,
        })
        .expect("heuristic should generate a sell candidate");

    assert_eq!(
        sell.0.len(),
        1,
        "heuristic sell plan must not try to sell two cotton mills with one merchant beer"
    );

    let mut sim = state.clone();
    let res =
        _engine::rules::execute_sell(&mut sim, pid, &sell.0, &sell.1, &sell.2, &sell.3, sell.4);
    assert!(
        res.is_ok(),
        "generated sell plan must execute successfully: {res:?}"
    );
}

#[test]
fn every_double_rail_move_from_legal_moves_executes() {
    use _engine::rules::{ResolvedMove, apply_move, legal_resolved_moves};
    let mut state = setup_clean_rail_state(2);
    let pid = 0;
    place_test_presence_tile(&mut state, Loc::Stafford, pid);
    place_test_coal_mine(&mut state, 1);
    place_test_brewery(&mut state, Loc::BrewerySouth, pid, 1);

    // Every generated NetworkDouble must execute; this is the regression guard
    // for the coal1/coal2 enumeration fix.
    for mv in legal_resolved_moves(&mut state) {
        if let ResolvedMove::NetworkDouble { .. } = &mv {
            let mut sim = state.clone();
            let res = apply_move(&mut sim, &mv);
            assert!(
                res.is_ok(),
                "raw double-rail move must execute (Task B): {} -> {res:?}",
                mv.describe(&state)
            );
        }
    }
}

#[test]
fn legal_moves_include_executable_multi_tile_sell() {
    use _engine::rules::{ResolvedMove, apply_move, legal_resolved_moves};
    let mut state = setup_clean_rail_state(3);
    let pid = 0;
    state.players[pid].hand = vec![Card::WildLocation];

    // Two unflipped cotton mills, both connected to the Warrington cotton
    // merchant via Stone–Stoke (34) + Stoke–Warrington (35).
    let cotton = BoardTile {
        player: pid,
        ind: IndustryType::CottonMill,
        def: industry_tiles(IndustryType::CottonMill)[0],
        flipped: false,
        resource_cubes: 0,
    };
    state.place_tile(Loc::Stone, 0, cotton.clone());
    state.place_tile(Loc::StokeOnTrent, 0, cotton);
    state.place_tile(
        Loc::Stone,
        1,
        BoardTile {
            player: pid,
            ind: IndustryType::CottonMill,
            def: industry_tiles(IndustryType::CottonMill)[0],
            flipped: false,
            resource_cubes: 0,
        },
    );
    state.place_tile(
        Loc::StokeOnTrent,
        1,
        BoardTile {
            player: pid,
            ind: IndustryType::CottonMill,
            def: industry_tiles(IndustryType::CottonMill)[0],
            flipped: false,
            resource_cubes: 0,
        },
    );
    place_test_brewery(&mut state, Loc::BrewerySouth, pid, 4);
    state.links[34] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });
    state.links[35] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    });
    state.merchants = vec![_engine::state::MerchantTile {
        loc: Loc::Warrington,
        buys: _engine::state::BuyType::Industry(IndustryType::CottonMill),
        has_beer: true,
    }];

    let moves = legal_resolved_moves(&mut state);
    let multi: Vec<&ResolvedMove> = moves
        .iter()
        .filter(|mv| matches!(mv, ResolvedMove::Sell { keys, .. } if keys.len() >= 2))
        .collect();
    assert!(!multi.is_empty(), "a multi-tile sell should be generated");
    assert!(
        moves
            .iter()
            .any(|mv| matches!(mv, ResolvedMove::Sell { keys, .. } if keys.len() == 4)),
        "a Sell action must be able to include all four tiles; no artificial 3-tile cap"
    );

    let multi_mv = multi
        .iter()
        .find(|mv| matches!(mv, ResolvedMove::Sell { keys, .. } if keys.len() == 2))
        .copied()
        .expect("a two-tile sell should be generated");
    let mut sim = state.clone();
    let res = apply_move(&mut sim, &multi_mv);
    assert!(res.is_ok(), "multi-tile sell must execute: {res:?}");
    let sold_keys = match multi_mv {
        ResolvedMove::Sell { keys, .. } => keys,
        _ => unreachable!(),
    };
    assert!(
        sold_keys.iter().all(|&key| sim.city_tiles[key]
            .as_ref()
            .map(|t| t.flipped)
            .unwrap_or(false)),
        "a multi-tile sell must flip every selected tile"
    );
}

// ---------------------------------------------------------------------------
// Market resource pricing: each cube costs its own market-slot price, so a
// multi-cube purchase pays the ascending slot prices (not N x the cheapest).
// ---------------------------------------------------------------------------

#[test]
fn market_iron_multicube_purchase_pays_ascending_slot_prices() {
    let mut state = setup(4);
    // IRON_MARKET_PRICES = [1,1,2,2,3,3,4,4,5,5]. With 7 cubes they occupy
    // slots 3..9 (prices 2,3,3,4,4,5): the two cheapest are £2 and £3, so a
    // 2-iron market purchase must cost £5 — NOT 2x the cheapest (£4).
    state.iron_market = 7;
    let opts = _engine::rules::iron_source_options(&state, 2);
    assert!(!opts.is_empty(), "expected legal 2-iron market selections");
    for sel in &opts {
        assert_eq!(sel.len(), 2);
        let paid: i32 = sel.iter().filter(|s| !s.free).map(|s| s.price as i32).sum();
        assert_eq!(paid, 5, "2-iron purchase must cost £2+£3, got £{paid}");
    }
}

#[test]
fn market_iron_even_market_two_cheapest_share_a_price_pair() {
    let state = setup(4);
    // With 8 cubes (initial) slots 2..9 hold prices 2,2,3,3,4,4,5,5; the two
    // cheapest are £2+£2 = £4.
    assert_eq!(state.iron_market, 8);
    let opts = _engine::rules::iron_source_options(&state, 2);
    assert!(!opts.is_empty(), "expected legal 2-iron market selections");
    for sel in &opts {
        assert_eq!(sel.len(), 2);
        let paid: i32 = sel.iter().filter(|s| !s.free).map(|s| s.price as i32).sum();
        assert_eq!(
            paid, 4,
            "full-market 2-iron purchase should cost £2+£2, got £{paid}"
        );
    }
}

#[test]
fn market_coal_multicube_purchase_pays_ascending_slot_prices() {
    let mut state = setup(4);
    // COAL_MARKET_PRICES = [1,1,2,2,3,3,4,4,5,5,6,6,7,7]. With 5 cubes they
    // occupy slots 9..13 (prices 5,6,6,7,7): a 2-coal market purchase costs
    // £5+£6 = £11, NOT 2x £5 = £10. Gloucester is a merchant, so the market is
    // reachable with no links built.
    state.coal_market = 5;
    let sources = _engine::graph::find_coal_sources(&state, Loc::Gloucester);
    let market_prices: Vec<u8> = sources
        .iter()
        .filter(|s| !s.free)
        .map(|s| s.price)
        .collect();
    assert_eq!(market_prices[0], 5, "cheapest market coal must be £5");
    assert_eq!(
        market_prices[1], 6,
        "second-cheapest market coal must be £6"
    );

    let opts = _engine::rules::coal_source_options(&state, Loc::Gloucester, 2);
    assert!(!opts.is_empty(), "expected legal 2-coal market selections");
    for sel in &opts {
        assert_eq!(sel.len(), 2);
        let paid: i32 = sel.iter().filter(|s| !s.free).map(|s| s.price as i32).sum();
        assert_eq!(paid, 11, "2-coal purchase must cost £5+£6, got £{paid}");
    }
}

#[test]
fn empty_market_draws_from_general_supply_at_empty_price() {
    let mut state = setup(4);
    state.coal_market = 0;
    let sources = _engine::graph::find_coal_sources(&state, Loc::Gloucester);
    let market: Vec<u8> = sources
        .iter()
        .filter(|s| !s.free)
        .map(|s| s.price)
        .collect();
    assert_eq!(market.len(), _engine::map::GENERAL_SUPPLY_CAP);
    assert!(
        market.iter().all(|&p| p == _engine::map::COAL_EMPTY_PRICE),
        "empty market should only offer General Supply at the empty price"
    );
    // A 2-coal draw from an empty market is still legal at £8 + £8.
    let opts = _engine::rules::coal_source_options(&state, Loc::Gloucester, 2);
    assert!(
        !opts.is_empty(),
        "expected General Supply 2-coal selections"
    );
    let paid: i32 = opts[0]
        .iter()
        .filter(|s| !s.free)
        .map(|s| s.price as i32)
        .sum();
    assert_eq!(paid, 16, "2-coal from empty market must cost £8+£8");
}

fn place_test_iron_works(state: &mut GameState, owner: usize, cubes: u8) {
    state.place_tile(
        Loc::Derby,
        2,
        BoardTile {
            player: owner,
            ind: IndustryType::IronWorks,
            def: industry_tiles(IndustryType::IronWorks)[0],
            flipped: false,
            resource_cubes: cubes,
        },
    );
}

// ---------------------------------------------------------------------------
// Cached free-resource & connectivity: `GameState` maintains free coal/iron
// sources and a lazily-rebuilt connected-component cache (no BFS on queries).
// ---------------------------------------------------------------------------

#[test]
fn coal_purchase_cost_is_free_first_then_slot_prices_then_general_supply() {
    let mut state = setup(4);
    place_test_coal_mine(&mut state, 0); // Cannock, 2 cubes
    // No merchant reachable yet (no links): free coal only, paid shortfall is
    // infeasible even though the mine is at the queried city itself.
    assert_eq!(
        _engine::graph::coal_purchase_cost(&state, Loc::Cannock, 2),
        Some(0)
    );
    assert_eq!(
        _engine::graph::coal_purchase_cost(&state, Loc::Cannock, 3),
        None
    );

    // Connect Cannock -> Walsall -> Birmingham -> Oxford (merchant), and
    // Cannock -> Stafford, so the mine is reachable and a merchant is too.
    let link = |p| {
        Some(_engine::state::Link {
            player: p,
            is_canal: false,
        })
    };
    state.links[17] = link(0); // Cannock-Walsall
    state.links[8] = link(0); // Walsall-Birmingham
    state.links[5] = link(0); // Birmingham-Oxford
    state.links[15] = link(0); // Cannock-Stafford

    assert_eq!(
        _engine::graph::coal_purchase_cost(&state, Loc::Oxford, 2),
        Some(0)
    );
    // Free 2 + one market cube at the cheapest slot (£1, market has 13).
    assert_eq!(
        _engine::graph::coal_purchase_cost(&state, Loc::Oxford, 3),
        Some(1)
    );

    // Odd market (5 cubes => slots 5,6,6,7,7): 2 free + 2 market = £5+£6.
    state.coal_market = 5;
    assert_eq!(
        _engine::graph::coal_purchase_cost(&state, Loc::Oxford, 4),
        Some(11)
    );
    // 2 free + 5 market + 1 General Supply at £8.
    assert_eq!(
        _engine::graph::coal_purchase_cost(&state, Loc::Oxford, 8),
        Some(31 + 8)
    );
}

#[test]
fn iron_purchase_cost_is_free_first_then_slot_prices_then_general_supply() {
    let mut state = setup(4);
    // Full market (8 cubes => slots 2,2,3,3,4,4,5,5): 2 iron = £2+£2.
    assert_eq!(_engine::graph::iron_purchase_cost(&state, 2), 4);
    // Odd market (7 cubes => slots 2,3,3,4,4,5): two cheapest are £2+£3.
    state.iron_market = 7;
    assert_eq!(_engine::graph::iron_purchase_cost(&state, 2), 5);
    // Market exhausted then General Supply at £6: 8 slots sum to 28 + 2x£6.
    state.iron_market = 8;
    assert_eq!(_engine::graph::iron_purchase_cost(&state, 10), 28 + 12);

    // Free-first: an unflipped iron works with 3 cubes covers the draw.
    place_test_iron_works(&mut state, 0, 3);
    assert_eq!(_engine::graph::iron_purchase_cost(&state, 3), 0);
    assert_eq!(_engine::graph::iron_purchase_cost(&state, 4), 2); // one market cube £2
}

#[test]
fn free_source_cache_tracks_placement_consume_and_era_end() {
    let mut state = setup(4);
    state.assert_caches_consistent();

    place_test_coal_mine(&mut state, 0); // Cannock, 2 cubes
    place_test_iron_works(&mut state, 1, 1);
    state.assert_caches_consistent();

    let coal_key = state.city_slot_key(Loc::Cannock, 0).unwrap();
    state.consume_from_city(coal_key); // 2 -> 1, still free
    state.assert_caches_consistent();
    state.consume_from_city(coal_key); // 1 -> 0, flips, no longer free
    state.assert_caches_consistent();

    // Level-1 tiles (and all links) are removed at the canal-era end; the
    // free-source cache must be rebuilt to empty.
    handle_turn_result(&mut state, TurnResult::EndCanalEra);
    state.assert_caches_consistent();
    let coal = _engine::graph::find_coal_sources(&state, Loc::Cannock);
    assert!(
        coal.iter().all(|s| !s.free),
        "era end must drop the level-1 coal mine from free sources"
    );
    let iron = _engine::graph::find_iron_sources(&state);
    assert!(
        iron.iter().all(|s| !s.free),
        "era end must drop the level-1 iron works"
    );
}

#[test]
fn connectivity_cache_self_heals_after_direct_link_writes() {
    let mut state = setup(4);
    place_test_coal_mine(&mut state, 0); // Cannock, 2 cubes

    // Direct link writes (bypass any maintenance hook): not reachable yet.
    let free = |s: &GameState, loc: Loc| {
        _engine::graph::find_coal_sources(s, loc)
            .iter()
            .filter(|x| x.free)
            .count()
    };
    assert_eq!(free(&state, Loc::Oxford), 0);

    let link = Some(_engine::state::Link {
        player: 0,
        is_canal: false,
    });
    state.links[17] = link; // Cannock-Walsall
    state.links[8] = link; // Walsall-Birmingham
    state.links[5] = link; // Birmingham-Oxford
    assert_eq!(
        free(&state, Loc::Oxford),
        2,
        "component cache must rebuild on the next query"
    );

    // Removing the last link severs the component again.
    state.links[5] = None;
    state.links[8] = None;
    state.links[17] = None;
    assert_eq!(free(&state, Loc::Oxford), 0);
}

#[test]
fn network_mask_is_kept_by_moves_and_self_heals_after_direct_link_writes() {
    let mut state = setup(4);
    let pid = 0;

    // Maintained path: a placed tile enters the player's network immediately.
    place_test_presence_tile(&mut state, Loc::Stafford, pid);
    assert!(_engine::graph::is_in_network(&state, pid, Loc::Stafford));
    assert!(!_engine::graph::is_in_network(&state, pid, Loc::Birmingham));
    assert!(_engine::graph::player_has_presence(&state, pid));
    assert!(!_engine::graph::player_has_presence(&state, 1));

    // Direct link writes bypass the maintenance hooks; the mask must self-heal
    // on the next batch entry (get_valid_network_targets) and, once healed,
    // direct is_in_network reads agree with the scan.
    state.links[15] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    }); // Cannock-Stafford
    let _ = get_valid_network_targets(&state, pid);
    assert!(
        _engine::graph::is_in_network(&state, pid, Loc::Cannock),
        "network mask must self-heal after a direct link write"
    );
    state.links[8] = Some(_engine::state::Link {
        player: pid,
        is_canal: false,
    }); // Walsall-Birmingham
    let _ = get_valid_network_targets(&state, pid);
    assert!(
        _engine::graph::is_in_network(&state, pid, Loc::Birmingham),
        "network mask must reflect chained direct link writes"
    );

    // Removing the last touching link severs the location again.
    state.links[15] = None;
    state.links[8] = None;
    let _ = get_valid_network_targets(&state, pid);
    assert!(
        !_engine::graph::is_in_network(&state, pid, Loc::Birmingham),
        "removing the last link must sever the location from the network"
    );
    // The own tile keeps its own location in the network.
    assert!(_engine::graph::is_in_network(&state, pid, Loc::Stafford));

    // A committed network move updates the mask through the rules layer.
    let conn_id = get_valid_network_targets(&state, pid)[0];
    execute_network(&mut state, pid, conn_id, None, 0).expect("network must succeed");
    let c = &connections()[conn_id];
    assert!(
        _engine::graph::is_in_network(&state, pid, c.a)
            && _engine::graph::is_in_network(&state, pid, c.b),
        "executed network must extend the player's network mask"
    );
}

#[test]
fn double_develop_rechecks_the_uncovered_same_industry_tile() {
    let mut state = setup_clean_rail_state(2);
    let pid = state.current_player_id();
    // Pottery Lv2 may be developed, but the Lv3 tile exposed afterwards may
    // not. A same-industry double develop must therefore be illegal.
    state.players[pid].industry_next[IndustryType::Pottery as usize] = 1;
    let iron = iron_source_options(&state, 2)
        .into_iter()
        .next()
        .expect("two iron sources must be available");
    let mv = ResolvedMove::Develop {
        ind1: IndustryType::Pottery,
        ind2: Some(IndustryType::Pottery),
        iron,
        card_index: 0,
    };

    assert!(
        !legal_resolved_moves(&mut state)
            .iter()
            .any(|candidate| matches!(
                candidate,
                ResolvedMove::Develop {
                    ind1: IndustryType::Pottery,
                    ind2: Some(IndustryType::Pottery),
                    ..
                }
            )),
        "the invalid same-industry double develop must not be generated"
    );

    let before_money = state.players[pid].money;
    let before_hand_len = state.players[pid].hand.len();
    let before_iron_market = state.iron_market;
    assert_eq!(
        apply_move(&mut state, &mv),
        Err("Second industry is not developable".into())
    );
    assert_eq!(
        state.players[pid].industry_next[IndustryType::Pottery as usize],
        1
    );
    assert_eq!(state.players[pid].money, before_money);
    assert_eq!(state.players[pid].hand.len(), before_hand_len);
    assert_eq!(state.iron_market, before_iron_market);
}

#[test]
fn heuristic_keeps_same_industry_double_develop_as_a_candidate() {
    let mut state = setup_clean_rail_state(2);
    let pid = state.current_player_id();

    // Leave Brewery as the only developable industry. Its first two tiles are
    // both developable, so the heuristic must be able to choose Brewery +
    // Brewery rather than falling back to a single develop.
    for ind in IndustryType::ALL {
        if ind != IndustryType::Brewery {
            state.players[pid].industry_next[ind as usize] =
                industry_tiles(ind).iter().map(|tile| tile.count).sum();
        }
    }

    let candidates = _engine::heuristic_ai::candidate_actions_k(&mut state, 1);
    let candidate_moves: Vec<String> = candidates
        .iter()
        .map(|decision| format!("{:?}", decision.mv))
        .collect();
    assert!(
        candidates.iter().any(|decision| matches!(
            decision.mv,
            ResolvedMove::Develop {
                ind1: IndustryType::Brewery,
                ind2: Some(IndustryType::Brewery),
                ..
            }
        )),
        "heuristic must retain a legal same-industry double develop; candidates: {candidate_moves:?}"
    );
}
