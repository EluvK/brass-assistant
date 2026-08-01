use brass_engine::data::{industry_tiles, Era, IndustryType};
use brass_engine::map::*;
use brass_engine::rules::{
    execute_build, execute_network, get_valid_build_targets, get_valid_network_targets,
};
use brass_engine::scoring;
use brass_engine::state::GameState;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn setup(players: usize) -> GameState<StdRng> {
    let rng = StdRng::seed_from_u64(42);
    GameState::new(rng, players)
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
        for e in brass_engine::map::merchant_tile_mix(p) {
            match e {
                brass_engine::map::MerchantMixEntry::Blank => blanks += 1,
                brass_engine::map::MerchantMixEntry::Any => any += 1,
                brass_engine::map::MerchantMixEntry::Buys(t) => match t {
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
        .filter(|m| m.buys == brass_engine::state::BuyType::Any)
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
            brass_engine::state::Card::Location(l) => *l == t.loc && !t.loc.is_farm(),
            brass_engine::state::Card::Industry { .. } => {
                c.is_industry(t.ind)
                    && (brass_engine::graph::is_in_network(&state, pid, t.loc)
                        || !brass_engine::graph::player_has_presence(&state, pid))
            }
            brass_engine::state::Card::WildLocation => !t.loc.is_farm(),
            brass_engine::state::Card::WildIndustry => true,
        })
        .expect("valid build target must have a matching card");

    let res = execute_build(&mut state, pid, t.loc, t.slot_index, t.ind, card_idx);
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
    let res = execute_network(&mut state, pid, conn_id, 0);
    assert!(res.is_ok(), "network failed: {:?}", res);
    assert!(state.links[conn_id].is_some());
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
        brass_engine::state::BoardTile {
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
fn loan_lowers_income_by_three_levels() {
    let mut state = setup(4);
    let pid = state.current_player_id();
    assert!(state.can_take_loan(pid));
    let before = state.players[pid].income_level();
    let res = brass_engine::rules::execute_loan(&mut state, pid, 0);
    assert!(res.is_ok());
    let after = state.players[pid].income_level();
    assert_eq!(before - after, 3, "loan should drop income 3 levels");
    assert_eq!(state.players[pid].money, INITIAL_MONEY + LOAN_AMOUNT);
}
