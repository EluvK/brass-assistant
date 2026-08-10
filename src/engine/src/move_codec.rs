//! Canonical string codec for `Move`, shared by the PyO3 bridge and (later)
//! self-play data serialization.
//!
//! The encoding is lossless: every field needed to re-execute the move is
//! preserved (resource-source identity + price/free flags + card choice), so
//! `apply_move` on the decoded move validates exactly like the original.
//!
//! Source compact encodings (intra-source separator is `;` so the top-level
//! `,` field splitter is safe; `|` separates multiple sources):
//!   CoalSource : `<kind(M|K)>;<key|max>;<price>;<free 0/1>`
//!   IronSource : `<key|max>;<price>;<free 0/1>`
//!   BeerSource : `<kind(O|P|M)>;<key|max>;<farm_idx|->;<merchant_idx|->`

use crate::data::IndustryType;
use crate::graph::{BeerSource, BeerSourceKind, CoalSource, CoalSourceKind, IronSource};
use crate::rules::Move;

const MAX_STR: &str = "max";
const NONE_STR: &str = "-";

fn u(v: usize) -> String {
    if v == usize::MAX {
        MAX_STR.to_string()
    } else {
        v.to_string()
    }
}

fn un(s: &str) -> Result<usize, String> {
    if s == MAX_STR {
        Ok(usize::MAX)
    } else {
        s.parse::<usize>().map_err(|e| format!("bad usize {s:?}: {e}"))
    }
}

fn opt(s: &str) -> Result<Option<usize>, String> {
    if s == NONE_STR {
        Ok(None)
    } else {
        un(s).map(Some)
    }
}

// --- source encoders --------------------------------------------------------

fn coal(s: &CoalSource) -> String {
    format!(
        "{};{};{};{}",
        match s.kind {
            CoalSourceKind::Mine => "M",
            CoalSourceKind::Market => "K",
        },
        u(s.key),
        s.price,
        s.free as u8
    )
}

fn iron(s: &IronSource) -> String {
    format!("{};{};{}", u(s.key), s.price, s.free as u8)
}

fn beer(s: &BeerSource) -> String {
    let kind = match s.kind {
        BeerSourceKind::Own => "O",
        BeerSourceKind::Opponent => "P",
        BeerSourceKind::Merchant => "M",
    };
    format!(
        "{};{};{};{}",
        kind,
        u(s.key),
        s.farm_idx.map(|x| x.to_string()).unwrap_or_else(|| NONE_STR.to_string()),
        s.merchant_idx.map(|x| x.to_string()).unwrap_or_else(|| NONE_STR.to_string())
    )
}

fn coal_list(v: &[CoalSource]) -> String {
    v.iter().map(coal).collect::<Vec<_>>().join("|")
}

fn iron_list(v: &[IronSource]) -> String {
    v.iter().map(iron).collect::<Vec<_>>().join("|")
}

fn u_list(v: &[usize]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(";")
}

fn b_list(v: &[bool]) -> String {
    v.iter().map(|b| *b as u8).map(|x| x.to_string()).collect::<Vec<_>>().join(";")
}

// --- source decoders --------------------------------------------------------

fn dec_coal(s: &str) -> Result<CoalSource, String> {
    let parts: Vec<&str> = s.split(';').collect();
    if parts.len() != 4 {
        return Err(format!("bad CoalSource {s:?}"));
    }
    Ok(CoalSource {
        kind: if parts[0] == "M" {
            CoalSourceKind::Mine
        } else if parts[0] == "K" {
            CoalSourceKind::Market
        } else {
            return Err(format!("bad coal kind {}", parts[0]));
        },
        key: un(parts[1])?,
        price: parts[2].parse().map_err(|e| format!("bad coal price: {e}"))?,
        free: parts[3] == "1",
    })
}

fn dec_iron(s: &str) -> Result<IronSource, String> {
    let parts: Vec<&str> = s.split(';').collect();
    if parts.len() != 3 {
        return Err(format!("bad IronSource {s:?}"));
    }
    Ok(IronSource {
        key: un(parts[0])?,
        price: parts[1].parse().map_err(|e| format!("bad iron price: {e}"))?,
        free: parts[2] == "1",
    })
}

fn dec_beer(s: &str) -> Result<BeerSource, String> {
    let parts: Vec<&str> = s.split(';').collect();
    if parts.len() != 4 {
        return Err(format!("bad BeerSource {s:?}"));
    }
    Ok(BeerSource {
        kind: match parts[0] {
            "O" => BeerSourceKind::Own,
            "P" => BeerSourceKind::Opponent,
            "M" => BeerSourceKind::Merchant,
            k => return Err(format!("bad beer kind {k:?}")),
        },
        key: un(parts[1])?,
        farm_idx: opt(parts[2])?,
        merchant_idx: opt(parts[3])?,
    })
}

fn coal_list_dec(s: &str) -> Result<Vec<CoalSource>, String> {
    s.split('|').filter(|x| !x.is_empty()).map(dec_coal).collect()
}

fn iron_list_dec(s: &str) -> Result<Vec<IronSource>, String> {
    s.split('|').filter(|x| !x.is_empty()).map(dec_iron).collect()
}

fn u_list_dec(s: &str) -> Result<Vec<usize>, String> {
    s.split(';').filter(|x| !x.is_empty()).map(un).collect()
}

fn b_list_dec(s: &str) -> Result<Vec<bool>, String> {
    s.split(';')
        .filter(|x| !x.is_empty())
        .map(|x| {
            x.parse::<u8>()
                .map(|b| b != 0)
                .map_err(|e| format!("bad bool {x:?}: {e}"))
        })
        .collect()
}

fn ind(n: usize) -> Result<IndustryType, String> {
    IndustryType::ALL
        .get(n)
        .copied()
        .ok_or_else(|| format!("bad industry index {n}"))
}

fn card_enc(c: usize) -> String {
    c.to_string()
}

// --- encode -----------------------------------------------------------------

pub fn encode(mv: &Move) -> String {
    match mv {
        Move::Build {
            loc,
            slot_index,
            ind,
            coal,
            iron,
            card_index,
        } => format!(
            "Build{{loc:{},slot:{},ind:{},coal:{},iron:{},card:{}}}",
            *loc as usize,
            slot_index,
            *ind as usize,
            coal_list(coal),
            iron_list(iron),
            card_enc(*card_index)
        ),
        Move::Network {
            conn_id,
            coal: coal_src,
            card_index,
        } => format!(
            "Network{{conn:{},coal:{},card:{}}}",
            conn_id,
            coal_src.as_ref().map(coal).unwrap_or_else(|| NONE_STR.to_string()),
            card_enc(*card_index)
        ),
        Move::NetworkDouble {
            conn1,
            conn2,
            coal1,
            coal2,
            beer: beer_src,
            card_index,
        } => format!(
            "NetDouble{{c1:{},c2:{},coal1:{},coal2:{},beer:{},card:{}}}",
            conn1,
            conn2,
            coal(coal1),
            coal(coal2),
            beer(beer_src),
            card_enc(*card_index)
        ),        Move::Develop {
            ind1,
            ind2,
            iron,
            card_index,
        } => format!(
            "Develop{{ind1:{},ind2:{},iron:{},card:{}}}",
            *ind1 as usize,
            ind2.map(|x| x as usize)
                .map(|x| x.to_string())
                .unwrap_or_else(|| NONE_STR.to_string()),
            iron_list(iron),
            card_enc(*card_index)
        ),
        Move::Sell {
            keys,
            merchant_indices,
            use_merchant_beer,
            card_index,
        } => format!(
            "Sell{{keys:{},merchants:{},beer:{},card:{}}}",
            u_list(keys),
            u_list(merchant_indices),
            b_list(use_merchant_beer),
            card_enc(*card_index)
        ),
        Move::ResolveFreeDevelop { ind1, ind2 } => format!(
            "FreeDevelop{{ind1:{},ind2:{}}}",
            *ind1 as usize,
            ind2.map(|x| x as usize)
                .map(|x| x.to_string())
                .unwrap_or_else(|| NONE_STR.to_string())
        ),
        Move::Loan { card_index } => format!("Loan{{card:{}}}", card_enc(*card_index)),
        Move::Scout { card_indices } => format!(
            "Scout{{cards:{}}}",
            card_indices.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(";")
        ),
        Move::Pass { card_index } => format!("Pass{{card:{}}}", card_enc(*card_index)),
    }
}

// --- decode -----------------------------------------------------------------

fn field<'a>(body: &'a str, name: &str) -> Result<&'a str, String> {
    for part in body.split(',') {
        if let Some(rest) = part.strip_prefix(&format!("{name}:")) {
            return Ok(rest);
        }
    }
    Err(format!("missing field {name} in {body}"))
}

fn field_u(body: &str, name: &str) -> Result<usize, String> {
    un(field(body, name)?)
}

fn field_ind(body: &str, name: &str) -> Result<IndustryType, String> {
    ind(field_u(body, name)?)
}

fn field_opt_ind(body: &str, name: &str) -> Result<Option<IndustryType>, String> {
    let v = field(body, name)?;
    if v == NONE_STR {
        Ok(None)
    } else {
        ind(un(v)?).map(Some)
    }
}

pub fn decode(s: &str) -> Result<Move, String> {
    let s = s.trim();
    let (head, rest) = s
        .split_once('{')
        .ok_or_else(|| format!("bad move {s:?}"))?;
    let body = rest.strip_suffix('}').ok_or_else(|| format!("bad move {s:?}"))?;

    match head {
        "Build" => {
            let coal = coal_list_dec(field(body, "coal")?)?;
            let iron = iron_list_dec(field(body, "iron")?)?;
            let card_index = field_u(body, "card")?;
            Ok(Move::Build {
                loc: crate::map::ALL_LOCATIONS
                    .get(field_u(body, "loc")?)
                    .copied()
                    .ok_or_else(|| format!("bad loc in {s:?}"))?,
                slot_index: field_u(body, "slot")?,
                ind: field_ind(body, "ind")?,
                coal,
                iron,
                card_index,
            })
        }
        "Network" => {
            let coal_str = field(body, "coal")?;
            let coal = if coal_str == NONE_STR {
                None
            } else {
                Some(dec_coal(coal_str)?)
            };
            Ok(Move::Network {
                conn_id: field_u(body, "conn")?,
                coal,
                card_index: field_u(body, "card")?,
            })
        }
        "NetDouble" => Ok(Move::NetworkDouble {
            conn1: field_u(body, "c1")?,
            conn2: field_u(body, "c2")?,
            coal1: dec_coal(field(body, "coal1")?)?,
            coal2: dec_coal(field(body, "coal2")?)?,
            beer: dec_beer(field(body, "beer")?)?,
            card_index: field_u(body, "card")?,
        }),
        "Develop" => Ok(Move::Develop {
            ind1: field_ind(body, "ind1")?,
            ind2: field_opt_ind(body, "ind2")?,
            iron: iron_list_dec(field(body, "iron")?)?,
            card_index: field_u(body, "card")?,
        }),
        "Sell" => {
            let keys = u_list_dec(field(body, "keys")?)?;
            let merchant_indices = u_list_dec(field(body, "merchants")?)?;
            let use_merchant_beer = b_list_dec(field(body, "beer")?)?;
            Ok(Move::Sell {
                keys,
                merchant_indices,
                use_merchant_beer,
                card_index: field_u(body, "card")?,
            })
        }
        "FreeDevelop" => Ok(Move::ResolveFreeDevelop {
            ind1: field_ind(body, "ind1")?,
            ind2: field_opt_ind(body, "ind2")?,
        }),
        "Loan" => Ok(Move::Loan {
            card_index: field_u(body, "card")?,
        }),
        "Scout" => {
            let cards = u_list_dec(field(body, "cards")?)?;
            let card_indices: [usize; 3] = cards
                .try_into()
                .map_err(|_| format!("scout needs 3 cards in {s:?}"))?;
            Ok(Move::Scout { card_indices })
        }
        "Pass" => Ok(Move::Pass {
            card_index: field_u(body, "card")?,
        }),
        other => Err(format!("unknown move type {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy;
    use crate::rules::{apply_move, legal_moves};
    use crate::state::GameState;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn fresh_state(seed: u64, players: usize) -> GameState {
        GameState::new(StdRng::seed_from_u64(seed), players)
    }

    #[test]
    fn encode_decode_roundtrip_over_legal_moves() {
        for seed in [1u64, 7, 42, 1337] {
            let mut state = fresh_state(seed, 4);
            let moves = legal_moves(&mut state);
            assert!(!moves.is_empty(), "seed {seed} produced no legal moves");
            for mv in moves {
                let s1 = encode(&mv);
                let decoded = decode(&s1).unwrap_or_else(|e| panic!("decode({s1:?}): {e}"));
                let s2 = encode(&decoded);
                assert_eq!(s1, s2, "roundtrip not idempotent for {s1:?}");
                // The decoded move must be legal & executable on a fresh clone.
                let mut sim = fresh_state(seed, 4);
                assert!(
                    apply_move(&mut sim, &decoded).is_ok(),
                    "decoded move failed to apply: {s1:?}"
                );
            }
        }
    }

    #[test]
    fn policy_slots_are_stable_and_in_range() {
        let size = policy::policy_table_size();
        for seed in [2u64, 99, 2026] {
            let mut state = fresh_state(seed, 4);
            let moves = legal_moves(&mut state);
            for mv in &moves {
                let slots = policy::move_slots(mv);
                assert!(!slots.is_empty());
                for &s in &slots {
                    assert!(
                        s < size,
                        "slot {s} out of range ({size}) for {}",
                        encode(mv)
                    );
                }
                // Encode->decode keeps the same slot set.
                let decoded = decode(&encode(mv)).unwrap();
                assert_eq!(slots, policy::move_slots(&decoded));
            }
            // Mask is a subset of the full table.
            let mask = policy::legal_mask(&mut state);
            for &s in &mask {
                assert!(s < size);
            }
        }
    }

    #[test]
    fn malformed_inputs_are_rejected_not_panicked() {
        // A garbage farm/merchant index in a beer source must error, not decode
        // into usize::MAX (which would panic on apply via farm_tiles[MAX]).
        let bad = "NetDouble{c1:0,c2:11,coal1:M;3;0;1,coal2:M;4;0;1,beer:O;3;garbage;-;-,card:0}".replace(";-;-", ";garbage;");
        assert!(decode(&bad).is_err(), "malformed beer farm_idx must be rejected");

        assert!(decode("Garbage{foo:1}").is_err());
        assert!(decode("Build{loc:99,slot:0,ind:0,coal:,iron:,card:0}").is_err());
        assert!(decode("NetDouble{c1:0,c2:11,coal1:M;3;0;1,coal2:M;4;0;1,beer:M;max;-;abc,card:0}").is_err());
    }

    #[test]
    fn distinct_build_slots_map_to_distinct_indices() {
        // Build at city 5 (Uttoxeter) slot 0 industry Brewery(5) vs CottonMill(0).
        let m1 = Move::Build {
            loc: crate::map::Loc::Uttoxeter,
            slot_index: 0,
            ind: IndustryType::Brewery,
            coal: vec![],
            iron: vec![],
            card_index: 0,
        };
        let m2 = Move::Build {
            loc: crate::map::Loc::Uttoxeter,
            slot_index: 0,
            ind: IndustryType::CottonMill,
            coal: vec![],
            iron: vec![],
            card_index: 0,
        };
        let s1 = policy::move_slots(&m1);
        let s2 = policy::move_slots(&m2);
        assert_eq!(s1.len(), 1);
        assert_ne!(s1[0], s2[0]);
        // Resource-source choices collapse to the same slot.
        let m3 = Move::Build {
            loc: crate::map::Loc::Uttoxeter,
            slot_index: 0,
            ind: IndustryType::Brewery,
            coal: vec![CoalSource {
                kind: CoalSourceKind::Market,
                key: usize::MAX,
                price: 7,
                free: false,
            }],
            iron: vec![],
            card_index: 1,
        };
        assert_eq!(policy::move_slots(&m3), s1);
    }
}
