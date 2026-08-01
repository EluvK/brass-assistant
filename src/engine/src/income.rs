//! Income track: progress-track SPACE (0-99) <-> income LEVEL (-10..30).
//!
//! Translated from gameData.js incomeLevelFromSpace / incomeHighestSpaceOfLevel.

use crate::map::{MAX_INCOME_SPACE, MIN_INCOME};

pub fn income_level_from_space(space: u8) -> i8 {
    let space = space as i16;
    if space <= 10 {
        return (space - 10) as i8;
    }
    if space <= 30 {
        return ((space - 10 + 1) / 2) as i8; // ceil((space-10)/2)
    }
    if space <= 60 {
        return (10 + (space - 30 + 2) / 3) as i8; // 10 + ceil((space-30)/3)
    }
    if space <= 96 {
        return (20 + (space - 60 + 3) / 4) as i8; // 20 + ceil((space-60)/4)
    }
    30
}

/// Highest space within the given level (used when a loan drops the marker).
pub fn income_highest_space_of_level(level: i8) -> u8 {
    if level <= 0 {
        return (level + 10) as u8;
    }
    if level <= 10 {
        return (10 + 2 * level) as u8;
    }
    if level <= 20 {
        return (30 + 3 * (level - 10)) as u8;
    }
    if level <= 29 {
        return (60 + 4 * (level - 20)) as u8;
    }
    MAX_INCOME_SPACE
}

/// Can a player at `level` take a loan without dropping below MIN_INCOME?
pub fn can_take_loan_at(level: i8) -> bool {
    level - crate::map::LOAN_INCOME_PENALTY >= MIN_INCOME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_at_initial_space_is_zero() {
        assert_eq!(income_level_from_space(10), 0);
    }

    #[test]
    fn round_trip_boundaries() {
        // For each space 0..=99, the level derived should be within a band
        // whose highest space matches expected.
        for space in 0..=99u8 {
            let level = income_level_from_space(space);
            let high = income_highest_space_of_level(level);
            assert!(high >= space, "space {space} -> level {level} -> high {high}");
            // The next level's highest space should be below space (level not off-by-one).
            if level < 30 {
                let next_high = income_highest_space_of_level(level + 1);
                assert!(
                    next_high > high,
                    "level {level}: high {high} not below next {next_high}"
                );
            }
        }
    }

    #[test]
    fn known_level_markers() {
        // spaces 11-30 -> levels 1..10 (2 spaces per level)
        assert_eq!(income_level_from_space(11), 1);
        assert_eq!(income_level_from_space(12), 1);
        assert_eq!(income_level_from_space(30), 10);
        assert_eq!(income_highest_space_of_level(1), 12);
        assert_eq!(income_highest_space_of_level(10), 30);
        // spaces 31-60 -> levels 11..20 (3 per level)
        assert_eq!(income_level_from_space(31), 11);
        assert_eq!(income_level_from_space(60), 20);
        // 61-96 -> 21..29
        assert_eq!(income_level_from_space(61), 21);
        assert_eq!(income_level_from_space(96), 29);
        // 97-99 -> 30
        assert_eq!(income_level_from_space(97), 30);
        assert_eq!(income_highest_space_of_level(30), 99);
        // negative band
        assert_eq!(income_level_from_space(0), -10);
        assert_eq!(income_highest_space_of_level(-10), 0);
    }
}
