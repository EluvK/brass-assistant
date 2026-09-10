"""Type declarations for the PyO3 ``brass_ai._engine`` extension.

Rust implementation: ``../../engine/src/bridge/pymod.rs`` (`PyGame` and its
``#[pymethods]`` block). Ctrl+click the relative path in IDEs that recognize
paths in comments; otherwise use it as the manual navigation target.
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

ACTION_FEATURE_DIM: int
ACTION_SCHEMA_VERSION: int
ACTION_REF_CAP: int
ACTION_KIND_COUNT: int
ACTION_NUMBERS: int
REF_KIND_COUNT: int
ACTION_OFF_KIND: int
ACTION_OFF_SLOT: int
ACTION_OFF_NUMBERS: int
ACTION_OFF_REF_COUNT: int
ACTION_OFF_REFS: int
ACTION_REF_CELL: int
ACTION_REF_LINK: int
ACTION_REF_MERCHANT: int
ACTION_REF_INDUSTRY: int
ACTION_REF_CARD: int
BOARD_CELLS: int
LINK_CELLS: int
MERCHANT_COUNT: int
SEAT_COUNT: int
INDUSTRY_COUNT: int
CARD_SEMANTIC_COUNT: int
TOKEN_COUNT: int
F_CELL: int
F_LINK: int
F_MERCHANT: int
F_SEAT: int
F_GLOBAL: int
VP_SCALE: float
STATE_TOKEN_SCHEMA_VERSION: int
LOCATION_COUNT: int
BOARD_CELL_LOCATIONS: List[int]
BOARD_CELL_SLOTS: List[int]
CONNECTION_ENDPOINTS: List[int]
CONNECTION_VIA_FARMS: List[int]

# Token feature offsets (docs/ai-action-encoding.md §2).
CELL_OWNER: int
CELL_OCCUPIED: int
CELL_INDUSTRY: int
CELL_FLIPPED: int
CELL_CUBES: int
CELL_LEVEL: int
CELL_VP: int
CELL_INCOME: int
CELL_IS_FARM: int
CELL_SLOT: int
CELL_ALLOWED: int
CELL_IN_NET: int
CELL_DIST: int
LINK_CANAL: int
LINK_RAIL: int
LINK_VIA_FARM: int
LINK_BUILT: int
LINK_OWNER: int
LINK_IS_CANAL: int
LINK_IN_NET: int
LINK_TOUCHES_NET: int
MERCHANT_BUY: int
MERCHANT_BEER: int
SEAT_MONEY: int
SEAT_INCOME_SPACE: int
SEAT_INCOME_LEVEL: int
SEAT_VP: int
SEAT_CANAL_LINKS: int
SEAT_RAIL_LINKS: int
SEAT_HAND_SIZE: int
SEAT_WILD_LOCATION: int
SEAT_WILD_INDUSTRY: int
SEAT_REMAINING: int
SEAT_SPENT: int
SEAT_IS_CURRENT: int
SEAT_HAND_SAMPLED_FLAG: int
SEAT_HAND_REAL: int
SEAT_HAND_SAMPLED: int
SEAT_HAND_PUBLIC: int
GLOBAL_ERA: int
GLOBAL_ROUND: int
GLOBAL_ROUNDS_REMAINING: int
GLOBAL_ACTIONS: int
GLOBAL_COAL_MARKET: int
GLOBAL_IRON_MARKET: int
GLOBAL_DECK: int
GLOBAL_DISCARD: int
GLOBAL_QUEUE: int

NetCallback = Callable[
    [Any, Any, Any, Any, Any, Any, Any],
    Tuple[NDArray[np.float32], NDArray[np.float32]],
]


# Rust implementation: ../../engine/src/bridge/pymod.rs (`PyGame`).
class GameState:
    def __init__(self, seed: int, players: int) -> None: ...

    @property
    def current_player_id(self) -> int: ...
    @property
    def player_count(self) -> int: ...
    @property
    def era(self) -> int: ...
    @property
    def round(self) -> int: ...
    @property
    def turn_order(self) -> List[int]: ...
    @property
    def game_over(self) -> bool: ...
    @property
    def current_player_money(self) -> int: ...

    def clone(self) -> "GameState": ...
    def snapshot(self) -> bytes: ...
    @staticmethod
    def from_snapshot(snapshot: Union[bytes, bytearray, Sequence[int]]) -> "GameState": ...

    def legal_moves(self) -> List[Tuple[int, str, str]]: ...
    def legal_operations(self) -> List[Tuple[int, str, int, List[int]]]: ...
    def resolve_legal_operation(self, operation_id: int, selected_cards: List[int]) -> str: ...
    def legal_candidates(self) -> List[Tuple[str, List[float]]]: ...
    def heuristic_candidates(
        self,
    ) -> Tuple[List[float], List[float], List[float], str, int, float, float, int]: ...
    def search_net(
        self,
        net_fn: NetCallback,
        sims: int,
        c_puct: float,
        max_depth: int,
        dirichlet_alpha: float,
        dirichlet_weight: float,
        add_root_noise: bool,
        batch_size: int = 64,
        candidate_k: int = 0,
        prior_top_k: int = 0,
        fpu: bool = True,
        fpu_reduction: float = 0.0,
    ) -> Tuple[Optional[str], List[Tuple[int, str, int]], List[int], int, int]: ...

    def apply_move(self, canonical: str) -> str: ...
    def apply_move_raw(self, canonical: str) -> Tuple[str, bool]: ...
    def advance_turn_raw(self) -> str: ...
    def finish_canal_era(self) -> None: ...
    def finish_game(self) -> None: ...

    def replay_header(self) -> str: ...
    def replay_move_detail(self, canonical: str) -> str: ...
    def replay_action_type(self, canonical: str) -> Tuple[str, int]: ...
    def replay_era_score(self, pid: int) -> str: ...
    def replay_tiles(self, pid: int, flipped_only: bool = False) -> str: ...
    def replay_cleanup(self) -> str: ...
    def replay_player_state(self, pid: int) -> str: ...
    def replay_board(self) -> str: ...
    def replay_merchants(self) -> str: ...
    def replay_hand(self, pid: int) -> str: ...
    def replay_stats_line(
        self,
        build: Sequence[int],
        network: int,
        dbl_rail: int,
        develop: int,
        sell: int,
        loan: int,
        pass_: int,
        scout: int,
    ) -> str: ...

    def final_econ(self) -> List[Tuple[int, int]]: ...
    def canal_econ(self) -> List[Tuple[int, int]]: ...
    def determinize(self) -> "GameState": ...
    def state_tokens(
        self, perspective: Optional[int] = None
    ) -> Tuple[
        NDArray[np.float32],  # cells (49, F_CELL)
        NDArray[np.float32],  # links (39, F_LINK)
        NDArray[np.float32],  # merchants (9, F_MERCHANT)
        NDArray[np.float32],  # seats (4, F_SEAT)
        NDArray[np.float32],
    ]: ...
    def player_vps(self) -> List[int]: ...
    def final_ranking(self) -> List[int]: ...
    def choose_heuristic(self) -> Tuple[str, str, float]: ...
