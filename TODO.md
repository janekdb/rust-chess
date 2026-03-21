
● Based on the plan, here's what remains:

AI Improvements

- Move ordering (ai.rs) — Sort captures before quiet moves in the negamax loop. Roughly doubles effective search depth at no extra cost.
- Piece-square tables (ai.rs) — Add 6 static [i32; 64] PST arrays for positional evaluation (center control, king safety, piece development). Currently the AI only counts material.
- Depth-to-mate bonus (ai.rs) — Return -30000 + (20 - depth) at checkmate so the AI prefers faster mates.

Tests

- game.rs tests — test_initial_status, test_update_status_checkmate, test_update_status_stalemate (regression for the bug just fixed), test_update_status_draw_insufficient,
  test_apply_move_increments_history, test_reset_clears_history, test_is_game_over
- board.rs tests — Corner spot checks (a1, h8, flipped), plus a full 64-square × 2-orientation round-trip test for square_rect → pixel_to_square
- ai.rs tests — test_mate_in_one, test_avoids_obvious_blunder, test_captures_hanging_piece, test_negamax_checkmate_score

Optional Features

- Undo/Takeback — Store position_history: Vec<Chess>, add takeback() method, add button to side panel
- Play as Black — White/Black toggle that resets the game and triggers AI to move first
- Iterative deepening — Run best_move at depth 1..N, return best from deepest completed search (better responsiveness at higher depths)
