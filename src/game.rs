use shakmaty::{Chess, Color, EnPassantMode, KnownOutcome, Move, Outcome, Position, Role, Square};
use shakmaty::zobrist::Zobrist64;

#[derive(Debug, Clone, PartialEq)]
pub enum GameStatus {
    Playing,
    Check,
    Checkmate(Color),
    Stalemate,
    Draw,
}

pub struct GameState {
    pub position: Chess,
    pub selected: Option<Square>,
    pub legal_moves_from: Vec<Move>,
    pub human_color: Color,
    pub ai_thinking: bool,
    pub pending_promotion: Option<(Square, Square)>,
    pub status: GameStatus,
    pub move_history: Vec<String>,
    pub flipped: bool,
    pub ai_depth: u32,
    /// The most recently played move (human or AI), used for board highlighting.
    pub last_move: Option<Move>,
    /// Zobrist64 hashes of every position that has appeared in the game,
    /// including the starting position.  Used to detect three-fold repetition
    /// in `update_status` and to seed the engine's repetition history in
    /// `best_move` so it avoids revisiting them.
    pub position_hashes: Vec<u64>,
}

impl GameState {
    fn zobrist_hash(pos: &Chess) -> u64 {
        u64::from(pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal))
    }

    pub fn new() -> Self {
        let position = Chess::default();
        let initial_hash = Self::zobrist_hash(&position);
        Self {
            position,
            selected: None,
            legal_moves_from: Vec::new(),
            human_color: Color::White,
            ai_thinking: false,
            pending_promotion: None,
            status: GameStatus::Playing,
            move_history: Vec::new(),
            flipped: false,
            ai_depth: 4,
            last_move: None,
            position_hashes: vec![initial_hash],
        }
    }

    pub fn reset(&mut self) {
        self.position = Chess::default();
        self.selected = None;
        self.legal_moves_from.clear();
        self.ai_thinking = false;
        self.pending_promotion = None;
        self.status = GameStatus::Playing;
        self.move_history.clear();
        self.last_move = None;
        self.position_hashes = vec![Self::zobrist_hash(&self.position)];
    }

    pub fn update_status(&mut self) {
        // Three-fold repetition: if the current position hash appears at least
        // three times in the game history (including now), it is a draw.
        // `position_hashes` is populated by `apply_move` before this call, so
        // the count already includes the current occurrence.
        let current_hash = Self::zobrist_hash(&self.position);
        let occurrences = self.position_hashes.iter().filter(|&&h| h == current_hash).count();
        if occurrences >= 3 {
            self.status = GameStatus::Draw;
            return;
        }

        // Compute legal moves exactly once.  The old code called is_stalemate()
        // (which generates legal moves internally) and then outcome() (which also
        // generates legal moves to detect checkmate), doubling the work per move.
        let legal = self.position.legal_moves();
        if legal.is_empty() {
            // Terminal position: checkmate or stalemate.
            self.status = if self.position.is_check() {
                // The player to move is in check with no escape → their opponent wins.
                GameStatus::Checkmate(!self.position.turn())
            } else {
                GameStatus::Stalemate
            };
            return;
        }
        // Position has legal moves: check for drawn positions that don't require
        // legal-move generation (insufficient material, 50-move rule, etc.).
        match self.position.outcome() {
            Outcome::Known(KnownOutcome::Draw) => {
                self.status = GameStatus::Draw;
            }
            _ => {
                self.status = if self.position.is_check() {
                    GameStatus::Check
                } else {
                    GameStatus::Playing
                };
            }
        }
    }

    pub fn select_square(&mut self, sq: Square) {
        // Fast path: deselect before paying for legal_moves().
        if self.selected == Some(sq) {
            self.selected = None;
            self.legal_moves_from.clear();
            return;
        }

        let legal = self.position.legal_moves();

        // If piece already selected, try to play move
        if let Some(from) = self.selected {

            // Check if there's a promotion move to this square
            let needs_promotion = legal.iter().any(|m| {
                if let Move::Normal { from: f, to, promotion, .. } = m {
                    *f == from && *to == sq && promotion.is_some()
                } else {
                    false
                }
            });

            if needs_promotion {
                self.pending_promotion = Some((from, sq));
                self.selected = None;
                self.legal_moves_from.clear();
                return;
            }

            // Try to find the move
            let mv = legal.iter().find(|m| match m {
                Move::Normal { from: f, to, promotion: None, .. } => *f == from && *to == sq,
                Move::Castle { king, rook } => {
                    *king == from && {
                        let turn = self.position.turn();
                        let (ks, qs) = if turn == Color::White {
                            (Square::G1, Square::C1)
                        } else {
                            (Square::G8, Square::C8)
                        };
                        // Cross-contamination fix: the original `sq == ks || sq == qs`
                        // appeared in BOTH castle arms, so clicking the queenside king
                        // destination (C1/C8) could match a kingside castle move, and
                        // vice-versa.  Check each side independently so only the
                        // correct castle fires.
                        //
                        // Accepted clicks:
                        //  • King destination for this side (ks or qs)
                        //  • Rook's starting square (*rook)
                        // Rook destinations (F1/D1) are NOT accepted: those squares
                        // are also reachable by normal one-square king moves, so they
                        // would be matched first as Move::Normal and never reach here.
                        let is_kingside = rook.file() > king.file();
                        if is_kingside {
                            sq == ks || *rook == sq
                        } else {
                            sq == qs || *rook == sq
                        }
                    }
                }
                Move::EnPassant { from: f, to, .. } => *f == from && *to == sq,
                _ => false,
            });

            if let Some(mv) = mv.cloned() {
                self.apply_move(mv);
                return;
            }
        }

        // Select new square if it has a piece of current turn's color
        let board = self.position.board();
        if let Some(piece) = board.piece_at(sq) {
            if piece.color == self.position.turn() {
                self.selected = Some(sq);
                self.legal_moves_from = legal
                    .into_iter()
                    .filter(|m| match m {
                        Move::Normal { from, .. } => *from == sq,
                        Move::EnPassant { from, .. } => *from == sq,
                        Move::Castle { king, .. } => *king == sq,
                        _ => false,
                    })
                    .collect();
                return;
            }
        }

        // If a piece was already selected and the clicked square is neither a
        // legal move destination nor a friendly piece to re-select, keep the
        // existing selection active.  Only clear when there was no prior
        // selection (clicking an empty/enemy square with nothing selected is
        // already a no-op since self.selected is already None).
        if self.selected.is_some() {
            return;
        }
        self.selected = None;
        self.legal_moves_from.clear();
    }

    pub fn apply_move(&mut self, mv: Move) {
        let notation = self.move_to_notation(&mv);
        self.last_move = Some(mv.clone());
        self.position = std::mem::replace(&mut self.position, Chess::default())
            .play(mv)
            .expect("legal move");
        self.selected = None;
        self.legal_moves_from.clear();
        self.move_history.push(notation);
        // Record the new position's hash for three-fold repetition detection
        // and for seeding the engine's search history.
        self.position_hashes.push(Self::zobrist_hash(&self.position));
        self.update_status();
    }

    /// Cancel a pending promotion, restoring selection to the pawn's from-square.
    pub fn cancel_promotion(&mut self) {
        if let Some((from, _)) = self.pending_promotion {
            self.pending_promotion = None;
            let legal = self.position.legal_moves();
            self.selected = Some(from);
            self.legal_moves_from = legal
                .into_iter()
                .filter(|m| match m {
                    Move::Normal { from: f, .. } => *f == from,
                    Move::EnPassant { from: f, .. } => *f == from,
                    Move::Castle { king, .. } => *king == from,
                    _ => false,
                })
                .collect();
        }
    }

    pub fn apply_promotion(&mut self, from: Square, to: Square, role: Role) {
        let legal = self.position.legal_moves();
        let mv = legal.into_iter().find(|m| {
            if let Move::Normal { from: f, to: t, promotion: Some(r), .. } = m {
                *f == from && *t == to && *r == role
            } else {
                false
            }
        });
        if let Some(mv) = mv {
            self.apply_move(mv);
            self.pending_promotion = None;
        }
    }

    fn move_to_notation(&self, mv: &Move) -> String {
        match mv {
            Move::Normal { from, to, promotion, .. } => {
                let promo = promotion
                    .map(|r| match r {
                        Role::Queen => "Q",
                        Role::Rook => "R",
                        Role::Bishop => "B",
                        Role::Knight => "N",
                        _ => "",
                    })
                    .unwrap_or("");
                format!("{}{}{}", from, to, promo)
            }
            Move::EnPassant { from, to, .. } => format!("{}{}ep", from, to),
            Move::Castle { king, rook } => {
                if rook.file() > king.file() {
                    "O-O".to_string()
                } else {
                    "O-O-O".to_string()
                }
            }
            Move::Put { .. } => "put".to_string(),
        }
    }

    pub fn is_game_over(&self) -> bool {
        matches!(self.status, GameStatus::Checkmate(_) | GameStatus::Stalemate | GameStatus::Draw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{fen::Fen, CastlingMode};

    fn pos_from_fen(s: &str) -> Chess {
        let fen: Fen = s.parse().unwrap();
        fen.into_position(CastlingMode::Standard).unwrap()
    }

    #[test]
    fn test_initial_status() {
        assert_eq!(GameState::new().status, GameStatus::Playing);
    }

    #[test]
    fn test_update_status_checkmate() {
        // Fool's mate: black queen on h4 delivers checkmate, white to move.
        let mut gs = GameState::new();
        gs.position = pos_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
        gs.update_status();
        assert_eq!(gs.status, GameStatus::Checkmate(Color::Black));
    }

    #[test]
    fn test_update_status_stalemate() {
        // Black king on a8, white queen b6, white king c6: black to move, no legal moves, not in check.
        let mut gs = GameState::new();
        gs.position = pos_from_fen("k7/8/1QK5/8/8/8/8/8 b - - 0 1");
        gs.update_status();
        assert_eq!(gs.status, GameStatus::Stalemate);
    }

    #[test]
    fn test_apply_move_increments_history() {
        let mut gs = GameState::new();
        let mv = gs.position.legal_moves().into_iter().next().unwrap();
        gs.apply_move(mv);
        assert_eq!(gs.move_history.len(), 1);
    }

    #[test]
    fn test_reset_clears_history() {
        let mut gs = GameState::new();
        let mv = gs.position.legal_moves().into_iter().next().unwrap();
        gs.apply_move(mv);
        assert_eq!(gs.move_history.len(), 1);
        gs.reset();
        assert!(gs.move_history.is_empty());
        assert_eq!(gs.status, GameStatus::Playing);
    }

    #[test]
    fn test_select_square_deselects_on_reclicking_same_piece() {
        let mut gs = GameState::new();
        // First click: select the e2 pawn.
        gs.select_square(Square::E2);
        assert_eq!(gs.selected, Some(Square::E2));
        assert!(!gs.legal_moves_from.is_empty(), "should have legal moves after selection");
        // Second click on the same square: should deselect.
        gs.select_square(Square::E2);
        assert!(gs.selected.is_none(), "clicking selected piece again must deselect it");
        assert!(gs.legal_moves_from.is_empty(), "legal_moves_from must be cleared on deselect");
    }

    #[test]
    fn test_select_square_different_piece_reselects() {
        let mut gs = GameState::new();
        gs.select_square(Square::E2);
        assert_eq!(gs.selected, Some(Square::E2));
        // Clicking a different own piece re-selects it.
        gs.select_square(Square::D2);
        assert_eq!(gs.selected, Some(Square::D2), "clicking another piece should reselect");
    }

    #[test]
    fn test_cancel_promotion_clears_pending() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1");
        gs.pending_promotion = Some((Square::A7, Square::A8));
        gs.cancel_promotion();
        assert!(gs.pending_promotion.is_none(), "cancel must clear pending_promotion");
    }

    #[test]
    fn test_cancel_promotion_restores_selection() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1");
        gs.pending_promotion = Some((Square::A7, Square::A8));
        gs.cancel_promotion();
        assert_eq!(gs.selected, Some(Square::A7), "cancel must re-select the pawn square");
        assert!(!gs.legal_moves_from.is_empty(), "legal_moves_from must be repopulated");
    }

    #[test]
    fn test_select_square_deselect_does_not_require_legal_moves() {
        // Deselecting by clicking the same square must work even in a position
        // where legal_moves() would be skipped (fast path before legal_moves()).
        // We verify the fast path by calling select → deselect and confirming state.
        let mut gs = GameState::new();
        gs.select_square(Square::E2); // select
        assert_eq!(gs.selected, Some(Square::E2));
        gs.select_square(Square::E2); // deselect via fast path
        assert!(gs.selected.is_none(), "fast-path deselect must clear selected");
        assert!(gs.legal_moves_from.is_empty(), "fast-path deselect must clear moves");
    }

    #[test]
    fn test_apply_move_position_updates_correctly() {
        // Verify apply_move (which now uses mem::replace instead of clone)
        // still correctly advances the position.
        let mut gs = GameState::new();
        let mv = gs.position.legal_moves().into_iter()
            .find(|m| matches!(m, Move::Normal { from, to, .. }
                if *from == Square::E2 && *to == Square::E4))
            .expect("e2e4 must be legal");
        gs.apply_move(mv);
        // After 1.e4 the pawn should be on e4, not e2.
        let board = gs.position.board();
        assert!(board.piece_at(Square::E4).is_some(), "pawn must be on e4 after e2-e4");
        assert!(board.piece_at(Square::E2).is_none(), "e2 must be empty after e2-e4");
    }

    #[test]
    fn test_cancel_promotion_no_op_when_none() {
        let mut gs = GameState::new();
        // No pending promotion — cancel should be a harmless no-op.
        gs.cancel_promotion();
        assert!(gs.pending_promotion.is_none());
        assert!(gs.selected.is_none());
    }

    #[test]
    fn test_apply_promotion_clears_pending_on_success() {
        // White pawn on e7, black king on e8 pushed aside. Promote to queen.
        // FEN: white pawn on e7, kings far away, black to not interfere.
        let mut gs = GameState::new();
        // White pawn on a7, kings on corners, black king on h8.
        gs.position = pos_from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1");
        gs.pending_promotion = Some((Square::A7, Square::A8));
        gs.apply_promotion(Square::A7, Square::A8, Role::Queen);
        assert!(gs.pending_promotion.is_none(), "pending_promotion must be cleared after successful promotion");
        assert_eq!(gs.move_history.len(), 1, "promotion must be recorded in history");
    }

    #[test]
    fn test_apply_promotion_keeps_pending_on_invalid() {
        // Supply a role that doesn't match any legal promotion to verify
        // pending_promotion is NOT cleared (user can retry).
        let mut gs = GameState::new();
        gs.position = pos_from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1");
        gs.pending_promotion = Some((Square::A7, Square::A8));
        // King is not a valid promotion piece.
        gs.apply_promotion(Square::A7, Square::A8, Role::King);
        assert!(gs.pending_promotion.is_some(), "pending_promotion must remain set when promotion fails");
    }

    // ── Bug-fix: last_move tracking ──────────────────────────────────────────

    #[test]
    fn test_last_move_none_on_new_game() {
        let gs = GameState::new();
        assert!(gs.last_move.is_none(), "last_move must be None at game start");
    }

    #[test]
    fn test_last_move_set_after_apply_move() {
        let mut gs = GameState::new();
        let mv = gs.position.legal_moves().into_iter()
            .find(|m| matches!(m, Move::Normal { from, to, .. }
                if *from == Square::E2 && *to == Square::E4))
            .expect("e2e4 must be legal");
        gs.apply_move(mv.clone());
        assert_eq!(gs.last_move, Some(mv), "last_move must be set to the played move");
    }

    #[test]
    fn test_last_move_updated_on_second_move() {
        let mut gs = GameState::new();
        // Play e2-e4
        let mv1 = gs.position.legal_moves().into_iter()
            .find(|m| matches!(m, Move::Normal { from, to, .. }
                if *from == Square::E2 && *to == Square::E4))
            .unwrap();
        gs.apply_move(mv1);
        // Play e7-e5
        let mv2 = gs.position.legal_moves().into_iter()
            .find(|m| matches!(m, Move::Normal { from, to, .. }
                if *from == Square::E7 && *to == Square::E5))
            .unwrap();
        gs.apply_move(mv2.clone());
        assert_eq!(gs.last_move, Some(mv2), "last_move must reflect the most recent move");
    }

    #[test]
    fn test_last_move_cleared_on_reset() {
        let mut gs = GameState::new();
        let mv = gs.position.legal_moves().into_iter().next().unwrap();
        gs.apply_move(mv);
        assert!(gs.last_move.is_some(), "last_move must be set before reset");
        gs.reset();
        assert!(gs.last_move.is_none(), "last_move must be cleared by reset");
    }

    #[test]
    fn test_last_move_set_after_promotion() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1");
        gs.pending_promotion = Some((Square::A7, Square::A8));
        gs.apply_promotion(Square::A7, Square::A8, Role::Queen);
        assert!(
            gs.last_move.is_some(),
            "last_move must be set after a successful promotion"
        );
        if let Some(Move::Normal { from, to, promotion: Some(role), .. }) = gs.last_move {
            assert_eq!(from, Square::A7);
            assert_eq!(to, Square::A8);
            assert_eq!(role, Role::Queen);
        } else {
            panic!("last_move must be the promotion move");
        }
    }

    #[test]
    fn test_last_move_not_set_after_failed_promotion() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1");
        gs.pending_promotion = Some((Square::A7, Square::A8));
        gs.apply_promotion(Square::A7, Square::A8, Role::King); // invalid
        assert!(
            gs.last_move.is_none(),
            "last_move must not be set after a failed promotion"
        );
    }

    // ── Bug-fix: castle cross-contamination ──────────────────────────────────
    //
    // The original code used `sq == ks || sq == qs` in BOTH castle arms, so
    // clicking G1 (kingside king-dest) could fire a queenside castle, and clicking
    // C1 (queenside king-dest) could fire a kingside castle, depending on which
    // Move::Castle appeared first in legal_moves().  The fix checks each side
    // independently so only the correct castle is triggered.

    // Helper: attempt a castle by selecting the king then clicking `target`.
    // Returns the post-move board state.
    fn attempt_castle(gs: &mut GameState, king_sq: Square, target: Square) {
        gs.select_square(king_sq);
        assert_eq!(gs.selected, Some(king_sq), "king must be selected before castle");
        gs.select_square(target);
    }

    // --- Regression: basic castle via king destination still works ---

    #[test]
    fn test_castle_white_kingside_via_king_destination() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
        attempt_castle(&mut gs, Square::E1, Square::G1);
        assert!(gs.position.board().piece_at(Square::G1).is_some(), "king must be on G1");
        assert!(gs.position.board().piece_at(Square::F1).is_some(), "rook must be on F1");
    }

    #[test]
    fn test_castle_white_kingside_via_rook_start() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
        attempt_castle(&mut gs, Square::E1, Square::H1);
        assert!(gs.position.board().piece_at(Square::G1).is_some(), "king must be on G1");
        assert!(gs.position.board().piece_at(Square::F1).is_some(), "rook must be on F1");
    }

    #[test]
    fn test_castle_white_queenside_via_king_destination() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
        attempt_castle(&mut gs, Square::E1, Square::C1);
        assert!(gs.position.board().piece_at(Square::C1).is_some(), "king must be on C1");
        assert!(gs.position.board().piece_at(Square::D1).is_some(), "rook must be on D1");
    }

    #[test]
    fn test_castle_white_queenside_via_rook_start() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
        attempt_castle(&mut gs, Square::E1, Square::A1);
        assert!(gs.position.board().piece_at(Square::C1).is_some(), "king must be on C1");
        assert!(gs.position.board().piece_at(Square::D1).is_some(), "rook must be on D1");
    }

    // --- Cross-contamination bug fix tests ---

    // Only kingside available: clicking C1 (queenside king-dest) must NOT fire
    // any castle — in the original code, `sq == qs` matched the kingside castle arm,
    // causing an incorrect kingside castle.
    #[test]
    fn test_castle_only_kingside_click_queenside_dest_does_not_castle() {
        let mut gs = GameState::new();
        // Only kingside rook present → only kingside castle legal.
        gs.position = pos_from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
        let history_before = gs.move_history.len();
        attempt_castle(&mut gs, Square::E1, Square::C1);
        // C1 is two squares away — no normal king move there, and no queenside
        // castle → the click must be a no-op (deselects or stays selected).
        assert_eq!(
            gs.move_history.len(), history_before,
            "clicking C1 with only kingside available must not castle"
        );
        // King must remain on E1.
        assert!(
            gs.position.board().piece_at(Square::E1).is_some(),
            "king must still be on E1"
        );
    }

    // Only queenside available: clicking G1 (kingside king-dest) must NOT fire any castle.
    #[test]
    fn test_castle_only_queenside_click_kingside_dest_does_not_castle() {
        let mut gs = GameState::new();
        // Only queenside rook present → only queenside castle legal.
        gs.position = pos_from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
        let history_before = gs.move_history.len();
        attempt_castle(&mut gs, Square::E1, Square::G1);
        assert_eq!(
            gs.move_history.len(), history_before,
            "clicking G1 with only queenside available must not castle"
        );
        assert!(
            gs.position.board().piece_at(Square::E1).is_some(),
            "king must still be on E1"
        );
    }

    // Both rights: clicking G1 fires kingside, NOT queenside.
    #[test]
    fn test_castle_both_rights_click_g1_fires_kingside() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        attempt_castle(&mut gs, Square::E1, Square::G1);
        let board = gs.position.board();
        assert!(board.piece_at(Square::G1).is_some(), "king must be on G1 (kingside)");
        assert!(board.piece_at(Square::C1).is_none(), "queenside must NOT have fired");
    }

    // Both rights: clicking C1 fires queenside, NOT kingside.
    #[test]
    fn test_castle_both_rights_click_c1_fires_queenside() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        attempt_castle(&mut gs, Square::E1, Square::C1);
        let board = gs.position.board();
        assert!(board.piece_at(Square::C1).is_some(), "king must be on C1 (queenside)");
        assert!(board.piece_at(Square::G1).is_none(), "kingside must NOT have fired");
    }

    // Both rights: clicking H1 (kingside rook start) fires kingside.
    #[test]
    fn test_castle_both_rights_click_h1_fires_kingside() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        attempt_castle(&mut gs, Square::E1, Square::H1);
        let board = gs.position.board();
        assert!(board.piece_at(Square::G1).is_some(), "king must be on G1 (kingside)");
        assert!(board.piece_at(Square::C1).is_none(), "queenside must NOT have fired");
    }

    // Both rights: clicking A1 (queenside rook start) fires queenside.
    #[test]
    fn test_castle_both_rights_click_a1_fires_queenside() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        attempt_castle(&mut gs, Square::E1, Square::A1);
        let board = gs.position.board();
        assert!(board.piece_at(Square::C1).is_some(), "king must be on C1 (queenside)");
        assert!(board.piece_at(Square::G1).is_none(), "kingside must NOT have fired");
    }

    // Black: both rights, click G8 → kingside.
    #[test]
    fn test_castle_black_both_rights_click_g8_fires_kingside() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1");
        gs.human_color = Color::Black;
        attempt_castle(&mut gs, Square::E8, Square::G8);
        let board = gs.position.board();
        assert!(board.piece_at(Square::G8).is_some(), "black king must be on G8");
        assert!(board.piece_at(Square::C8).is_none(), "black queenside must NOT have fired");
    }

    // Black: both rights, click C8 → queenside.
    #[test]
    fn test_castle_black_both_rights_click_c8_fires_queenside() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1");
        gs.human_color = Color::Black;
        attempt_castle(&mut gs, Square::E8, Square::C8);
        let board = gs.position.board();
        assert!(board.piece_at(Square::C8).is_some(), "black king must be on C8");
        assert!(board.piece_at(Square::G8).is_none(), "black kingside must NOT have fired");
    }

    // ── Bug-fix: AI trigger gated on pending_promotion.is_none() ─────────────

    // After a SUCCESSFUL promotion the dialog is gone (pending = None).
    // app.rs should trigger AI in this case.
    #[test]
    fn test_promotion_success_clears_pending_ai_can_trigger() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1");
        gs.pending_promotion = Some((Square::A7, Square::A8));
        gs.apply_promotion(Square::A7, Square::A8, Role::Queen);
        assert!(
            gs.pending_promotion.is_none(),
            "successful promotion: pending_promotion must be None so AI is triggered"
        );
    }

    // After a FAILED promotion the dialog stays open (pending = Some).
    // app.rs must NOT trigger AI in this case (was the bug).
    #[test]
    fn test_promotion_failure_keeps_pending_ai_must_not_trigger() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1");
        gs.pending_promotion = Some((Square::A7, Square::A8));
        gs.apply_promotion(Square::A7, Square::A8, Role::King); // invalid piece
        // is_game_over() is false here — the old buggy code would have triggered AI.
        assert!(!gs.is_game_over(), "game is not over (old code would have triggered AI here)");
        assert!(
            gs.pending_promotion.is_some(),
            "failed promotion: pending_promotion must remain Some, blocking AI trigger"
        );
    }

    // The position must be unchanged after a failed promotion.
    #[test]
    fn test_promotion_failure_does_not_advance_position() {
        let mut gs = GameState::new();
        gs.position = pos_from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1");
        gs.pending_promotion = Some((Square::A7, Square::A8));
        let history_len_before = gs.move_history.len();
        gs.apply_promotion(Square::A7, Square::A8, Role::King);
        assert_eq!(
            gs.move_history.len(),
            history_len_before,
            "failed promotion must not record a move"
        );
        // Pawn must still be on A7.
        assert!(
            gs.position.board().piece_at(Square::A7).is_some(),
            "pawn must still be on A7 after failed promotion"
        );
    }

    // ── update_status single legal-move computation (bug-fix regression) ──────
    //
    // Before the fix, update_status called is_stalemate() (which generates legal
    // moves internally) and then outcome() (which regenerates legal moves for
    // checkmate detection) — doubling the work after every move.
    //
    // The fix computes legal_moves() exactly once, branches on empty+is_check for
    // checkmate, empty+!is_check for stalemate, and falls through to outcome() only
    // for positions that still have legal moves (where outcome() no longer needs to
    // regenerate them for terminal detection).
    //
    // Tests verify all five GameStatus variants under the new code path.

    #[test]
    fn test_update_status_playing_normal_position() {
        // Starting position: legal moves exist, not in check, no draw condition.
        let mut gs = GameState::new();
        gs.update_status();
        assert_eq!(gs.status, GameStatus::Playing,
            "starting position must be Playing after update_status");
    }

    #[test]
    fn test_update_status_check_when_in_check() {
        // White king on e1, black queen on e2 — white is in check but has evasions.
        let mut gs = GameState::new();
        gs.position = pos_from_fen("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1");
        gs.update_status();
        assert_eq!(gs.status, GameStatus::Check,
            "position with king in check must yield GameStatus::Check");
    }

    #[test]
    fn test_update_status_checkmate_black_wins() {
        // Fool's mate: black queen on h4, white to move — no legal moves, in check.
        // Winner = black = !Color::White.
        let mut gs = GameState::new();
        gs.position = pos_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
        gs.update_status();
        assert_eq!(gs.status, GameStatus::Checkmate(Color::Black),
            "black delivers checkmate → winner must be Color::Black");
    }

    #[test]
    fn test_update_status_checkmate_white_wins() {
        // Back-rank mate: black Ka8 checkmated, black to move — no legal moves, in check.
        // Winner = white = !Color::Black.
        let mut gs = GameState::new();
        gs.position = pos_from_fen("k1Q5/8/K7/8/8/8/8/8 b - - 0 1");
        gs.update_status();
        assert_eq!(gs.status, GameStatus::Checkmate(Color::White),
            "white delivers checkmate → winner must be Color::White");
    }

    #[test]
    fn test_update_status_stalemate_no_legal_moves_no_check() {
        // Black king on a8, white Qb6, Kc6: black to move, no legal moves, not in check.
        let mut gs = GameState::new();
        gs.position = pos_from_fen("k7/8/1QK5/8/8/8/8/8 b - - 0 1");
        gs.update_status();
        assert_eq!(gs.status, GameStatus::Stalemate,
            "no legal moves and not in check must yield Stalemate, not Draw");
    }

    #[test]
    fn test_update_status_draw_insufficient_material() {
        // K vs K: legal moves exist but shakmaty reports outcome as Draw.
        let mut gs = GameState::new();
        gs.position = pos_from_fen("8/8/4k3/8/8/4K3/8/8 w - - 0 1");
        gs.update_status();
        assert_eq!(gs.status, GameStatus::Draw,
            "K vs K must yield GameStatus::Draw (insufficient material)");
    }

    #[test]
    fn test_update_status_stalemate_not_conflated_with_draw() {
        // Regression: before the fix, if stalemate were to fall through to outcome()
        // and outcome() returned Draw for it, the status would be Draw not Stalemate.
        // Verify the status is specifically Stalemate, not the generic Draw.
        let mut gs = GameState::new();
        gs.position = pos_from_fen("k7/8/1QK5/8/8/8/8/8 b - - 0 1");
        gs.update_status();
        assert_ne!(gs.status, GameStatus::Draw,
            "stalemate must show as Stalemate, not conflated with Draw");
        assert_eq!(gs.status, GameStatus::Stalemate);
    }

    #[test]
    fn test_is_game_over() {
        let mut gs = GameState::new();
        assert!(!gs.is_game_over());
        gs.status = GameStatus::Check;
        assert!(!gs.is_game_over());
        gs.status = GameStatus::Stalemate;
        assert!(gs.is_game_over());
        gs.status = GameStatus::Checkmate(Color::White);
        assert!(gs.is_game_over());
        gs.status = GameStatus::Draw;
        assert!(gs.is_game_over());
    }

    // ── Repetition detection ──────────────────────────────────────────────────

    /// Apply a Normal (non-promotion) move by searching the legal move list.
    fn apply_normal(gs: &mut GameState, from: Square, to: Square) {
        let legal = gs.position.legal_moves();
        let mv = legal
            .into_iter()
            .find(|m| matches!(m, Move::Normal { from: f, to: t, promotion: None, .. } if *f == from && *t == to))
            .expect("move must be legal");
        gs.apply_move(mv);
    }

    /// `GameState::new()` seeds `position_hashes` with the starting position's hash.
    #[test]
    fn test_position_hashes_seeded_on_new() {
        let gs = GameState::new();
        assert_eq!(gs.position_hashes.len(), 1, "new game must seed exactly one hash");
        let expected = GameState::zobrist_hash(&gs.position);
        assert_eq!(gs.position_hashes[0], expected, "initial hash must match starting position");
    }

    /// `apply_move` appends the new position's hash after each move.
    #[test]
    fn test_position_hashes_grow_with_moves() {
        let mut gs = GameState::new();
        assert_eq!(gs.position_hashes.len(), 1);
        apply_normal(&mut gs, Square::E2, Square::E4);
        assert_eq!(gs.position_hashes.len(), 2, "one move must add one hash");
        let expected = GameState::zobrist_hash(&gs.position);
        assert_eq!(*gs.position_hashes.last().unwrap(), expected);
    }

    /// `reset()` clears the hash list and re-seeds it with the starting position.
    #[test]
    fn test_position_hashes_reset_on_reset() {
        let mut gs = GameState::new();
        apply_normal(&mut gs, Square::E2, Square::E4);
        apply_normal(&mut gs, Square::E7, Square::E5);
        assert_eq!(gs.position_hashes.len(), 3);
        gs.reset();
        assert_eq!(gs.position_hashes.len(), 1, "reset must leave exactly one hash");
        let expected = GameState::zobrist_hash(&gs.position);
        assert_eq!(gs.position_hashes[0], expected, "reset hash must match fresh starting position");
    }

    /// `update_status` directly detects three-fold repetition via the hash list.
    /// Hash injection simulates previous visits without replaying actual moves.
    #[test]
    fn test_threefold_repetition_detected_as_draw() {
        let mut gs = GameState::new();
        // Starting position appears once in position_hashes already.
        let hash = gs.position_hashes[0];
        // Inject a 2nd and 3rd occurrence (interleaved with distinct intermediate hashes).
        gs.position_hashes.push(0xdead_beef_cafe_babe_u64); // a different position
        gs.position_hashes.push(hash);                       // 2nd occurrence
        gs.position_hashes.push(0x1234_5678_9abc_def0_u64); // a different position
        gs.position_hashes.push(hash);                       // 3rd occurrence
        // gs.position is still the starting position → current hash = hash.
        gs.update_status();
        assert_eq!(gs.status, GameStatus::Draw, "three occurrences must declare Draw");
    }

    /// Two-fold repetition must NOT trigger a draw — only the third matters.
    #[test]
    fn test_twofold_repetition_is_not_draw() {
        let mut gs = GameState::new();
        let hash = gs.position_hashes[0];
        gs.position_hashes.push(0xdead_beef_cafe_babe_u64); // a different position
        gs.position_hashes.push(hash);                       // 2nd occurrence only
        gs.update_status();
        assert_ne!(gs.status, GameStatus::Draw, "two occurrences must NOT declare Draw");
        assert_eq!(gs.status, GameStatus::Playing, "two occurrences must remain Playing");
    }

    /// Integration test: actual knight moves that cycle the starting position
    /// create a three-fold repetition after two complete cycles.
    ///
    /// Cycle: 1.Nf3 1...Nf6 2.Ng1 2...Ng8 → starting position (2nd occurrence).
    /// Repeat once more → 3rd occurrence → Draw.
    ///
    /// Knight moves do not affect castling rights, en passant, or pieces, so
    /// the Zobrist hash after each full cycle matches the initial hash exactly.
    #[test]
    fn test_threefold_repetition_from_knight_cycle() {
        let mut gs = GameState::new();
        assert_eq!(gs.status, GameStatus::Playing);

        // First cycle: 2nd occurrence of the starting position.
        apply_normal(&mut gs, Square::G1, Square::F3); // 1.Nf3
        apply_normal(&mut gs, Square::G8, Square::F6); // 1...Nf6
        apply_normal(&mut gs, Square::F3, Square::G1); // 2.Ng1
        apply_normal(&mut gs, Square::F6, Square::G8); // 2...Ng8 → 2nd occurrence
        assert_eq!(gs.status, GameStatus::Playing, "2nd occurrence must not yet be Draw");

        // Second cycle: 3rd occurrence → Draw by three-fold repetition.
        apply_normal(&mut gs, Square::G1, Square::F3);
        apply_normal(&mut gs, Square::G8, Square::F6);
        apply_normal(&mut gs, Square::F3, Square::G1);
        apply_normal(&mut gs, Square::F6, Square::G8); // 3rd occurrence
        assert_eq!(gs.status, GameStatus::Draw, "3rd occurrence must be Draw");
    }

    // ── select_square: keep selection on illegal click (bug-fix) ──────────────
    //
    // Before the fix, clicking any square that is neither a legal move destination
    // nor a friendly piece while a piece was selected would unconditionally clear
    // `self.selected`.  The fix preserves the selection so the user does not have
    // to re-select their piece after a mis-click.

    /// Clicking an empty unreachable square while a piece is selected must keep
    /// the selection active (not clear it).
    #[test]
    fn test_select_square_keeps_selection_on_empty_square_click() {
        let mut gs = GameState::new(); // starting position, White to move
        // Select the e2 pawn.
        gs.select_square(Square::E2);
        assert_eq!(gs.selected, Some(Square::E2), "e2 pawn should be selected");

        // Click e6 — empty in the starting position, not a legal destination
        // for the e2 pawn (which can only reach e3 or e4).
        gs.select_square(Square::E6);

        assert_eq!(
            gs.selected,
            Some(Square::E2),
            "selection must stay on e2 after clicking empty unreachable square e6"
        );
    }

    /// Clicking an enemy piece that cannot be captured while a piece is selected
    /// must keep the selection active.
    #[test]
    fn test_select_square_keeps_selection_on_enemy_piece_click() {
        let mut gs = GameState::new(); // starting position, White to move
        // Select the e2 pawn.
        gs.select_square(Square::E2);
        assert_eq!(gs.selected, Some(Square::E2));

        // Click e7 — enemy pawn, not capturable by the e2 pawn.
        gs.select_square(Square::E7);

        assert_eq!(
            gs.selected,
            Some(Square::E2),
            "selection must stay on e2 after clicking unreachable enemy piece on e7"
        );
    }

    /// Clicking the already-selected square deselects (regression: must still work).
    #[test]
    fn test_select_square_deselects_on_same_square_click() {
        let mut gs = GameState::new();
        gs.select_square(Square::E2);
        assert_eq!(gs.selected, Some(Square::E2));
        gs.select_square(Square::E2); // second click on same square
        assert_eq!(gs.selected, None, "clicking selected square again must deselect it");
    }

    /// Clicking a different friendly piece while one is selected re-selects it
    /// (regression: must still work).
    #[test]
    fn test_select_square_reselects_on_different_friendly_piece() {
        let mut gs = GameState::new();
        gs.select_square(Square::E2);
        assert_eq!(gs.selected, Some(Square::E2));
        gs.select_square(Square::D2); // different friendly pawn
        assert_eq!(
            gs.selected,
            Some(Square::D2),
            "clicking another friendly piece must re-select to that piece"
        );
    }
}
