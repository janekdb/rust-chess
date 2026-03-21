use egui::{Color32, CornerRadius, FontId, Painter, Pos2, Rect, Stroke, Vec2};
use shakmaty::{Color as ChessColor, File, Move, Piece, Position, Rank, Role, Square};

pub const SQUARE_SIZE: f32 = 70.0;
/// Width of the label margin strips (left for rank numbers, bottom for file letters).
pub const LABEL_MARGIN: f32 = 14.0;
/// Playable 8×8 grid size in pixels (excludes label margins).
pub const INNER_SIZE: f32 = SQUARE_SIZE * 8.0;
/// Total rendered board area: grid + left rank-label strip + bottom file-label strip.
pub const BOARD_SIZE: f32 = INNER_SIZE + LABEL_MARGIN;

const LIGHT: Color32 = Color32::from_rgb(240, 217, 181);
const DARK: Color32 = Color32::from_rgb(181, 136, 99);
const SELECTED: Color32 = Color32::from_rgba_premultiplied(100, 200, 100, 180);
const HINT: Color32 = Color32::from_rgba_premultiplied(50, 150, 50, 120);
const LAST_MOVE: Color32 = Color32::from_rgba_premultiplied(200, 200, 50, 100);

fn piece_char(piece: Piece) -> &'static str {
    match (piece.color, piece.role) {
        (ChessColor::White, Role::King) => "♔",
        (ChessColor::White, Role::Queen) => "♕",
        (ChessColor::White, Role::Rook) => "♖",
        (ChessColor::White, Role::Bishop) => "♗",
        (ChessColor::White, Role::Knight) => "♘",
        (ChessColor::White, Role::Pawn) => "♙",
        (ChessColor::Black, Role::King) => "♚",
        (ChessColor::Black, Role::Queen) => "♛",
        (ChessColor::Black, Role::Rook) => "♜",
        (ChessColor::Black, Role::Bishop) => "♝",
        (ChessColor::Black, Role::Knight) => "♞",
        (ChessColor::Black, Role::Pawn) => "♟",
    }
}

/// Convert square to pixel rect given top-left corner of the 8×8 square grid (sq_origin).
pub fn square_rect(sq: Square, sq_origin: Pos2, flipped: bool) -> Rect {
    let (col, row) = if flipped {
        (7 - sq.file() as usize, sq.rank() as usize)
    } else {
        (sq.file() as usize, 7 - sq.rank() as usize)
    };
    let x = sq_origin.x + col as f32 * SQUARE_SIZE;
    let y = sq_origin.y + row as f32 * SQUARE_SIZE;
    Rect::from_min_size(Pos2::new(x, y), Vec2::splat(SQUARE_SIZE))
}

/// Convert pixel position to square given top-left corner of the 8×8 square grid (sq_origin).
pub fn pixel_to_square(pos: Pos2, sq_origin: Pos2, flipped: bool) -> Option<Square> {
    let rel_x = pos.x - sq_origin.x;
    let rel_y = pos.y - sq_origin.y;
    if rel_x < 0.0 || rel_y < 0.0 || rel_x >= INNER_SIZE || rel_y >= INNER_SIZE {
        return None;
    }
    let col = (rel_x / SQUARE_SIZE) as usize;
    let row = (rel_y / SQUARE_SIZE) as usize;
    let (file_idx, rank_idx) = if flipped {
        (7 - col, row)
    } else {
        (col, 7 - row)
    };
    let file = File::new(file_idx as u32);
    let rank = Rank::new(rank_idx as u32);
    Some(Square::from_coords(file, rank))
}

/// Returns the two squares to highlight for a last-move indicator.
/// For castling, returns the king and rook *destination* squares (not sources).
pub fn move_highlight_squares(mv: &Move) -> (Square, Square) {
    match mv {
        Move::Normal { from, to, .. } => (*from, *to),
        Move::EnPassant { from, to, .. } => (*from, *to),
        Move::Castle { king, rook } => {
            let rank = king.rank();
            if rook.file() > king.file() {
                // Kingside: king → g-file, rook → f-file
                (Square::from_coords(File::new(6), rank), Square::from_coords(File::new(5), rank))
            } else {
                // Queenside: king → c-file, rook → d-file
                (Square::from_coords(File::new(2), rank), Square::from_coords(File::new(3), rank))
            }
        }
        Move::Put { to, .. } => (*to, *to),
    }
}

pub fn draw_board(
    painter: &Painter,
    origin: Pos2,
    position: &shakmaty::Chess,
    selected: Option<Square>,
    legal_moves: &[Move],
    last_move: Option<&Move>,
    flipped: bool,
) {
    // sq_origin is the top-left of the actual 8×8 square grid, offset from the
    // board origin by the left rank-label margin.
    let sq_origin = Pos2::new(origin.x + LABEL_MARGIN, origin.y);

    // Draw squares
    for rank in 0..8u32 {
        for file in 0..8u32 {
            let sq = Square::from_coords(File::new(file), Rank::new(rank));
            let rect = square_rect(sq, sq_origin, flipped);
            let is_light = (file + rank) % 2 == 0;
            let mut color = if is_light { LIGHT } else { DARK };

            // Highlight last move
            if let Some(lm) = last_move {
                let (from, to) = move_highlight_squares(lm);
                if sq == from || sq == to {
                    color = blend(color, LAST_MOVE);
                }
            }

            painter.rect_filled(rect, CornerRadius::ZERO, color);
        }
    }

    // Selected square
    if let Some(sel) = selected {
        let rect = square_rect(sel, sq_origin, flipped);
        painter.rect_filled(rect, CornerRadius::ZERO, SELECTED);
    }

    // Legal move hints
    let hint_destinations: Vec<Square> = legal_moves
        .iter()
        .map(|m| match m {
            Move::Normal { to, .. } => *to,
            Move::EnPassant { to, .. } => *to,
            Move::Castle { king, .. } => {
                // Show king destination
                let turn = position.turn();
                if matches!(m, Move::Castle { rook, .. } if rook.file() > king.file()) {
                    if turn == shakmaty::Color::White { Square::G1 } else { Square::G8 }
                } else {
                    if turn == shakmaty::Color::White { Square::C1 } else { Square::C8 }
                }
            }
            Move::Put { to, .. } => *to,
        })
        .collect();

    for dest in hint_destinations {
        let rect = square_rect(dest, sq_origin, flipped);
        let center = rect.center();
        let board_piece = position.board().piece_at(dest);
        if board_piece.is_some() {
            // Enemy piece: draw ring
            painter.circle_stroke(center, SQUARE_SIZE * 0.42, Stroke::new(4.0, HINT));
        } else {
            // Empty: draw dot
            painter.circle_filled(center, SQUARE_SIZE * 0.18, HINT);
        }
    }

    // Draw pieces
    let board = position.board();
    for rank in 0..8u32 {
        for file in 0..8u32 {
            let sq = Square::from_coords(File::new(file), Rank::new(rank));
            if let Some(piece) = board.piece_at(sq) {
                let rect = square_rect(sq, sq_origin, flipped);
                let ch = piece_char(piece);
                let text_color = if piece.color == ChessColor::White {
                    Color32::WHITE
                } else {
                    Color32::BLACK
                };
                // Draw shadow/outline for white pieces for visibility
                if piece.color == ChessColor::White {
                    painter.text(
                        rect.center() + Vec2::new(1.0, 1.0),
                        egui::Align2::CENTER_CENTER,
                        ch,
                        FontId::proportional(52.0),
                        Color32::from_rgb(80, 80, 80),
                    );
                }
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    ch,
                    FontId::proportional(52.0),
                    text_color,
                );
            }
        }
    }

    // Coordinate labels — drawn in the margin strips so they never overlap pieces.
    // Rank numbers go in the left margin (x < sq_origin.x).
    // File letters go in the bottom margin (y > sq_origin.y + INNER_SIZE).
    let label_color = Color32::from_rgb(120, 80, 40);
    for i in 0..8usize {
        // Rank numbers: centred vertically in each row, in the left margin strip.
        let rank_idx = if flipped { i } else { 7 - i };
        let rank_label = (b'1' + rank_idx as u8) as char;
        let y = sq_origin.y + i as f32 * SQUARE_SIZE + SQUARE_SIZE * 0.5;
        painter.text(
            Pos2::new(origin.x + LABEL_MARGIN * 0.5, y),
            egui::Align2::CENTER_CENTER,
            rank_label.to_string(),
            FontId::proportional(11.0),
            label_color,
        );

        // File letters: centred horizontally in each column, in the bottom margin strip.
        let file_idx = if flipped { 7 - i } else { i };
        let file_label = (b'a' + file_idx as u8) as char;
        let x = sq_origin.x + i as f32 * SQUARE_SIZE + SQUARE_SIZE * 0.5;
        painter.text(
            Pos2::new(x, sq_origin.y + INNER_SIZE + LABEL_MARGIN * 0.5),
            egui::Align2::CENTER_CENTER,
            file_label.to_string(),
            FontId::proportional(11.0),
            label_color,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Pos2;
    use shakmaty::{Bitboard, Square};

    // ── Castle highlight squares ──────────────────────────────────────────────

    #[test]
    fn test_castle_highlight_kingside_white() {
        // Standard kingside castle for white: king e1→g1, rook h1→f1.
        let mv = Move::Castle { king: Square::E1, rook: Square::H1 };
        let (king_dest, rook_dest) = move_highlight_squares(&mv);
        assert_eq!(king_dest, Square::G1, "kingside: king should land on g1");
        assert_eq!(rook_dest, Square::F1, "kingside: rook should land on f1");
    }

    #[test]
    fn test_castle_highlight_queenside_white() {
        // Standard queenside castle for white: king e1→c1, rook a1→d1.
        let mv = Move::Castle { king: Square::E1, rook: Square::A1 };
        let (king_dest, rook_dest) = move_highlight_squares(&mv);
        assert_eq!(king_dest, Square::C1, "queenside: king should land on c1");
        assert_eq!(rook_dest, Square::D1, "queenside: rook should land on d1");
    }

    #[test]
    fn test_castle_highlight_kingside_black() {
        // Standard kingside castle for black: king e8→g8, rook h8→f8.
        let mv = Move::Castle { king: Square::E8, rook: Square::H8 };
        let (king_dest, rook_dest) = move_highlight_squares(&mv);
        assert_eq!(king_dest, Square::G8, "kingside black: king should land on g8");
        assert_eq!(rook_dest, Square::F8, "kingside black: rook should land on f8");
    }

    #[test]
    fn test_castle_highlight_queenside_black() {
        // Standard queenside castle for black: king e8→c8, rook a8→d8.
        let mv = Move::Castle { king: Square::E8, rook: Square::A8 };
        let (king_dest, rook_dest) = move_highlight_squares(&mv);
        assert_eq!(king_dest, Square::C8, "queenside black: king should land on c8");
        assert_eq!(rook_dest, Square::D8, "queenside black: rook should land on d8");
    }

    #[test]
    fn test_normal_move_highlight_unchanged() {
        let mv = Move::Normal {
            role: shakmaty::Role::Pawn,
            from: Square::E2,
            capture: None,
            to: Square::E4,
            promotion: None,
        };
        let (from, to) = move_highlight_squares(&mv);
        assert_eq!(from, Square::E2);
        assert_eq!(to, Square::E4);
    }

    fn origin() -> Pos2 {
        Pos2::new(0.0, 0.0)
    }

    #[test]
    fn test_a1_normal_orientation() {
        // a1: file 0, rank 0 → col=0, row=7 → bottom-left
        let rect = square_rect(Square::A1, origin(), false);
        assert_eq!(rect.min.x, 0.0);
        assert_eq!(rect.min.y, 7.0 * SQUARE_SIZE);
    }

    #[test]
    fn test_h8_normal_orientation() {
        // h8: file 7, rank 7 → col=7, row=0 → top-right
        let rect = square_rect(Square::H8, origin(), false);
        assert_eq!(rect.min.x, 7.0 * SQUARE_SIZE);
        assert_eq!(rect.min.y, 0.0);
    }

    #[test]
    fn test_a1_flipped() {
        // flipped: col = 7-0 = 7, row = 0 → top-right
        let rect = square_rect(Square::A1, origin(), true);
        assert_eq!(rect.min.x, 7.0 * SQUARE_SIZE);
        assert_eq!(rect.min.y, 0.0);
    }

    #[test]
    fn test_h8_flipped() {
        // flipped: col = 7-7 = 0, row = 7 → bottom-left
        let rect = square_rect(Square::H8, origin(), true);
        assert_eq!(rect.min.x, 0.0);
        assert_eq!(rect.min.y, 7.0 * SQUARE_SIZE);
    }

    #[test]
    fn test_round_trip_all_squares_normal() {
        for sq in Bitboard::FULL {
            let center = square_rect(sq, origin(), false).center();
            assert_eq!(
                pixel_to_square(center, origin(), false),
                Some(sq),
                "round-trip failed for {sq:?}"
            );
        }
    }

    #[test]
    fn test_round_trip_all_squares_flipped() {
        for sq in Bitboard::FULL {
            let center = square_rect(sq, origin(), true).center();
            assert_eq!(
                pixel_to_square(center, origin(), true),
                Some(sq),
                "round-trip failed (flipped) for {sq:?}"
            );
        }
    }

    #[test]
    fn test_out_of_bounds_returns_none() {
        let o = origin();
        assert_eq!(pixel_to_square(Pos2::new(-1.0, 0.0), o, false), None);
        assert_eq!(pixel_to_square(Pos2::new(0.0, -1.0), o, false), None);
        // INNER_SIZE is the exact boundary of the square grid.
        assert_eq!(pixel_to_square(Pos2::new(INNER_SIZE, 0.0), o, false), None);
        assert_eq!(pixel_to_square(Pos2::new(0.0, INNER_SIZE), o, false), None);
        // BOARD_SIZE (> INNER_SIZE) is also out of bounds.
        assert_eq!(pixel_to_square(Pos2::new(BOARD_SIZE, 0.0), o, false), None);
    }

    #[test]
    fn test_sq_origin_offset_matches_label_margin() {
        // square_rect and pixel_to_square take the *square-grid* origin (sq_origin),
        // which is LABEL_MARGIN pixels right of the full board origin.
        // Passing a board origin offset by LABEL_MARGIN must be equivalent to
        // passing sq_origin directly.
        let board_origin = Pos2::new(100.0, 50.0);
        let sq_origin = Pos2::new(board_origin.x + LABEL_MARGIN, board_origin.y);
        // Place a1 via sq_origin — should be bottom-left of the grid.
        let rect = square_rect(Square::A1, sq_origin, false);
        assert!((rect.min.x - sq_origin.x).abs() < f32::EPSILON,
            "a1 left edge must align with sq_origin.x");
        assert!((rect.min.y - (sq_origin.y + 7.0 * SQUARE_SIZE)).abs() < f32::EPSILON,
            "a1 top edge must be at row 7 below sq_origin");
        // Round-trip through pixel_to_square with the same sq_origin.
        let center = rect.center();
        assert_eq!(pixel_to_square(center, sq_origin, false), Some(Square::A1));
    }

    #[test]
    fn test_label_margin_layout_constants() {
        // Verify the margin constants are self-consistent and the grid fits inside BOARD_SIZE.
        assert!(LABEL_MARGIN > 0.0, "label margin must be positive");
        assert!((INNER_SIZE + LABEL_MARGIN - BOARD_SIZE).abs() < f32::EPSILON,
            "INNER_SIZE + LABEL_MARGIN must equal BOARD_SIZE");
        assert!(BOARD_SIZE > INNER_SIZE, "BOARD_SIZE must exceed the playable grid size");
    }
}

fn blend(base: Color32, overlay: Color32) -> Color32 {
    let a = overlay.a() as f32 / 255.0;
    Color32::from_rgb(
        (base.r() as f32 * (1.0 - a) + overlay.r() as f32 * a) as u8,
        (base.g() as f32 * (1.0 - a) + overlay.g() as f32 * a) as u8,
        (base.b() as f32 * (1.0 - a) + overlay.b() as f32 * a) as u8,
    )
}
