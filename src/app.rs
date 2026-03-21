use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use egui::{Context, Sense, Vec2};
use shakmaty::{Color, Move, Position, Role};

use crate::ai;
use crate::board::{self, BOARD_SIZE};
use crate::game::{GameState, GameStatus};

pub struct ChessApp {
    state: GameState,
    game_generation: u64,
    tx: Sender<(u64, Option<Move>)>,
    rx: Receiver<(u64, Option<Move>)>,
}

impl ChessApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            state: GameState::new(),
            game_generation: 0,
            tx,
            rx,
        }
    }

    fn trigger_ai(&self, ctx: Context) {
        let pos = self.state.position.clone();
        let depth = self.state.ai_depth;
        let tx = self.tx.clone();
        let game_gen = self.game_generation;
        let game_history = self.state.position_hashes.clone();
        thread::spawn(move || {
            let mv = ai::best_move(&pos, depth, &game_history);
            let _ = tx.send((game_gen, mv));
            ctx.request_repaint();
        });
    }
}

impl eframe::App for ChessApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll AI response — discard moves from a previous game game_generation.
        if let Ok((game_gen, mv)) = self.rx.try_recv() {
            if game_gen == self.game_generation {
                if let Some(mv) = mv {
                    self.state.apply_move(mv); // last_move recorded inside apply_move
                }
                self.state.ai_thinking = false;
            }
        }

        // Side panel
        egui::SidePanel::right("info_panel").min_width(180.0).show(ctx, |ui| {
            ui.heading("Rust-Chess");
            ui.separator();

            // Status
            let status_text = match &self.state.status {
                GameStatus::Playing => {
                    let turn = if self.state.position.turn() == Color::White { "White" } else { "Black" };
                    format!("{}'s turn", turn)
                }
                GameStatus::Check => {
                    let turn = if self.state.position.turn() == Color::White { "White" } else { "Black" };
                    format!("{} is in Check!", turn)
                }
                GameStatus::Checkmate(winner) => {
                    let w = if *winner == Color::White { "White" } else { "Black" };
                    format!("Checkmate! {} wins!", w)
                }
                GameStatus::Stalemate => "Stalemate! Draw.".to_string(),
                GameStatus::Draw => "Draw!".to_string(),
            };
            ui.label(&status_text);

            if self.state.ai_thinking {
                ui.label("AI is thinking...");
            }

            ui.separator();

            if ui.button("New Game").clicked() {
                self.game_generation += 1;
                self.state.reset(); // last_move cleared inside reset()
                // Drain any queued moves that belong to the old game.
                while self.rx.try_recv().is_ok() {}
            }

            ui.separator();

            // Flip board
            if ui.checkbox(&mut self.state.flipped, "Flip Board").clicked() {}

            // AI depth
            ui.label("AI Depth:");
            ui.add(egui::Slider::new(&mut self.state.ai_depth, 1..=5));

            ui.separator();

            // Move history
            ui.label("Move History:");
            egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                for (i, mv) in self.state.move_history.iter().enumerate() {
                    let label = if i % 2 == 0 {
                        format!("{}. {}", i / 2 + 1, mv)
                    } else {
                        format!("   {}", mv)
                    };
                    ui.label(label);
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let board_size = BOARD_SIZE;
            let offset_x = (available.x - board_size) / 2.0;
            let offset_y = (available.y - board_size) / 2.0;
            let origin = ui.min_rect().min + Vec2::new(offset_x.max(0.0), offset_y.max(0.0));

            // Allocate board area with click sense.
            // inner_origin is the top-left of the 8×8 square grid (right of the rank-label margin).
            let inner_origin = origin + Vec2::new(board::LABEL_MARGIN, 0.0);
            let board_rect = egui::Rect::from_min_size(origin, Vec2::splat(board_size));
            let response = ui.allocate_rect(board_rect, Sense::click());

            // Draw board
            let painter = ui.painter();
            board::draw_board(
                painter,
                origin,
                &self.state.position,
                self.state.selected,
                &self.state.legal_moves_from,
                self.state.last_move.as_ref(),
                self.state.flipped,
            );

            // Handle clicks
            let is_human_turn = self.state.position.turn() == self.state.human_color;
            let can_interact = is_human_turn
                && !self.state.ai_thinking
                && !self.state.is_game_over()
                && self.state.pending_promotion.is_none();

            if can_interact {
                if let Some(click_pos) = response.interact_pointer_pos() {
                    if response.clicked() {
                        if let Some(sq) =
                            board::pixel_to_square(click_pos, inner_origin, self.state.flipped)
                        {
                            let history_len = self.state.move_history.len();
                            self.state.select_square(sq);

                            // Trigger AI only when a move was actually made.
                            if self.state.move_history.len() > history_len
                                && self.state.pending_promotion.is_none()
                                && !self.state.is_game_over()
                            {
                                self.state.ai_thinking = true;
                                self.trigger_ai(ctx.clone());
                            }
                        }
                    }
                }
            }

            // Promotion dialog
            if let Some((from, to)) = self.state.pending_promotion {
                let mut do_cancel = false;
                egui::Window::new("Promote Pawn")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.label("Choose promotion piece:");
                        ui.horizontal(|ui| {
                            for (label, role) in [
                                ("♕ Queen", Role::Queen),
                                ("♖ Rook", Role::Rook),
                                ("♗ Bishop", Role::Bishop),
                                ("♘ Knight", Role::Knight),
                            ] {
                                if ui.button(label).clicked() {
                                    self.state.apply_promotion(from, to, role);
                                    // Only trigger the AI when the promotion actually
                                    // succeeded (pending_promotion cleared).  If it
                                    // failed the dialog stays open and it is still the
                                    // human's turn — triggering the AI here would let
                                    // it play a move on the human's behalf.
                                    if self.state.pending_promotion.is_none()
                                        && !self.state.is_game_over()
                                    {
                                        self.state.ai_thinking = true;
                                        self.trigger_ai(ctx.clone());
                                    }
                                }
                            }
                            if ui.button("✕ Cancel").clicked() {
                                do_cancel = true;
                            }
                        });
                    });
                if do_cancel {
                    self.state.cancel_promotion();
                }
            }
        });
    }
}
