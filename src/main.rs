mod ai;
mod app;
mod board;
mod game;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("X-Chess")
            .with_inner_size([780.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "X-Chess",
        options,
        Box::new(|cc| Ok(Box::new(app::ChessApp::new(cc)))),
    )
}
