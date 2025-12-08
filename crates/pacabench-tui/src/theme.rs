//! Cyberpunk color theme and styling
//!
//! A neon-on-dark aesthetic inspired by retro-futuristic terminals.

#![allow(dead_code)]

use ratatui::style::{Color, Modifier, Style};

// ═══════════════════════════════════════════════════════════════════════════════
// COLORS - Neon Cyberpunk Palette
// ═══════════════════════════════════════════════════════════════════════════════

/// Primary neon cyan - the signature accent
pub const NEON_CYAN: Color = Color::Rgb(0, 255, 255);

/// Hot magenta for warnings and highlights
pub const NEON_MAGENTA: Color = Color::Rgb(255, 0, 128);

/// Electric yellow for attention-grabbing elements
pub const NEON_YELLOW: Color = Color::Rgb(255, 255, 0);

/// Toxic green for success states
pub const NEON_GREEN: Color = Color::Rgb(57, 255, 20);

/// Alarm red for errors and failures
pub const NEON_RED: Color = Color::Rgb(255, 50, 50);

/// Electric orange for in-progress states
pub const NEON_ORANGE: Color = Color::Rgb(255, 165, 0);

/// Deep background - almost black
pub const BG_DEEP: Color = Color::Rgb(8, 8, 12);

/// Panel background - slightly lighter
pub const BG_PANEL: Color = Color::Rgb(16, 16, 24);

/// Surface background for elevated elements
pub const BG_SURFACE: Color = Color::Rgb(24, 24, 36);

/// Highlight background
pub const BG_HIGHLIGHT: Color = Color::Rgb(32, 32, 48);

/// Muted text for secondary information
pub const TEXT_MUTED: Color = Color::Rgb(90, 90, 110);

/// Dim text for tertiary information
pub const TEXT_DIM: Color = Color::Rgb(60, 60, 75);

/// Primary text color
pub const TEXT_PRIMARY: Color = Color::Rgb(220, 220, 235);

/// Progress bar gradient colors
pub const GRADIENT_START: Color = Color::Rgb(0, 80, 120);
pub const GRADIENT_MID: Color = Color::Rgb(0, 180, 180);
pub const GRADIENT_END: Color = Color::Rgb(0, 255, 255);

// ═══════════════════════════════════════════════════════════════════════════════
// ANIMATION - Centralized timing and character sets
// ═══════════════════════════════════════════════════════════════════════════════

/// Animation context for consistent timing across all widgets
#[derive(Debug, Clone, Copy)]
pub struct Animation {
    frame: u64,
}

impl Animation {
    pub fn new(frame: u64) -> Self {
        Self { frame }
    }

    /// Get spinner character (smooth rotation)
    /// Uses: ◐ ◓ ◑ ◒ (quarter turns)
    pub fn spinner(&self) -> char {
        const SPINNERS: [char; 4] = ['◐', '◓', '◑', '◒'];
        SPINNERS[(self.frame / 4) as usize % 4]
    }

    /// Get braille spinner (8 states, smoother)
    pub fn braille_spinner(&self) -> char {
        const BRAILLE: [char; 8] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];
        BRAILLE[(self.frame / 3) as usize % 8]
    }

    /// Get dot spinner for subtle animation
    pub fn dot_spinner(&self) -> char {
        const DOTS: [char; 4] = ['⣾', '⣽', '⣻', '⢿'];
        DOTS[(self.frame / 4) as usize % 4]
    }

    /// Pulsing opacity (0.0 to 1.0)
    pub fn pulse(&self) -> f64 {
        let t = (self.frame as f64 / 20.0).sin();
        (t + 1.0) / 2.0 // normalize to 0-1
    }

    /// Blink state (on/off)
    pub fn blink(&self) -> bool {
        ((self.frame / 15) & 1) == 0
    }

    /// Progress bar animation offset for "flowing" effect
    pub fn flow_offset(&self) -> usize {
        (self.frame / 2) as usize % 4
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THEME - Style configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Theme configuration for the entire UI
#[derive(Debug, Clone)]
pub struct Theme {
    pub background: Style,
    pub border: Style,
    pub border_focused: Style,
    pub border_active: Style,
    pub title: Style,
    pub text: Style,
    pub text_muted: Style,
    pub text_dim: Style,
    pub success: Style,
    pub error: Style,
    pub warning: Style,
    pub progress: Style,
    pub highlight: Style,
    pub status_bar: Style,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            background: Style::default().bg(BG_DEEP),
            border: Style::default().fg(TEXT_DIM),
            border_focused: Style::default().fg(NEON_CYAN),
            border_active: Style::default().fg(NEON_GREEN).add_modifier(Modifier::BOLD),
            title: Style::default().fg(NEON_CYAN).add_modifier(Modifier::BOLD),
            text: Style::default().fg(TEXT_PRIMARY),
            text_muted: Style::default().fg(TEXT_MUTED),
            text_dim: Style::default().fg(TEXT_DIM),
            success: Style::default().fg(NEON_GREEN),
            error: Style::default().fg(NEON_RED),
            warning: Style::default().fg(NEON_YELLOW),
            progress: Style::default().fg(NEON_ORANGE),
            highlight: Style::default()
                .fg(NEON_MAGENTA)
                .add_modifier(Modifier::BOLD),
            status_bar: Style::default().bg(BG_SURFACE).fg(TEXT_PRIMARY),
        }
    }
}

impl Theme {
    /// Get style for a given accuracy percentage
    pub fn accuracy_style(&self, accuracy: f64) -> Style {
        if accuracy >= 0.9 {
            self.success
        } else if accuracy >= 0.7 {
            self.warning
        } else {
            self.error
        }
    }

    /// Get a color for progress percentage (gradient)
    pub fn progress_color(&self, pct: f64) -> Color {
        let pct = pct.clamp(0.0, 1.0);
        if pct < 0.5 {
            let t = pct * 2.0;
            interpolate_color(GRADIENT_START, GRADIENT_MID, t)
        } else {
            let t = (pct - 0.5) * 2.0;
            interpolate_color(GRADIENT_MID, GRADIENT_END, t)
        }
    }
}

fn interpolate_color(a: Color, b: Color, t: f64) -> Color {
    let (r1, g1, b1) = color_to_rgb(a);
    let (r2, g2, b2) = color_to_rgb(b);

    let r = (r1 as f64 + (r2 as f64 - r1 as f64) * t) as u8;
    let g = (g1 as f64 + (g2 as f64 - g1 as f64) * t) as u8;
    let b_val = (b1 as f64 + (b2 as f64 - b1 as f64) * t) as u8;

    Color::Rgb(r, g, b_val)
}

fn color_to_rgb(c: Color) -> (u8, u8, u8) {
    match c {
        Color::Rgb(r, g, b) => (r, g, b),
        _ => (128, 128, 128),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHARACTERS - Unicode symbols and box drawing
// ═══════════════════════════════════════════════════════════════════════════════

pub mod chars {
    /// Progress bar characters
    pub const PROGRESS_FULL: &str = "█";
    pub const PROGRESS_HIGH: &str = "▓";
    pub const PROGRESS_MED: &str = "▒";
    pub const PROGRESS_LOW: &str = "░";

    /// Sparkline characters (8 levels)
    pub const SPARK: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    /// Status indicators
    pub const CHECK: &str = "✓";
    pub const CROSS: &str = "✗";
    pub const DOT_FILLED: &str = "●";
    pub const DOT_EMPTY: &str = "○";
    pub const WARN: &str = "⚠";

    /// Decorative
    pub const LIGHTNING: &str = "⚡";
    pub const ARROW_RIGHT: &str = "▶";
    pub const CHEVRON: &str = "»";
    pub const SEPARATOR: &str = "│";
    pub const ELLIPSIS: &str = "…";
}

// ═══════════════════════════════════════════════════════════════════════════════
// ASCII ART - Logos and banners
// ═══════════════════════════════════════════════════════════════════════════════

/// Main ASCII art logo (fits in ~75 chars width)
pub const LOGO_LARGE: [&str; 6] = [
    r"██████╗  █████╗  ██████╗ █████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗",
    r"██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║",
    r"██████╔╝███████║██║     ███████║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║",
    r"██╔═══╝ ██╔══██║██║     ██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║",
    r"██║     ██║  ██║╚██████╗██║  ██║██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║",
    r"╚═╝     ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝",
];

/// Compact logo for headers
pub const LOGO_COMPACT: &str = "▄▄ PACABENCH ▄▄";

/// Mini inline logo
pub const LOGO_MINI: &str = "⚡ PACABENCH";
