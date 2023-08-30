/// A red color pixel value used for marking magnitude and vibration Mat object.
pub const RED_COLOR: (f64, f64, f64, f64) = (0.0, 0.0, 255.0, 0.0);

/// A cyan color pixel value used for marking magnitude and vibration Mat object.
pub const CYAN_COLOR: (f64, f64, f64, f64) = (255.0, 255.0, 0.0, 0.0);

/// A green color pixel value used for marking magnitude and vibration Mat object.
pub const GREEN_COLOR: (f64, f64, f64, f64) = (0.0, 255.0, 0.0, 0.0);

/// A yellow color pixel value used for marking magnitude and vibration Mat object.
pub const YELLOW_COLOR: (f64, f64, f64, f64) = (0.0, 255.0, 255.0, 0.0);

/// A black color pixel value used for marking magnitude and vibration Mat object.
pub const BLACK_COLOR: (f64, f64, f64, f64) = (0.0, 0.0, 0.0, 0.0);

#[derive(Copy, Clone)]
pub struct ColorBounds {
    channel_1: i32,
    channel_2: i32,
    channel_3: i32,
    channel_4: i32,
}

impl ColorBounds {
    pub fn new(ch1: i32, ch2: i32, ch3: i32, ch4: i32) -> Self {
        ColorBounds {
            channel_1: ch1,
            channel_2: ch2,
            channel_3: ch3,
            channel_4: ch4,
        }
    }

    pub fn get(&self, index: i32) -> i32 {
        match index {
            1 => self.channel_1,
            2 => self.channel_2,
            3 => self.channel_3,
            4 => self.channel_4,
            _ => 0,
        }
    }
}

impl Default for ColorBounds {
    fn default() -> Self {
        ColorBounds {
            channel_1: 8,
            channel_2: 9,
            channel_3: 10,
            channel_4: 11,
        }
    }
}
