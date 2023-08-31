#[derive(Default, Clone, Debug)]
pub struct Statistic {
    pub ch1: u16,
    pub ch2: u16,
    pub ch3: u16,
    pub ch4: u16,
}

#[derive(Default, Clone, Debug)]
pub struct Dispersion {
    pub ch1: f32,
    pub ch2: f32,
    pub ch3: f32,
    pub ch4: f32,
}

impl Dispersion {
    pub fn new(ch1: f32, ch2: f32, ch3: f32, ch4: f32) -> Self {
        Dispersion { ch1, ch2, ch3, ch4 }
    }
}

impl From<Vec<f32>> for Dispersion {
    fn from(value: Vec<f32>) -> Self {
        Dispersion {
            ch1: *value.first().unwrap(),
            ch2: *value.get(1).unwrap(),
            ch3: *value.get(2).unwrap(),
            ch4: *value.get(3).unwrap(),
        }
    }
}
