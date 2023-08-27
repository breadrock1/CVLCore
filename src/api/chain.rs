use crate::core::bounds::ColorBounds;
use crate::core::mat::CvlMat;
use crate::errors::*;
use crate::*;
use std::rc::Rc;

pub struct ProcessingSettings {
    pub frames_count: usize,
    pub neighbours: usize,
    pub window_size: usize,
    pub is_reduced_abs: bool,
    pub canny_ksize: usize,
    pub canny_sigma: f64,
    pub canny_is_l2: bool,
}

impl Default for ProcessingSettings {
    fn default() -> Self {
        ProcessingSettings {
            frames_count: 5,
            neighbours: 8,
            window_size: 2,
            is_reduced_abs: true,
            canny_ksize: 3,
            canny_sigma: 0.05,
            canny_is_l2: true,
        }
    }
}

pub struct ChainProcessing {
    result: ProcessingResult,
    frames: Vec<Rc<CvlMat>>,
    bounds: ColorBounds,
    settings: ProcessingSettings,
}

impl Default for ChainProcessing {
    fn default() -> Self {
        let proc_settings = ProcessingSettings::default();
        ChainProcessing {
            frames: Vec::with_capacity(proc_settings.frames_count),
            bounds: ColorBounds::default(),
            result: Ok(CvlMat::default()),
            settings: proc_settings,
        }
    }
}

impl ChainProcessing {
    pub fn new(proc_settings: ProcessingSettings) -> Self {
        ChainProcessing {
            frames: Vec::with_capacity(proc_settings.frames_count),
            bounds: ColorBounds::default(),
            result: Ok(CvlMat::default()),
            settings: proc_settings,
        }
    }

    pub fn set_frames(&mut self, mat_frames: &Vec<Rc<CvlMat>>) {
        let test = mat_frames.to_owned();
        let _ = &self.frames.extend(test);
    }

    pub fn settings(&mut self) -> &mut ProcessingSettings {
        &mut self.settings
    }

    pub fn run_chain(&mut self, mat: CvlMat) -> &mut Self {
        self.result = Ok(mat);
        self
    }

    pub fn grayscale(&mut self) -> &mut Self {
        self.result = match &self.result {
            Ok(res) => gen_grayscale_frame(res),
            Err(err) => {
                let msg = format!("Failed exec grayscale chain function: {}", err);
                Err(ProcessingError::GenGrayScale(msg))
            }
        };

        self
    }

    pub fn canny(&mut self) -> &mut Self {
        self.result = match &self.result {
            Ok(res) => gen_canny_frame_by_sigma(res, 3i32, 0.05, true),
            Err(err) => {
                let msg = format!("Failed exec canny chain function: {}", err);
                Err(ProcessingError::GenCanny(msg))
            }
        };

        self
    }

    pub fn append_frame(&mut self) -> &mut Self {
        self.result = match &self.result {
            Err(_) => Err(ProcessingError::GenAbs),
            Ok(res) => {
                let frame = res.to_owned();
                let _ = &self.frames.push(Rc::new(frame));
                Ok(CvlMat::default())
            }
        };

        self
    }

    pub fn reduce_abs(&mut self) -> &mut Self {
        let frames_count = &self.frames.len();
        if frames_count < &5 {
            self.result = Err(ProcessingError::GenAbs);
            return self;
        }

        self.result = match &self.result {
            Err(_) => Err(ProcessingError::GenAbs),
            Ok(_) => {
                let _ = &self.frames.remove(0);
                gen_abs_frame_reduce(&self.frames)
            }
        };

        self
    }

    pub fn abs_recursively(&mut self) -> &mut Self {
        self.result = match &self.result {
            Err(_) => Err(ProcessingError::GenAbs),
            Ok(_) => {
                let _ = &self.frames.remove(0);
                gen_abs_frame(&self.frames)
            }
        };

        self
    }

    pub fn vibrating(&mut self) -> &mut Self {
        self.result = match &self.result {
            Err(_) => Err(ProcessingError::GenAbs),
            Ok(res) => compute_vibration(res, 8, 2, &self.bounds),
        };

        self
    }

    pub fn get_result(&self) -> ChainResult {
        match &self.result {
            Ok(res) => Ok(res.to_owned()),
            Err(err) => {
                let msg = format!("Failed exec canny: {}", err);
                Err(ProcessingError::ComputeVibration(msg))
            }
        }
    }
}
