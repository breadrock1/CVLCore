use crate::core::cvl::CvlMat;
use crate::errors::{CaptureError, CaptureResult, ReadFrameError, ReadFrameResult};
use opencv::core::Mat;
use opencv::hub_prelude::VideoCaptureTrait;
use opencv::videoio::{VideoCapture, CAP_ANY};
use std::str::FromStr;

pub struct CvlCapture {
    capture: VideoCapture,
    api: i32,
}

impl CvlCapture {
    pub fn new() -> Self {
        CvlCapture::default()
    }

    pub fn open_stream(&mut self, source: &str) -> CaptureResult {
        let vcap = &mut self.capture;
        let src_index = i32::from_str(source);
        if src_index.is_err() {
            let result = vcap.open_file(source, self.api);
            if !result.unwrap() {
                let msg = format!("Failed open passed file: {}", source);
                return Err(CaptureError::OpenStream(msg));
            }
            return Ok(());
        }

        let com_port = src_index.unwrap();
        match vcap.open(com_port, self.api).unwrap() {
            true => Ok(()),
            false => {
                let msg = format!("There is no such port: {}", com_port);
                Err(CaptureError::OpenStream(msg))
            }
        }
    }

    pub fn read_frame(&mut self) -> ReadFrameResult {
        let mut frame = Mat::default();
        match self.capture.read(&mut frame).unwrap() {
            false => Err(ReadFrameError::NextFrameError),
            true => Ok(CvlMat::from(frame)),
        }
    }

    pub fn close_stream(&mut self) -> CaptureResult {
        match self.capture.release() {
            Ok(_) => Ok(()),
            Err(_) => Err(CaptureError::CloseStream),
        }
    }
}

impl Default for CvlCapture {
    fn default() -> Self {
        let capture = VideoCapture::default().unwrap();
        CvlCapture {
            capture,
            api: CAP_ANY,
        }
    }
}
