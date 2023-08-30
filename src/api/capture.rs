use crate::core::mat::CvlMat;
use crate::errors::{CaptureError, CaptureResult, ReadFrameError, ReadFrameResult};
use opencv::core::Mat;
use opencv::hub_prelude::VideoCaptureTrait;
use opencv::videoio::{VideoCapture, CAP_ANY};
use std::str::FromStr;

pub enum StreamSource {
    VideoFile,
    WebCamera,
    RtspStream,
}

pub struct CvlCapture {
    capture: VideoCapture,
    api: i32,
}

impl CvlCapture {
    pub fn new() -> Self {
        CvlCapture::default()
    }

    pub fn open_stream(&mut self, address: &str, source_type: StreamSource) -> CaptureResult {
        let vcap = &mut self.capture;
        let open_result = match source_type {
            StreamSource::VideoFile => vcap.open_file(address, self.api),
            StreamSource::RtspStream => vcap.open_file(address, self.api),
            StreamSource::WebCamera => {
                match i32::from_str(address) {
                    Ok(port) => vcap.open(port, self.api),
                    Err(_) => Ok(false),
                }
            }
        };

        match open_result {
            Ok(_) => Ok(()),
            Err(err) => {
                let msg = format!("Failed open passed file {}: {}", address, err);
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
