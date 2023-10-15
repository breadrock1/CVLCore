use crate::core::mat::CvlMat;
use thiserror::Error;

pub type ChainResult = Result<CvlMat, ProcessingError>;
pub type ProcessingResult = Result<CvlMat, ProcessingError>;

#[derive(Debug, Error)]
pub enum ProcessingError {
    #[error("Caught error while computing frame median values.")]
    ComputeMedian,
    #[error("Caught error while computing frame vibrating pixels.")]
    ComputeVibration(String),
    #[error("Caught error while processing abs() for passed Mats.")]
    GenAbs,
    #[error("Caught error while processing canny() for passed Mat.")]
    GenCanny(String),
    #[error("Caught error while processing difference for both Mats.")]
    GenDifferences(String),
    #[error("Caught error while processing distribution for passed Mat.")]
    GenDistribution(String),
    #[error("Caught error while transforming Mat to grayscale.")]
    GenGrayScale(String),
    #[error("Caught error while transforming Mat to threshold.")]
    GenThreshold(String),
    #[error("Caught error while transforming Mat to sobel.")]
    GenSobel(String),
    #[error("Caught error while computing statistics.")]
    ComputeStatistic,
}

pub type CaptureResult = Result<(), CaptureError>;

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("Caught error while opening video stream.")]
    OpenStream(String),
    #[error("Caught error while closing video stream.")]
    CloseStream,
    #[error("Not supported video stream source.")]
    UnsupportedSource,
}

pub type ReadFrameResult = Result<CvlMat, ReadFrameError>;

#[derive(Debug, Error)]
pub enum ReadFrameError {
    #[error("Caught error while reading next frame of stream.")]
    NextFrameError,
}
