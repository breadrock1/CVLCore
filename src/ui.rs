use crate::core::cvl::CvlMat;
use opencv::highgui::{destroy_window, imshow, named_window, wait_key};
use opencv::highgui::{WINDOW_AUTOSIZE, WINDOW_GUI_NORMAL};

pub enum WindowSignals {
    CLoseApplication,
    KeepProcessing,
}

pub struct MainWindow {
    title: String,
    window_flags: i32,
}

impl MainWindow {
    pub fn new(title: &str) -> Self {
        MainWindow {
            title: title.to_string(),
            window_flags: WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL,
        }
    }

    pub fn create_window(&self) {
        let name = &self.title.as_str();
        named_window(name, self.window_flags).unwrap();
    }

    pub fn close_window(&self) {
        let name = &self.title.as_str();
        destroy_window(name).unwrap();
    }

    pub fn show_frame(&self, frame: &CvlMat) {
        let name = &self.title.as_str();
        imshow(name, frame.frame()).unwrap();
    }

    pub fn wait_event(&self) -> WindowSignals {
        match wait_key(10).unwrap() {
            113 => WindowSignals::CLoseApplication,
            _ => WindowSignals::KeepProcessing,
        }
    }
}
