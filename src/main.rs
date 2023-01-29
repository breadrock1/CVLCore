// use anyhow::Result; // Automatically handle the error types
use opencv::{
    prelude::*,
    videoio::{VideoCapture, CAP_ANY},
    highgui::{destroy_window, named_window, imshow, wait_key, WINDOW_FULLSCREEN}
};

fn exit_app() {
    destroy_window("window").unwrap()
}

fn main() {
    named_window("window", WINDOW_FULLSCREEN).unwrap();

    let mut cam = VideoCapture::new(0, CAP_ANY).unwrap();
    let mut frame = Mat::default();

    loop {
        cam.read(&mut frame).unwrap();
        imshow("window", &frame).unwrap();

        match wait_key(1) {
            Ok(key) if key == 113 => {
                exit_app();
                break
            },
            _ => continue
        }
    }
}