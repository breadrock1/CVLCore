extern crate cvlcore;
use cvlcore::api::capture::*;
use cvlcore::api::chain::*;
use cvlcore::errors::*;
use cvlcore::ui::window::*;

fn main() -> CaptureResult {
    let url_address = std::env::args()
        .last()
        .expect("RTSP address has not been passed!");

    let window_name = "CVLDetector Demo";
    let window = MainWindow::new(window_name);
    window.create_window();

    let mut vcap = CvlCapture::default();
    vcap.open_stream(url_address.as_str(), StreamSource::RtspStream)?;
    processing_stream(&mut vcap, &window);

    window.close_window();
    Ok(())
}

fn processing_stream(vcap: &mut CvlCapture, window: &MainWindow) {
    let mut own_chain = ChainProcessing::default();
    while let Ok(frame) = vcap.read_frame() {
        let precessing_result = own_chain
            .run_chain(frame)
            .grayscale()
            .canny()
            .append_frame()
            .reduce_abs()
            .vibrating();

        let chain_result = precessing_result.get_result();
        if chain_result.is_err() {
            println!("{}", chain_result.err().unwrap());
            continue;
        }

        let cvl_mat = chain_result.unwrap();
        window.show_frame(&cvl_mat);
        match window.wait_event() {
            WindowSignals::KeepProcessing => {}
            WindowSignals::CLoseApplication => break,
        }
    }
}
