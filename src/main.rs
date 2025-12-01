use core::f32;
use std::fs::File;
use std::time::Duration;

use rodio::source::Source;
use rustfft::{FftPlanner, num_complex::Complex};
use symphonia::core::{
    audio::{Layout, SampleBuffer, SignalSpec},
    errors::Error,
    io::MediaSourceStream,
    probe::Hint,
};

/// AltarBoy loops have significant audible phasing without phase alignment,
/// so act as a good test for the process
const FILES: &[&str] = &[
    "Saturn Loop -100",
    "Saturn Loop -075",
    "Saturn Loop -050",
    "Saturn Loop -025",
    "Saturn Loop 000",
    "Saturn Loop +025",
    "Saturn Loop +050",
    "Saturn Loop +075",
    "Saturn Loop +100",
];

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
#[allow(dead_code)]
enum RadiusWeight {
    /// Weight = r^2, appears to produce the best results
    #[default]
    Pow,
    /// Weight = r
    Amp,
    /// Weight = 1
    One,
}

/// Given a flat buffer of interleaved samples with a format given by `signal_spec`, align the
/// phase of each audio buffer.
fn phase_align(
    concat_samples: &mut [f32],
    num_files: usize,
    signal_spec: SignalSpec,
    weight_mode: RadiusWeight,
) {
    fn calc_weights(
        fft: &[Complex<f32>],
        num_files: usize,
        sample_rate: u32,
        weight_mode: RadiusWeight,
        weights: &mut [f64],
    ) {
        for chan_buf in fft.chunks_exact(num_files) {
            for ((i, src), dst) in chan_buf.iter().enumerate().zip(&mut *weights).skip(1) {
                let bin_freq_seconds = fft.len() as f64 / (i as f64 * sample_rate as f64);

                let weight = match weight_mode {
                    RadiusWeight::Pow => src.norm_sqr() as f64,
                    RadiusWeight::Amp => src.norm() as f64,
                    RadiusWeight::One => 1.,
                };
                *dst = weight * bin_freq_seconds;
            }
        }
    }

    fn calc_phase_offset(fft: &[Complex<f32>], weights: &[f64]) -> f64 {
        let sum = fft
            .iter()
            .zip(weights)
            .map(|(cartesian, weight)| {
                let cartesian = Complex::new(cartesian.re as f64, cartesian.im as f64);
                let theta = cartesian.arg();

                // We calculate an average in cartesian space so -pi and pi don't average to 0
                Complex::from_polar(*weight, theta)
            })
            .sum::<Complex<f64>>();
        let (_, average_theta) = sum.to_polar();

        average_theta
    }

    fn apply_phase_offset(fft: &mut [Complex<f32>], offset: f64) {
        for src_cartesian in fft {
            let cartesian = Complex::new(src_cartesian.re as f64, src_cartesian.im as f64);
            let (r, theta) = cartesian.to_polar();

            let new_theta = theta + offset;

            if !new_theta.is_finite() {
                continue;
            }

            let new_cartesian = Complex::from_polar(r, new_theta);

            src_cartesian.re = new_cartesian.re as f32;
            src_cartesian.im = new_cartesian.im as f32;
        }
    }

    let num_channels = signal_spec.channels.count();
    let samples_per_file = concat_samples.len() / num_files;
    let samples_per_channel = samples_per_file / num_channels;
    let mut fft_in_buf = Vec::<Complex<f32>>::with_capacity(samples_per_channel);
    let mut fft_out_buf = vec![Complex::<f32>::default(); concat_samples.len()];

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(samples_per_channel);
    let ifft = planner.plan_fft_inverse(samples_per_channel);

    let mut scratch = vec![Complex::<f32>::default(); fft.get_inplace_scratch_len()];

    for file in 0..num_files {
        for channel in 0..num_channels {
            fft_in_buf.clear();
            fft_in_buf.extend(
                concat_samples[(samples_per_file * file)..(samples_per_file * (file + 1))]
                    .iter()
                    .skip(channel)
                    .step_by(num_channels)
                    .map(|sample| Complex::new(*sample, 0.)),
            );

            let out_range = (samples_per_channel * (file * num_channels + channel))
                ..(samples_per_channel * (file * num_channels + channel + 1));
            fft.process_outofplace_with_scratch(
                &mut fft_in_buf,
                &mut fft_out_buf[out_range],
                &mut scratch,
            );
        }
    }

    let mut weights = vec![0f64; samples_per_channel];

    calc_weights(
        &fft_out_buf,
        num_files,
        signal_spec.rate,
        weight_mode,
        &mut weights,
    );

    let mut fft_chan_bufs = fft_out_buf.chunks_exact(samples_per_channel);

    let mut phases = Vec::with_capacity(num_files);
    let mut offset_accumulator: f64 = 0.;

    loop {
        let Some(left) = fft_chan_bufs.next() else {
            break;
        };
        let right = fft_chan_bufs.next().unwrap();

        let left_phase = calc_phase_offset(left, &weights);
        let right_phase = calc_phase_offset(right, &weights);

        let avg_phase = (left_phase + right_phase) / 2.;

        phases.push(avg_phase);

        offset_accumulator += avg_phase;
    }

    let midpoint_phase_offset = offset_accumulator / num_files as f64;

    eprintln!("Average phase: {midpoint_phase_offset}");

    let mut fft_chan_bufs = fft_out_buf.chunks_exact_mut(samples_per_channel);
    let mut phases = phases.into_iter();

    loop {
        let Some(left) = fft_chan_bufs.next() else {
            break;
        };
        let right = fft_chan_bufs.next().unwrap();
        let phase = phases.next().unwrap();
        let phase_diff = midpoint_phase_offset - phase;

        apply_phase_offset(left, phase_diff);
        apply_phase_offset(right, phase_diff);
    }

    ifft.process_with_scratch(&mut fft_out_buf, &mut scratch);

    for file in 0..num_files {
        for channel in 0..num_channels {
            let out_range = (samples_per_channel * (file * num_channels + channel))
                ..(samples_per_channel * (file * num_channels + channel + 1));

            for (dst, src) in concat_samples
                [(samples_per_file * file)..(samples_per_file * (file + 1))]
                .iter_mut()
                .skip(channel)
                .step_by(num_channels)
                .zip(&fft_out_buf[out_range])
            {
                *dst = src.re;
            }
        }
    }
}

fn min_max(buf: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for sample in buf.iter() {
        min = min.min(*sample);
        max = max.max(*sample);
    }

    (min, max)
}

fn remap_range(samples: &mut [f32], out_min: f32, out_max: f32) {
    let (min, max) = min_max(&samples);

    for sample in samples {
        let remap_zero_one = (*sample - min) / (max - min);
        *sample = (remap_zero_one * (out_max - out_min)) + out_min;
    }
}

const AMT_LFO_PERIOD_SECS: f32 = 5.;
const AMT_LFO_FREQUENCY: f32 = AMT_LFO_PERIOD_SECS.recip();

/// An n-ary crossfader that applies an equal-gain crossfade between different
/// buffers. The buffers are presumed to be correlated, in order to ensure that
/// the equal-gain crossfade does not affect overall volume.
struct MultiCrossfader {
    concat_samples: Vec<f32>,
    signal_spec: SignalSpec,
    num_files: usize,
    cur_sample: usize,
}

impl Iterator for MultiCrossfader {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let max_file_idx = self.num_files - 1;

        let samples = &self.concat_samples;
        let file_len = samples.len() / self.num_files;
        let seconds_per_sample = (self.signal_spec.rate as f32).recip();
        let amt =
            ((self.cur_sample as f32 * seconds_per_sample * AMT_LFO_FREQUENCY).sin() + 1.) / 2.;
        let index = amt * max_file_idx as f32;
        let index_lo = index.floor() as usize;
        let index_hi = index.ceil() as usize;
        let cross_perc = index - index_lo as f32;
        let sample_indices = [index_lo, index_hi]
            .map(|i| i.min(max_file_idx) * file_len + self.cur_sample % file_len);
        let samples = sample_indices.map(|i| samples[i]);

        self.cur_sample = self.cur_sample + 1;

        // Equal-gain crossfade as buffers should be correlated
        Some(samples[0] * (1. - cross_perc) + samples[1] * cross_perc)
    }
}

impl Source for MultiCrossfader {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> rodio::ChannelCount {
        self.signal_spec.channels.count() as _
    }

    fn sample_rate(&self) -> rodio::SampleRate {
        self.signal_spec.rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

fn main() {
    let registry = symphonia::default::get_codecs();
    let probe = symphonia::default::get_probe();

    let files = FILES
        .iter()
        .map(|filename| File::open(format!("assets/{filename}.flac")).unwrap())
        .map(|file| MediaSourceStream::new(Box::new(file), Default::default()))
        .map(|stream| {
            probe
                .format(
                    &Hint::new().with_extension("flac"),
                    stream,
                    &Default::default(),
                    &Default::default(),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();

    let mut num_files: usize = 0;

    let spec = SignalSpec::new_with_layout(
        files[0].format.tracks()[0]
            .codec_params
            .sample_rate
            .unwrap(),
        Layout::Stereo,
    );
    let mut samples = Vec::<f32>::new();

    for mut file in files {
        let tracks = file.format.tracks().iter().cloned().collect::<Vec<_>>();
        for track in tracks {
            num_files += 1;

            let mut audio_buffer = SampleBuffer::new(
                track
                    .codec_params
                    .max_frames_per_packet
                    .or(track.codec_params.n_frames)
                    .unwrap(),
                spec,
            );
            let mut codec = registry
                .make(&track.codec_params, &Default::default())
                .unwrap();
            loop {
                match file.format.next_packet() {
                    Ok(packet) => {
                        let decoded = codec.decode(&packet).unwrap();
                        audio_buffer.copy_interleaved_ref(decoded);
                        samples.extend_from_slice(audio_buffer.samples());
                    }
                    Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        break;
                    }
                    Err(e) => panic!("Unknown error: {e}"),
                }
            }
        }
    }

    let samples_per_file = samples.len() / num_files;

    if let Err(e) = rodio::output_to_wav(
        &mut MultiCrossfader {
            concat_samples: samples.clone(),
            num_files,
            signal_spec: spec,
            cur_sample: 0,
        }
        .take_duration(Duration::from_secs_f32(AMT_LFO_PERIOD_SECS)),
        "crossfade-out-unaligned.wav",
    ) {
        eprintln!("Error while writing unaligned wav: {e}");
    }

    // Phase alignment can cause extreme volume changes for some reason. This is probably
    // resolvable without this hack, but for now we just calculate the min/max before
    // the phase alignment and then re-apply it afterwards
    let mins_maxs = samples
        .chunks_exact(samples_per_file)
        .map(|chunk| min_max(chunk))
        .collect::<Vec<_>>();

    phase_align(&mut samples, num_files, spec, RadiusWeight::Pow);

    // Remap each buffer back to its original range after phase alignment
    for (file, (min, max)) in samples.chunks_exact_mut(samples_per_file).zip(mins_maxs) {
        remap_range(file, min, max);
    }

    if let Err(e) = rodio::output_to_wav(
        &mut MultiCrossfader {
            concat_samples: samples.clone(),
            num_files,
            signal_spec: spec,
            cur_sample: 0,
        }
        .take_duration(Duration::from_secs_f32(AMT_LFO_PERIOD_SECS)),
        "crossfade-out-aligned.wav",
    ) {
        eprintln!("Error while writing aligned wav: {e}");
    }

    // --- Rodio audio output ---

    let source = MultiCrossfader {
        concat_samples: samples,
        num_files,
        signal_spec: spec,
        cur_sample: 0,
    };

    let stream_handle =
        rodio::OutputStreamBuilder::open_default_stream().expect("open default audio stream");
    let sink = rodio::Sink::connect_new(&stream_handle.mixer());

    sink.append(source.automatic_gain_control(0.8, 0.01, 0.1, 0.99));
    sink.sleep_until_end();
}
