use std::num::NonZeroUsize;
use std::ops::{Add, Div, Mul, Sub};
use std::time::Duration;
use std::{fs::File, ops::Range};

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
    "AltarBoy Loop -100",
    "AltarBoy Loop -075",
    "AltarBoy Loop -050",
    "AltarBoy Loop -025",
    "AltarBoy Loop 000",
    "AltarBoy Loop +025",
    "AltarBoy Loop +050",
    "AltarBoy Loop +075",
    "AltarBoy Loop +100",
];

struct WeightStrategy {
    in_range: Range<f64>,
    out_range: Range<f64>,
    exp: f64,
}

impl WeightStrategy {
    fn calc(&self, val: f64) -> f64 {
        if self.exp == 0. {
            return self.out_range.end;
        }

        let val = remap(val, self.in_range.clone(), 0f64..1f64).powf(self.exp);
        let bound_a = f64::EPSILON.powf(self.exp);
        let bound_b = 1f64.powf(self.exp);
        let range = bound_a.min(bound_b)..bound_a.max(bound_b);

        self.clamp(remap(val, range, self.out_range.clone()))
    }

    fn clamp(&self, val: f64) -> f64 {
        let lower = self.out_range.start.min(self.out_range.end);
        let higher = self.out_range.start.max(self.out_range.end);

        val.clamp(lower, higher)
    }
}

struct Weights {
    freq: WeightStrategy,
    radius: WeightStrategy,
    /// The number of primary features to detect (if `None`, use all buckets)
    limit_features: Option<NonZeroUsize>,
}

struct ChanFmt {
    idx: usize,
    total: usize,
}

impl std::fmt::Display for ChanFmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.total, self.idx) {
            (1, _) => write!(f, "mono"),
            (2, 0) => write!(f, "mid "),
            (2, 1) => write!(f, "side"),
            _ => write!(f, "ch{} ", self.idx),
        }
    }
}

/// Given a flat buffer of interleaved samples with a format given by `signal_spec`, align the
/// phase of each audio buffer, using the feature detection parameters in `weight_mode`.
///
/// This processes the buffer as mid/side rather than left/right to prevent unwanted panning
/// if the buffers end up more phased than before.
fn phase_align(
    concat_samples: &mut [f32],
    num_files: usize,
    signal_spec: SignalSpec,
    weight_mode: &Weights,
) {
    /// Either LR->MS or MS->LR, the function is its own inverse. Variable names
    /// are chosen based on LR->MS.
    fn mid_side_convert<'a>(frames: impl IntoIterator<Item = (&'a mut f32, &'a mut f32)>) {
        for (l, r) in frames {
            let mid = *l + *r;
            let side = *l - *r;

            *l = mid;
            *r = side;
        }
    }

    // fn rms_error_complex<'a>(
    //     vals: impl IntoIterator<Item = (&'a Complex<f32>, &'a Complex<f32>)>,
    // ) -> f32 {
    //     let Complex { re: sum, im: count } = vals
    //         .into_iter()
    //         .map(|(a, b)| Complex::new((a - b).norm_sqr(), 1.))
    //         .sum::<Complex<f32>>();
    //     (sum / count).sqrt()
    // }

    fn rms_error_scalar<'a>(
        vals: impl IntoIterator<Item = (&'a f32, &'a f32)>,
        range_left: &Range<f32>,
        range_right: &Range<f32>,
    ) -> f32 {
        let target_range = 0f32..1f32;

        let Complex { re: sum, im: count } = vals
            .into_iter()
            .map(|(a, b)| {
                let a = remap(*a, range_left.clone(), target_range.clone());
                let b = remap(*b, range_right.clone(), target_range.clone());
                // TODO: Somewhat janky way to
                Complex::new((a - b).powi(2), 1.)
            })
            .sum::<Complex<f32>>();
        (sum / count).sqrt()
    }

    /// Calculate weights, either for a single buffer or for a collection of buffers
    fn calc_weights<'a>(
        fft: impl IntoIterator<Item = &'a [Complex<f32>]>,
        weights: &mut [f64],
        signal_spec: SignalSpec,
        weight_mode: &Weights,
    ) -> Option<Vec<usize>> {
        let mut zeroed = false;

        for chan_buf in fft {
            for ((i, src), dst) in chan_buf.iter().enumerate().zip(&mut *weights).skip(1) {
                let bin_freq = signal_spec.rate as f64 * i as f64 / chan_buf.len() as f64;
                let freq_weight = weight_mode.freq.calc(bin_freq);
                let rad_weight = weight_mode.radius.calc(src.norm() as f64 * freq_weight);
                if zeroed {
                    *dst += rad_weight;
                } else {
                    *dst = rad_weight;
                }
            }

            zeroed = true;
        }

        if let Some(nth_highest) = weight_mode.limit_features
            && nth_highest.get() < weights.len()
        {
            let mut indices = (0..weights.len()).collect::<Vec<_>>();

            indices.sort_by(|a, b| {
                weights[*a]
                    .partial_cmp(&weights[*b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .reverse()
            });

            for i in &indices[nth_highest.get()..] {
                weights[*i] = 0.;
            }

            indices.truncate(nth_highest.get());

            Some(indices)
        } else {
            None
        }
    }

    fn calc_phase_offset(fft: &[Complex<f32>], weights: &[f64]) -> f64 {
        let sum = fft
            .iter()
            .zip(weights)
            .filter(|(_, weight)| **weight > 0.)
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

    fn apply_phase_offset(fft: &mut [Complex<f32>], offset: f32) {
        for src_cartesian in fft {
            let (r, theta) = src_cartesian.to_polar();
            *src_cartesian = Complex::from_polar(r, theta + offset);
        }
    }

    fn out_range(
        samples_per_channel: usize,
        num_channels: usize,
        file: usize,
        channel: usize,
    ) -> Range<usize> {
        (samples_per_channel * (file * num_channels + channel))
            ..(samples_per_channel * (file * num_channels + channel + 1))
    }

    fn get_channel(
        concat_samples: &[f32],
        samples_per_file: usize,
        num_channels: usize,
        file: usize,
        channel: usize,
    ) -> impl Iterator<Item = &'_ f32> {
        concat_samples[(samples_per_file * file)..(samples_per_file * (file + 1))]
            .iter()
            .skip(channel)
            .step_by(num_channels)
    }

    let num_channels = signal_spec.channels.count();
    let samples_per_file = concat_samples.len() / num_files;

    let samples_per_channel = samples_per_file / num_channels;

    if num_channels == 2 {
        for file in concat_samples.chunks_exact_mut(samples_per_file) {
            mid_side_convert(file.chunks_exact_mut(2).map(|chunk| {
                let (l, r) = chunk.split_at_mut(1);
                (&mut l[0], &mut r[0])
            }));
        }
    } else {
        eprintln!("Could not do mid/side conversion as input is not stereo");
    }

    // Phase alignment can cause extreme volume changes for some reason. This is probably
    // resolvable without this hack, but for now we just calculate the min/max before
    // the phase alignment and then re-apply it afterwards
    let mins_maxs = concat_samples
        .chunks_exact(samples_per_file)
        .map(min_max)
        .collect::<Vec<_>>();

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
                get_channel(
                    concat_samples,
                    samples_per_file,
                    num_channels,
                    file,
                    channel,
                )
                .map(|sample| Complex::new(*sample, 0.)),
            );

            let out_range = out_range(samples_per_channel, num_channels, file, channel);
            fft.process_outofplace_with_scratch(
                &mut fft_in_buf,
                &mut fft_out_buf[out_range],
                &mut scratch,
            );
        }
    }

    let mut weights = vec![0f64; samples_per_file];

    eprintln!("Weights:");

    for (chan_idx, chan_weights) in weights.chunks_exact_mut(samples_per_channel).enumerate() {
        let top_indices = calc_weights(
            fft_out_buf
                .chunks_exact(samples_per_channel)
                .skip(chan_idx)
                .step_by(num_channels),
            chan_weights,
            signal_spec,
            weight_mode,
        );

        let chan_name = ChanFmt {
            idx: chan_idx,
            total: num_channels,
        };

        eprint!("  {chan_name}");
        if let Some(top_indices) = top_indices {
            eprintln!();
            for i in top_indices {
                let bin_freq = signal_spec.rate as f64 * i as f64 / samples_per_channel as f64;
                let weight = chan_weights[i];
                eprintln!("    {bin_freq:.0}Hz: {weight}");
            }
        } else {
            eprintln!(" (not printing as number of features is not limited)");
        }
    }

    let mut fft_chan_bufs = fft_out_buf.chunks_exact(samples_per_channel);

    let mut phases = Vec::with_capacity(num_files * num_channels);
    let mut midpoints = vec![0.; num_channels];

    'outer: loop {
        for (chan_idx, midpoint) in midpoints.iter_mut().enumerate() {
            let Some(chan) = fft_chan_bufs.next() else {
                break 'outer;
            };
            let phase = calc_phase_offset(
                chan,
                &weights[samples_per_channel * chan_idx..samples_per_channel * (chan_idx + 1)],
            );

            phases.push(phase);

            *midpoint += phase / num_files as f64;
        }
    }

    eprintln!("Average phases: {midpoints:?}");

    eprintln!();
    eprintln!("Time-domain errors (before alignment):");

    let mut pre_errors = Vec::<f32>::with_capacity((num_files - 1) * num_channels);

    for file in 0..num_files - 1 {
        let (cur_min, cur_max) = mins_maxs[file];
        let (next_min, next_max) = mins_maxs[file + 1];
        let cur_range = cur_min..cur_max;
        let next_range = next_min..next_max;

        eprintln!("  f{}/f{}:", file, file + 1);
        for channel in 0..num_channels {
            let error = rms_error_scalar(
                get_channel(
                    concat_samples,
                    samples_per_file,
                    num_channels,
                    file,
                    channel,
                )
                .zip(get_channel(
                    concat_samples,
                    samples_per_file,
                    num_channels,
                    file + 1,
                    channel,
                )),
                &cur_range,
                &next_range,
            );
            pre_errors.push(error);
            let chan_name = ChanFmt {
                idx: channel,
                total: num_channels,
            };
            eprintln!("    {chan_name} {error}")
        }
    }

    let mut fft_chan_bufs = fft_out_buf.chunks_exact_mut(samples_per_channel);
    let mut phases = phases.into_iter();

    'outer: loop {
        for midpoint in &midpoints {
            let Some(chan) = fft_chan_bufs.next() else {
                break 'outer;
            };
            let phase = phases.next().unwrap();
            let phase_diff = midpoint - phase;

            apply_phase_offset(chan, phase_diff as _);
        }
    }

    ifft.process_with_scratch(&mut fft_out_buf, &mut scratch);

    for file in 0..num_files {
        for channel in 0..num_channels {
            let out_range = out_range(samples_per_channel, num_channels, file, channel);

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

    // Remap each buffer back to its original range after phase alignment
    for (file, (min, max)) in concat_samples
        .chunks_exact_mut(samples_per_file)
        .zip(&mins_maxs)
    {
        remap_samples(file, *min, *max);
    }

    eprintln!();
    eprintln!("Time-domain errors (after alignment):");

    let mut pre_errors = pre_errors.into_iter();

    for file in 0..num_files - 1 {
        let (cur_min, cur_max) = mins_maxs[file];
        let (next_min, next_max) = mins_maxs[file + 1];
        let cur_range = cur_min..cur_max;
        let next_range = next_min..next_max;

        eprintln!("  f{}/f{}", file, file + 1);
        for channel in 0..num_channels {
            let error = rms_error_scalar(
                get_channel(
                    concat_samples,
                    samples_per_file,
                    num_channels,
                    file,
                    channel,
                )
                .zip(get_channel(
                    concat_samples,
                    samples_per_file,
                    num_channels,
                    file + 1,
                    channel,
                )),
                &cur_range,
                &next_range,
            );
            let pre_error = pre_errors.next().unwrap();
            let delta = 100. * (error - pre_error) / pre_error;

            let chan_name = ChanFmt {
                idx: channel,
                total: num_channels,
            };
            eprintln!("    {chan_name} {error} (delta: {delta}%)")
        }
    }

    if num_channels == 2 {
        for file in concat_samples.chunks_exact_mut(samples_per_file) {
            mid_side_convert(file.chunks_exact_mut(2).map(|chunk| {
                let (l, r) = chunk.split_at_mut(1);
                (&mut l[0], &mut r[0])
            }));
        }
    } else {
        eprintln!("Could not do left/right conversion as input is not stereo");
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

fn remap<T>(val: T, from: Range<T>, to: Range<T>) -> T
where
    T: Copy + Sub<Output = T> + Div<Output = T> + Mul<Output = T> + Add<Output = T>,
{
    let remap_zero_one = (val - from.start) / (from.end - from.start);
    (remap_zero_one * (to.end - to.start)) + to.start
}

fn remap_samples(samples: &mut [f32], out_min: f32, out_max: f32) {
    let (min, max) = min_max(samples);

    for sample in samples {
        *sample = remap(*sample, min..max, out_min..out_max);
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

        self.cur_sample += 1;

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
                    Hint::new().with_extension("flac"),
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
        let tracks = file.format.tracks().to_vec();
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

    phase_align(
        &mut samples,
        num_files,
        spec,
        &Weights {
            freq: WeightStrategy {
                in_range: 0f64..48000f64,
                out_range: 0f64..1f64,
                exp: 1.,
            },
            radius: WeightStrategy {
                in_range: 0f64..4_196f64,
                out_range: 0f64..1f64,
                exp: 1.2,
            },
            limit_features: NonZeroUsize::new(8),
        },
    );

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
    let sink = rodio::Sink::connect_new(stream_handle.mixer());

    sink.append(source.automatic_gain_control(0.8, 0.01, 0.1, 0.99));
    sink.sleep_until_end();
}
