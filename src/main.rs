//! # Automatic phase alignment
//!
//! Testbed for FFT-based offline automatic phase alignment, designed for correlated
//! signals that were processed with effects that cause different phasing for
//! different parameters.
//!
//! ## Usecases:
//!
//! - More-seamless crossfading between samples (see `MultiCrossfader`).
//! - Delta compression - i.e. use a lossy, high-channel-count format like Opus and
//!   only store a limited number of full-quality channels, with the intermediate
//!   samples being stored as a diff. If the samples are highly correlated this means
//!   that the full-quality channels can get a much higher proportion of the allocated
//!   bitrate for that block, with the delta channels being possible to serialise at a
//!   lower bitrate.

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

pub struct WeightStrategy {
    pub in_range: Range<f64>,
    pub out_range: Range<f64>,
    pub exp: f64,
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

pub struct Weights {
    pub freq: WeightStrategy,
    pub radius: WeightStrategy,
    // TODO: Implement this, same as for midpoints
    // channel_calc: ChannelCalc,
    /// The number of primary features to detect (if `None`, use all buckets)
    pub limit_features: Option<NonZeroUsize>,
}

pub enum RealignMask {
    /// Align each bucket based on its calculated weight (before feature limiting).
    /// Causes changes in timbre but can reduce the chance of phase alignment making
    /// phasing worse.
    Weighted,
    /// Only change the phase of the major features (see `Weights.limit_features`).
    FeaturesOnly,
    /// Align all buckets by the same amount. Should not affect the sound of the effect
    /// at all, but may cause unwanted artifacts when crossfading noisier effects.
    Transparent,
}

pub enum ChannelCalc {
    /// Only calculate the phasing midpoint based on a certain channel idx. Useful
    /// if some phasing on the side channel is acceptable.
    OneChannel(usize),
    /// Calculate the midpoint across all channels. Avoids introducing phasing
    /// between channels while still accounting for the phase of all channels.
    Combined,
    /// Calculate a separate midpoint for each channel. May introduce phasing
    /// between channels.
    Separate,
}

pub struct AlignConfig {
    pub weights: Weights,
    /// TODO: I think `Transparent` is really the only option that makes sense for this, it
    /// can probably be removed.
    pub realign_mask: RealignMask,
    pub midpoint_calc: ChannelCalc,
    /// If input channel count is 2, this specifies if mid/side or left/right processing should
    /// be used.
    pub process_mode: Option<TwoChannelProcessMode>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TwoChannelProcessMode {
    MidSide,
    Stereo,
}

/// Given a flat buffer of interleaved samples with a format given by `signal_spec`, align the
/// phase of each audio buffer, using the feature detection parameters in `weight_mode`.
///
/// This processes the buffer as mid/side rather than left/right to prevent unwanted panning
/// if the buffers end up more phased than before.
///
/// ### Possible improvements
///
/// - Maybe for each pair of buffers we can "slide" the phase (either in N linear steps, via
///   binary search, or some combination) between the midpoint and its unmodified phase, choosing
///   the option which minimises error. The number of permutations needed to do this perfectly
///   would be extremely high, though, as the error needs to be optimised between each pair of
///   neighbours (and each sample has two neighbours). This would reduce the chance of alignment
///   actually increasing the overall amount of phasing.
/// - We could probably try multiple different configs, checking the error and choosing the one
///   that works the best automatically. The process is surprisingly fast, so we probably have
///   the performance budget for it.
pub fn phase_align(
    concat_samples: &mut [f32],
    num_files: usize,
    signal_spec: SignalSpec,
    config: &AlignConfig,
) {
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum ChanCount {
        Stereo,
        MidSide,
        Count(usize),
    }

    struct ChanFmt {
        idx: usize,
        total: ChanCount,
    }

    impl std::fmt::Display for ChanFmt {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match (self.total, self.idx) {
                (ChanCount::Count(1), _) => write!(f, "mono"),
                (ChanCount::MidSide, 0) => write!(f, "mid "),
                (ChanCount::MidSide, 1) => write!(f, "side"),
                (ChanCount::Stereo, 0) => write!(f, "left "),
                (ChanCount::Stereo, 1) => write!(f, "right"),
                _ => write!(f, "ch{} ", self.idx),
            }
        }
    }

    #[derive(Debug, Clone)]
    enum Midpoints {
        OneChannel(usize, f64),
        Combined(usize, f64),
        Separate(Vec<f64>),
    }

    impl Midpoints {
        fn new(num_channels: usize, config: &ChannelCalc) -> Self {
            match config {
                ChannelCalc::OneChannel(chan) => Midpoints::OneChannel(*chan, 0.),
                ChannelCalc::Combined => Midpoints::Combined(num_channels, 0.),
                ChannelCalc::Separate => Midpoints::Separate(vec![0.; num_channels]),
            }
        }

        fn add(&mut self, chan: usize, amt: f64) {
            match self {
                Self::OneChannel(filter_chan, acc) if chan == *filter_chan => *acc += amt,
                Self::OneChannel(_, _) => {}
                Self::Combined(count, acc) => *acc += amt / *count as f64,
                Self::Separate(items) => items[chan] += amt / items.len() as f64,
            }
        }

        fn get(&self, chan: usize) -> f64 {
            match self {
                Self::OneChannel(_, amt) | Self::Combined(_, amt) => *amt,
                Self::Separate(items) => items[chan],
            }
        }
    }

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
    ///
    /// # Returns
    ///
    /// If configured to limit to the top N weights, will return a vector of indices to
    /// the top N weights.
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

            let max_weight = weights[indices[0]];

            for weight in weights.iter_mut() {
                *weight /= max_weight;
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

    fn apply_phase_offset<'a>(
        iter: impl Iterator<Item = (&'a mut Complex<f32>, &'a f64)>,
        offset: f32,
    ) {
        for (src_cartesian, weight) in iter {
            let (r, theta) = src_cartesian.to_polar();
            *src_cartesian = Complex::from_polar(r, theta + offset * *weight as f32);
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
    let fmt_chan_count = match (num_channels, config.process_mode) {
        (2, Some(TwoChannelProcessMode::MidSide)) => ChanCount::MidSide,
        (2, Some(TwoChannelProcessMode::Stereo)) => ChanCount::Stereo,
        (n, _) => ChanCount::Count(n),
    };
    let samples_per_file = concat_samples.len() / num_files;

    let samples_per_channel = samples_per_file / num_channels;

    if num_channels == 2 && matches!(config.process_mode, Some(TwoChannelProcessMode::MidSide)) {
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
    let mut filtered_weights = vec![0f64; samples_per_file];

    eprintln!("Weights:");

    for (chan_idx, (chan_weights, chan_weights_filtered)) in weights
        .chunks_exact_mut(samples_per_channel)
        .zip(filtered_weights.chunks_exact_mut(samples_per_channel))
        .enumerate()
    {
        let top_indices = calc_weights(
            fft_out_buf
                .chunks_exact(samples_per_channel)
                .skip(chan_idx)
                .step_by(num_channels),
            chan_weights,
            signal_spec,
            &config.weights,
        );

        let chan_name = ChanFmt {
            idx: chan_idx,
            total: fmt_chan_count,
        };

        eprint!("  {chan_name}");
        if let Some(top_indices) = top_indices {
            eprintln!();
            for i in top_indices {
                chan_weights_filtered[i] = chan_weights[i];
                let bin_freq = signal_spec.rate as f64 * i as f64 / samples_per_channel as f64;
                let weight = chan_weights[i];
                eprintln!("    {bin_freq:.0}Hz: {weight}");
            }
        } else {
            chan_weights_filtered.copy_from_slice(chan_weights);
            eprintln!(" (not printing as number of features is not limited)");
        }
    }

    let mut fft_chan_bufs = fft_out_buf.chunks_exact(samples_per_channel);

    let mut phases = Vec::with_capacity(num_files * num_channels);
    let mut midpoints = Midpoints::new(num_channels, &config.midpoint_calc);

    'outer: loop {
        for chan_idx in 0..num_channels {
            let Some(chan) = fft_chan_bufs.next() else {
                break 'outer;
            };
            let phase = calc_phase_offset(
                chan,
                &filtered_weights
                    [samples_per_channel * chan_idx..samples_per_channel * (chan_idx + 1)],
            );

            phases.push(phase);

            midpoints.add(chan_idx, phase);
        }
    }

    eprintln!("Average phase offset: {midpoints:?}");

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
                total: fmt_chan_count,
            };
            eprintln!("    {chan_name} {error}")
        }
    }

    let mut fft_chan_bufs = fft_out_buf.chunks_exact_mut(samples_per_channel);
    let mut phases = phases.into_iter();

    'outer: loop {
        for chan_idx in 0..num_channels {
            let midpoint = midpoints.get(chan_idx);
            let Some(chan) = fft_chan_bufs.next() else {
                break 'outer;
            };
            let phase = phases.next().unwrap();
            let phase_diff = midpoint - phase;

            match config.realign_mask {
                RealignMask::Weighted => {
                    let weights = &weights
                        [samples_per_channel * chan_idx..samples_per_channel * (chan_idx + 1)];

                    apply_phase_offset(chan.iter_mut().zip(weights), phase_diff as _)
                }
                RealignMask::FeaturesOnly => {
                    let weights = &filtered_weights
                        [samples_per_channel * chan_idx..samples_per_channel * (chan_idx + 1)];

                    apply_phase_offset(chan.iter_mut().zip(weights), phase_diff as _)
                }
                RealignMask::Transparent => {
                    apply_phase_offset(chan.iter_mut().zip(std::iter::repeat(&1.)), phase_diff as _)
                }
            };
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
                total: fmt_chan_count,
            };
            eprintln!("    {chan_name} {error} (delta: {delta}%)")
        }
    }

    if num_channels == 2 && matches!(config.process_mode, Some(TwoChannelProcessMode::MidSide)) {
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

        // Equal-gain crossfade as buffers are correlated
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
        &AlignConfig {
            weights: Weights {
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
            realign_mask: RealignMask::Transparent,
            midpoint_calc: ChannelCalc::OneChannel(0),
            process_mode: Some(TwoChannelProcessMode::MidSide),
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
