#ifndef RPP_MEL_SCALE_H
#define RPP_MEL_SCALE_H

struct BaseMelScale {
    public:
        virtual Rpp32f hz_to_mel(Rpp32f hz) = 0;
        virtual Rpp32f mel_to_hz(Rpp32f mel) = 0;
        virtual ~BaseMelScale() = default;
};

struct HtkMelScale : public BaseMelScale {
    Rpp32f hz_to_mel(Rpp32f hz) { return 1127.0f * std::log(1.0f + hz / 700.0f); }
    Rpp32f mel_to_hz(Rpp32f mel) { return 700.0f * (std::exp(mel / 1127.0f) - 1.0f); }
    public:
        ~HtkMelScale() {};
};

struct SlaneyMelScale : public BaseMelScale {
	const Rpp32f freq_low = 0;
	const Rpp32f fsp = 200.0 / 3.0;
	const Rpp32f min_log_hz = 1000.0;
	const Rpp32f min_log_mel = (min_log_hz - freq_low) / fsp;
	const Rpp32f step_log = 0.068751777;  // Equivalent to std::log(6.4) / 27.0;

    const Rpp32f inv_min_log_hz = 1.0f / 1000.0;
    const Rpp32f inv_step_log = 1.0f / step_log;
    const Rpp32f inv_fsp = 1.0f / fsp;

	Rpp32f hz_to_mel(Rpp32f hz) {
		Rpp32f mel = 0.0f;
		if (hz >= min_log_hz)
		    mel = min_log_mel + std::log(hz *inv_min_log_hz) * inv_step_log;
        else
		    mel = (hz - freq_low) * inv_fsp;

		return mel;
	}

	Rpp32f mel_to_hz(Rpp32f mel) {
		Rpp32f hz = 0.0f;
		if (mel >= min_log_mel)
			hz = min_log_hz * std::exp(step_log * (mel - min_log_mel));
        else
			hz = freq_low + mel * fsp;
		return hz;
	}
    public:
        ~SlaneyMelScale() {};
};

#endif //RPP_MEL_SCALE_H