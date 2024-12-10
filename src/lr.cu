#include <string>
#include <lr.cuh>
#include <util.cuh>

using namespace std;

/*
 * ==================================
 * LearningRateSchedule
 * ==================================
 */
LearningRateSchedule& LearningRateSchedule::make(PyObject* lrsDict, double baseRate) {
	string type = pyDictGetString(lrsDict, "type");
	if (type == "default") {
		return *new LearningRateSchedule(baseRate, 0);
	} else {
		PyObject* paramsDict = PyDict_GetItemString(lrsDict, "params");
		double tgtFactor = pyDictGetFloat(paramsDict, "tgtFactor");
		double noiseStdev = pyDictGetFloat(paramsDict, "noiseStdev");
		if (type == "linear") {
			return *new LinearLRS(baseRate, tgtFactor, noiseStdev);
		} else if (type == "exp") {
			return *new ExpLRS(baseRate, tgtFactor, noiseStdev);
		} else if (type == "dexp") {
			double numSteps = pyDictGetInt(paramsDict, "numSteps");
			return *new DiscreteExpLRS(baseRate, tgtFactor, noiseStdev, numSteps);
		} else if (type == "jdexp") {
            double numSteps = pyDictGetInt(paramsDict, "numSteps");
            return *new JumpyDiscreteExpLRS(baseRate, tgtFactor, noiseStdev, numSteps);
        }
	}
	throw string("Unknown learning rate schedule type ") + type;
}

LearningRateSchedule::LearningRateSchedule(double baseRate, double noiseStdev)
    : _baseRate(baseRate), _noiseStdev(noiseStdev), _haveRandnSpare(false), _randnSpare(0) {
}

LearningRateSchedule::LearningRateSchedule(double baseRate)
    : _baseRate(baseRate), _noiseStdev(0), _haveRandnSpare(false), _randnSpare(0) {
}

double LearningRateSchedule::getRate(double progress) {
	return _noiseStdev > 0 ? _getRate(progress) * (1 + abs(randn()) * _noiseStdev)
	                       : _getRate(progress);
}

double LearningRateSchedule::_getRate(double progress) {
    return _baseRate;
}

inline double LearningRateSchedule::randn() {
    if (!_haveRandnSpare) {
        double T = 2 * 3.1415 * rand();
        double R = std::sqrt(-2 * std::log(rand()));
        _randnSpare = R * std::sin(T);
        _haveRandnSpare = true;
        return R * std::cos(T);
    }
    _haveRandnSpare = false;
    return _randnSpare;
}

// This should never generate zero
inline double LearningRateSchedule::rand() const {
    return double(1L + random()) / (1L + RAND_MAX);
}

inline double LearningRateSchedule::abs(double x) const {
    return x > 0 ? x : -x;
}

double LearningRateSchedule::getBaseRate() const {
	return _baseRate;
}

LearningRateSchedule::~LearningRateSchedule() {
}

/*
 * ==================================
 * LinearLRS
 * ==================================
 */
LinearLRS::LinearLRS(double baseRate, double tgtFactor, double noiseStdev)
: LearningRateSchedule(baseRate, noiseStdev) {
	_finalRate = baseRate / tgtFactor;
}

double LinearLRS::_getRate(double progress) {
	return _baseRate * (1 - progress) + _finalRate * progress;
}

/*
 * ==================================
 * ExpLRS
 * ==================================
 */
ExpLRS::ExpLRS(double baseRate, double tgtFactor, double noiseStdev)
: LearningRateSchedule(baseRate, noiseStdev) {
	double finalRate = baseRate / tgtFactor;
	_pow = baseRate == 0 ? 1 : (std::log(finalRate) / std::log(baseRate) - 1);
}

double ExpLRS::_getRate(double progress) {
	return std::pow(_baseRate, 1.0 + progress * _pow);
}

/*
 * ==================================
 * TanhLRS
 * ==================================
 */
TanhLRS::TanhLRS(double baseRate, double tgtFactor, double noiseStdev)
: LearningRateSchedule(baseRate, noiseStdev), _alpha(0), _beta(0) {
	if (baseRate > 0) {
		double finalRate = baseRate / tgtFactor;
		_beta = 0.5 * (baseRate + finalRate);
		_alpha = 2 * atanh((baseRate - finalRate) / (baseRate + finalRate));
	}
}

double TanhLRS::_getRate(double progress) {
	return _beta * (tanh(-_alpha * (progress - 0.5)) + 1.0);
}

/*
 * ==================================
 * DiscreteExpLRS
 * ==================================
 */
DiscreteExpLRS::DiscreteExpLRS(double baseRate, double tgtFactor, double noiseStdev, int numSteps)
: LearningRateSchedule(baseRate, noiseStdev) {
	ExpLRS elrs(baseRate, tgtFactor, 0);
	double finalRate = baseRate / tgtFactor;
	for (int i = 0; i < numSteps - 1; i++) {
		double progress = double(i) / (numSteps - 1);
		_rates.push_back(elrs._getRate(progress));
	}
	_rates.push_back(finalRate);
	//printf("initialized base %e, final %e, stpes %d\n", baseRate, finalRate, numSteps);
}

double DiscreteExpLRS::_getRate(double progress) {
	for (int i = 0; i < _rates.size(); ++i) {
		if (progress <= double(i + 1) / _rates.size()) {
			return _rates[i];
		}
	}
	return _rates.back();
}

/*
 * ==================================
 * JumpyDiscreteExpLRS
 * ==================================
 */
JumpyDiscreteExpLRS::JumpyDiscreteExpLRS(double baseRate, double tgtFactor, double noiseStdev, int numSteps)
: DiscreteExpLRS(baseRate, tgtFactor, noiseStdev, numSteps) {
}

double JumpyDiscreteExpLRS::_getRate(double progress) {
    int rateIdx = 0;
    for (int i = 0; i < _rates.size(); ++i) {
        if (progress <= double(i + 1) / _rates.size()) {
            rateIdx = i;
            break;
        }
    }
    // The midpoint of the interval that progress falls into.
    double intervalMid = double(rateIdx + 0.5) / _rates.size();
    // Jumpy learning rate works like this:
    // If progress is before the midpoint of the current interval,
    //    it returns the same learning rate as would DiscreteExpLRS.
    // Else,
    //    it returns the learning rate of the *previous* interval (provided there is one).
//    rateIdx -= rateIdx > 0 && progress > 0.2 && progress < 0.9 && progress > intervalMid;

    // Uncomment this (and comment line above) to use variant 2:
    // Instead of using the learning rate of the previous interval, this uses
    // the geometric average of the learning rates of the current and previous
    // intervals.
    bool jump = rateIdx > 0 && progress > 0.2 && progress < 0.9 && progress > intervalMid;
    return jump ? sqrt(_rates[rateIdx] * _rates[rateIdx - 1]) : _rates[rateIdx];
//    return _rates[rateIdx];
}
