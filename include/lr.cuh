#ifndef LR_CUH
#define	LR_CUH

#include <string>
#include <vector>
#include <iostream>
#include <helper_cuda.h>
#include <assert.h>
#include <nvmatrix.cuh>
#include <matrix.h>
#include <util.cuh>
#include <Python.h>

/*
 * The maximum learning rate is _baseRate.
 * The minimum learning rate is _baseRate / _tgtFactor.
 *
 * These classes define annealing schedules that interpolate between these
 * two extrema.
 */
class LearningRateSchedule {
protected:
	double _baseRate, _noiseStdev, _randnSpare;
	bool _haveRandnSpare;
	virtual double _getRate(double progress);
	double randn();
	double rand() const;
	double abs(double x) const;
public:
	LearningRateSchedule(double base);
	LearningRateSchedule(double base, double noiseStdev);
	double getRate(double progress);
	double getBaseRate() const;
	virtual ~LearningRateSchedule();

	static LearningRateSchedule& make(PyObject* lrsDict, double base);
};

class LinearLRS : public LearningRateSchedule {
protected:
	double _finalRate;
public:
	LinearLRS(double base, double tgtFactor, double noiseStdev);
	virtual double _getRate(double progress);
};

class ExpLRS : public LearningRateSchedule {
protected:
	double _pow;
public:
	ExpLRS(double baseRate, double tgtFactor, double noiseStdev);
	virtual double _getRate(double progress);
};

class TanhLRS : public LearningRateSchedule {
protected:
	double _alpha, _beta;
public:
	TanhLRS(double baseRate, double tgtFactor, double noiseStdev);
	virtual double _getRate(double progress);
};

class DiscreteExpLRS : public LearningRateSchedule {
protected:
	std::vector<double> _rates;
public:
	DiscreteExpLRS(double baseRate, double tgtFactor, double noiseStdev, int numSteps);
	virtual double _getRate(double progress);
};

class JumpyDiscreteExpLRS : public DiscreteExpLRS {
public:
	JumpyDiscreteExpLRS(double baseRate, double tgtFactor, double noiseStdev, int numSteps);
	virtual double _getRate(double progress);
};

#endif	/* LR_CUH */
