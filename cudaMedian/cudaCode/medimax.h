#pragma once
//#include <vector>
//#include "iu/iucore.h"

//#define _flowCNN_scale_

/// Class to compute correlation score from features; also can return argmin flow and the mininum score (including deriv.).
class MediMax
{
public:
	MediMax();
	~MediMax();

    static int forward(float *d_vol, float *d_out_FS, int in, int ih, int iw, int numDisp );
    static int backward( float *d_vol, float *d_inGrad, float *d_outGrad0, int in, int ih, int iw, int numDisp );

private:
	// no copies!
	MediMax(MediMax const&);
	void operator=(MediMax const&);

	static unsigned int run;
};

unsigned int MediMax::run = 0;